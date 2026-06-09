"""Pipelined eval: parallel SAP + serial FLUX render queue + parallel VL."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from openevolve_sap.exp_logging.live_logger import LiveEvolutionLogger
from openevolve_sap.render_queue import FluxRenderQueue, RenderJob
from openevolve_sap.sap_eval_settings import (
    get_sap_max_concurrent,
    get_vl_max_concurrent,
    use_pipeline_parallel_sap,
)


def run_pipelined_eval(
    *,
    prompts: list[str],
    index_map: dict[int, int],
    run_id: str,
    status_path: Path,
    run_dir: Path,
    api_key: str,
    eval_seeds: list[int],
    eval_steps: int,
    visualization_only: bool,
    live: LiveEvolutionLogger,
    render_queue: FluxRenderQueue,
    decompose_one: Callable[[int, str], dict | None],
    decompose_batch: Callable[[list[str]], dict[int, dict | None] | None],
    save_decomposition: Callable[[Path, str, dict], Path],
    prompt_dir_fn: Callable[[Path, int], Path],
    render_fn: Callable[..., list[Path]],
    score_fn: Callable[..., float | None],
    collect_vl: Callable[..., None],
    append_jsonl: Callable[[Path, dict], None],
    maybe_cleanup: Callable[[int, int], None],
) -> tuple[list[dict], list[dict], list[float], list[dict], list[dict]]:
    """
    Execute SAP → FLUX → VL pipeline with overlap.

    Returns: prompt_records, sampled_outputs, alignments, score_records, saved_images
    """
    n = len(prompts)
    sap_workers = max(1, get_sap_max_concurrent())
    vl_workers = max(1, get_vl_max_concurrent())

    prompt_records: list[dict] = []
    records_by_global: dict[int, dict] = {}
    sampled_outputs: list[dict] = []
    alignments: list[float] = []
    score_records: list[dict] = []
    saved_images: list[dict] = []

    sap_by_local: dict[int, dict | None] = {i: None for i in range(n)}

    with ThreadPoolExecutor(max_workers=sap_workers, thread_name_prefix="sap") as sap_pool:
        live.stage("pipeline", f"start {n} prompts", extra={"sap_workers": sap_workers, "vl_workers": vl_workers})

        batch_future: Future | None = None
        if not use_pipeline_parallel_sap():
            batch_future = sap_pool.submit(decompose_batch, prompts)

        sap_futures: dict[int, Future] = {}
        if use_pipeline_parallel_sap():
            live.stage("sap", f"parallel decompose x{n}")
            live.set_queue_depth(sap=n)
            for local_i, prompt in enumerate(prompts):
                sap_futures[local_i] = sap_pool.submit(decompose_one, local_i, prompt)
        elif batch_future is not None:
            live.stage("sap", f"batch decompose x{n}")
            live.set_queue_depth(sap=1)

        def _resolve_sap(local_i: int) -> dict | None:
            if use_pipeline_parallel_sap():
                live.set_queue_depth(sap=max(0, sum(1 for f in sap_futures.values() if not f.done()) - 1))
                return sap_futures[local_i].result()
            if batch_future is not None:
                if local_i == 0:
                    batch = batch_future.result()
                    if batch is not None:
                        sap_by_local.update(batch)
                        live.set_queue_depth(sap=0)
                        return batch.get(0)
                return sap_by_local.get(local_i)
            return decompose_one(local_i, prompts[local_i])

        render_job_ids: dict[int, int] = {}
        vl_futures: dict[int, Future] = {}

        with ThreadPoolExecutor(max_workers=vl_workers, thread_name_prefix="vl") as vl_pool:
            for local_i, prompt in enumerate(prompts):
                global_idx = index_map[local_i]
                sap_out = _resolve_sap(local_i)
                if sap_out is None:
                    append_jsonl(
                        status_path,
                        {
                            "event": "sap_parse_failed",
                            "run_id": run_id,
                            "prompt_index": global_idx,
                            "prompt": prompt,
                        },
                    )
                    live.error(f"SAP failed", prompt_index=global_idx)
                    continue

                sampled_outputs.append(sap_out)
                prompt_dir = prompt_dir_fn(run_dir, global_idx)
                decomp_path = save_decomposition(prompt_dir, prompt, sap_out)
                record = {
                    "prompt_index": global_idx,
                    "original_prompt": prompt,
                    "prompt_dir": str(prompt_dir),
                    "decomposition_path": str(decomp_path),
                    "images": [],
                    "score": None,
                    "alignment_score": None,
                }
                prompt_records.append(record)
                records_by_global[global_idx] = record
                append_jsonl(
                    status_path,
                    {
                        "event": "decomposition_saved",
                        "run_id": run_id,
                        "prompt_index": global_idx,
                        "prompt": prompt,
                        "decomposition_path": str(decomp_path),
                        "prompts_list": sap_out.get("prompts_list", []),
                        "switch_prompts_steps": sap_out.get("switch_prompts_steps", []),
                    },
                )
                live.stage("sap", "decompose done", prompt_index=global_idx)

                # Collect any completed VL scores while GPU queue advances
                done_vl = [idx for idx, fut in list(vl_futures.items()) if fut.done()]
                for idx in done_vl:
                    fut = vl_futures.pop(idx)
                    rec = records_by_global.get(idx)
                    if rec is None:
                        continue
                    p = rec["original_prompt"]
                    pdir = Path(rec["prompt_dir"])
                    dpath = Path(rec["decomposition_path"])
                    collect_vl(fut, idx, p, pdir, dpath, record=rec, alignments=alignments, score_records=score_records)
                    live.stage("vl", "score done", prompt_index=idx)

                live.set_queue_depth(vl=len(vl_futures), render=render_queue.depth)

                job_id = local_i
                render_job_ids[local_i] = job_id
                render_queue.submit(
                    RenderJob(
                        job_id=job_id,
                        idx=global_idx,
                        prompt=prompt,
                        sap_out=sap_out,
                        prompt_dir=prompt_dir,
                        run_id=run_id,
                        status_path=status_path,
                        record=record,
                        seeds=eval_seeds,
                        num_inference_steps=eval_steps,
                        render_fn=render_fn,
                    )
                )
                live.stage("render", "queued", prompt_index=global_idx, extra={"queue": render_queue.depth})

                if local_i > 0:
                    prev_local = local_i - 1
                    prev_global = index_map[prev_local]
                    prev_paths = render_queue.get(render_job_ids[prev_local])
                    live.stage("render", f"done {len(prev_paths)} images", prompt_index=prev_global)
                    maybe_cleanup(n, prev_local)

                    if prev_paths:
                        prev_rec = records_by_global.get(prev_global)
                        if prev_rec is not None:
                            prev_prompt = prev_rec["original_prompt"]
                            prev_dir = Path(prev_rec["prompt_dir"])
                            prev_decomp = Path(prev_rec["decomposition_path"])
                            saved_images.append(
                                {
                                    "prompt_index": prev_global,
                                    "prompt": prev_prompt,
                                    "prompt_dir": str(prev_dir),
                                    "decomposition_path": str(prev_decomp),
                                    "image_path": str(prev_paths[0]),
                                    "image_paths": [str(p) for p in prev_paths],
                                    "images": list(prev_rec["images"]),
                                }
                            )
                            if not visualization_only:
                                vl_futures[prev_global] = vl_pool.submit(
                                    score_fn,
                                    prev_global,
                                    prev_prompt,
                                    prev_paths,
                                    prev_dir,
                                    prev_decomp,
                                    api_key,
                                    run_id,
                                    status_path,
                                    prev_rec,
                                )
                                live.set_queue_depth(vl=len(vl_futures))
                                live.stage("vl", "scoring", prompt_index=prev_global)

            if n > 0 and render_job_ids:
                last_local = n - 1
                last_global = index_map[last_local]
                last_paths = render_queue.get(render_job_ids[last_local])
                live.stage("render", f"done {len(last_paths)} images", prompt_index=last_global)
                maybe_cleanup(n, last_local)

                if last_paths:
                    last_rec = records_by_global.get(last_global)
                    if last_rec is None:
                        last_rec = {}
                    last_prompt = last_rec.get("original_prompt", prompts[last_local])
                    last_dir = Path(last_rec.get("prompt_dir", prompt_dir_fn(run_dir, last_global)))
                    last_decomp = Path(last_rec.get("decomposition_path", last_dir / "decomposition.json"))
                    saved_images.append(
                        {
                            "prompt_index": last_global,
                            "prompt": last_prompt,
                            "prompt_dir": str(last_dir),
                            "decomposition_path": str(last_decomp),
                            "image_path": str(last_paths[0]),
                            "image_paths": [str(p) for p in last_paths],
                            "images": list(last_rec["images"]),
                        }
                    )
                    if not visualization_only:
                        vl_futures[last_global] = vl_pool.submit(
                            score_fn,
                            last_global,
                            last_prompt,
                            last_paths,
                            last_dir,
                            last_decomp,
                            api_key,
                            run_id,
                            status_path,
                            last_rec,
                        )
                        live.stage("vl", "scoring", prompt_index=last_global)

            if not visualization_only:
                live.set_queue_depth(vl=len(vl_futures))
                for idx, fut in list(vl_futures.items()):
                    rec = records_by_global.get(idx)
                    if rec is None:
                        continue
                    pdir = Path(rec["prompt_dir"])
                    dpath = Path(rec["decomposition_path"])
                    collect_vl(
                        fut,
                        idx,
                        rec["original_prompt"],
                        pdir,
                        dpath,
                        record=rec,
                        alignments=alignments,
                        score_records=score_records,
                    )
                    live.stage("vl", "score done", prompt_index=idx)

    live.set_queue_depth(render=0, vl=0, sap=0)
    return prompt_records, sampled_outputs, alignments, score_records, saved_images
