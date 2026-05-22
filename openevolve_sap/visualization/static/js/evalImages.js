/** Shared helpers for eval result images in sidebar, list, and gallery. */

export function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function parseArtifactsJson(artifacts) {
    if (!artifacts) return {};
    if (typeof artifacts === 'object') return artifacts;
    if (typeof artifacts === 'string') {
        try {
            const parsed = JSON.parse(artifacts);
            return parsed && typeof parsed === 'object' ? parsed : {};
        } catch (_) {
            return {};
        }
    }
    return {};
}

function extractRunIdFromArtifacts(artifacts) {
    artifacts = parseArtifactsJson(artifacts);
    if (!artifacts || typeof artifacts !== 'object') return null;
    const paths = [artifacts.manifest_path, artifacts.score_records_path];
    for (const p of paths) {
        if (!p) continue;
        const m = String(p).match(/eval_results\/([^/]+)\//);
        if (m) return m[1];
    }
    const rec = (artifacts.prompt_records || [])[0];
    if (rec?.prompt_dir) {
        const m = String(rec.prompt_dir).match(/eval_results\/([^/]+)\//);
        if (m) return m[1];
    }
    return artifacts.run_id || null;
}

function evalImagesFromArtifacts(artifacts) {
    const parsed = parseArtifactsJson(artifacts);
    const runId = extractRunIdFromArtifacts(parsed);
    if (!runId) return [];
    return (parsed.prompt_records || [])
        .filter(rec => rec.image_path || rec.images?.[0]?.image_path)
        .map(rec => {
            const idx = rec.prompt_index ?? 0;
            const score = rec.score || {};
            return {
                prompt_index: idx,
                url: `/api/eval_image/${encodeURIComponent(runId)}/${idx}`,
                original_prompt: rec.original_prompt || rec.prompt || '',
                alignment_score: rec.alignment_score ?? score['alignment score'] ?? score.alignment_score,
                alignment_explanation: score['alignment explanation'] || '',
            };
        });
}

export function getEvalImagesForNode(d) {
    if (!d) return [];
    if (d.eval_images && d.eval_images.length) return d.eval_images;
    const artifacts = parseArtifactsJson(d.artifacts_json);
    if (artifacts.prompt_records?.length) {
        return evalImagesFromArtifacts(artifacts);
    }
    const runId = artifacts.run_id;
    if (!runId) return [];
    const runs = window._cachedEvalRuns || [];
    const run = runs.find(r => r.run_id === runId || r.program_id === d.id);
    if (!run) return [];
    return (run.prompts || []).map(p => ({
        prompt_index: p.prompt_index ?? 0,
        url: `/api/eval_image/${encodeURIComponent(run.run_id)}/${p.prompt_index ?? 0}`,
        original_prompt: p.original_prompt || '',
        alignment_score: p.alignment_score,
        alignment_explanation: p.alignment_explanation || '',
    }));
}

export function renderEvalImagesHtml(images, { compact = false } = {}) {
    if (!images || !images.length) {
        return '<p class="eval-images-empty">No evaluation images for this program yet.</p>';
    }
    const cls = compact ? 'eval-image-thumb compact' : 'eval-image-thumb';
    return images.map(img => {
        const caption = escapeHtml((img.original_prompt || '').slice(0, 200));
        const align = img.alignment_score != null ? ` · align: ${Number(img.alignment_score).toFixed(2)}` : '';
        return `
        <div class="eval-image-block">
            <div class="eval-image-caption"><b>Prompt ${img.prompt_index}</b>${align}
            ${caption ? `<br><span class="eval-prompt-text">${caption}</span>` : ''}</div>
            <img class="${cls}" src="${img.url}" alt="prompt ${img.prompt_index}" loading="lazy"
                 onclick="window.open(this.src,'_blank')" title="Open full size"/>
        </div>`;
    }).join('');
}
