import { showSidebarContent, showSidebar, setSidebarSticky } from './sidebar.js';
import { renderEvalImagesHtml } from './evalImages.js';

const galleryEl = document.getElementById('eval-gallery');
let lastRunsJson = null;

function formatScore(v) {
    if (v == null || isNaN(v)) return '—';
    return Number(v).toFixed(3);
}

function renderEvalGallery(runs) {
    if (!galleryEl) return;
    if (!runs || !runs.length) {
        galleryEl.innerHTML = '<p style="color:#888;">No eval runs yet. Evaluations appear here as they complete.</p>';
        return;
    }
    galleryEl.innerHTML = runs.map(run => {
        const images = run.eval_images || (run.prompts || []).map(p => ({
            prompt_index: p.prompt_index ?? 0,
            url: `/api/eval_image/${encodeURIComponent(run.run_id)}/${p.prompt_index ?? 0}`,
            original_prompt: p.original_prompt || '',
            alignment_score: p.alignment_score,
        }));
        const thumbs = images.slice(0, 3).map(img => {
            const alt = (img.original_prompt || '').slice(0, 40);
            return `<img src="${img.url}" alt="${alt}" loading="lazy" title="${alt}"/>`;
        }).join('');
        return `
        <div class="eval-card" data-run-id="${run.run_id}">
            <div class="eval-card-header">${run.run_id}<br><small>${run.program_id || ''}</small></div>
            <div class="eval-card-score">combined: ${formatScore(run.combined_score)}
                · align: ${formatScore(run.alignment_score)} · gemma: ${formatScore(run.gemma_score)}</div>
            <div class="eval-thumbs">${thumbs || '<span>No images</span>'}</div>
        </div>`;
    }).join('');

    galleryEl.querySelectorAll('.eval-card').forEach(card => {
        card.addEventListener('click', () => {
            const runId = card.getAttribute('data-run-id');
            const run = runs.find(r => r.run_id === runId);
            if (!run) return;
            const images = run.eval_images || (run.prompts || []).map(p => ({
                prompt_index: p.prompt_index ?? 0,
                url: `/api/eval_image/${encodeURIComponent(run.run_id)}/${p.prompt_index ?? 0}`,
                original_prompt: p.original_prompt || '',
                alignment_score: p.alignment_score,
                alignment_explanation: p.alignment_explanation || '',
            }));
            showSidebarContent({
                id: run.program_id || runId,
                generation: 0,
                island: 0,
                parent_id: 'None',
                metrics: {
                    combined_score: run.combined_score,
                    alignment_score: run.alignment_score,
                    gemma_score: run.gemma_score,
                },
                eval_images: images,
                artifacts_json: { run_id: runId },
            });
            setSidebarSticky(true);
            showSidebar();
        });
    });
}

function fetchEvalRuns() {
    return fetch('/api/eval_runs')
        .then(r => r.json())
        .then(payload => {
            const runs = payload.runs || [];
            window._cachedEvalRuns = runs;
            return runs;
        });
}

function refreshEvalGallery() {
    const cached = window._cachedEvalRuns;
    if (cached && cached.length) {
        const json = JSON.stringify(cached);
        if (json !== lastRunsJson) {
            lastRunsJson = json;
            renderEvalGallery(cached);
        }
    }
    fetchEvalRuns().then(runs => {
        const json = JSON.stringify(runs);
        if (json === lastRunsJson) return;
        lastRunsJson = json;
        renderEvalGallery(runs);
    }).catch(() => {
        if (galleryEl) galleryEl.innerHTML = '<p style="color:#c00;">Failed to load eval runs.</p>';
    });
}

window.refreshEvalGallery = refreshEvalGallery;

if (galleryEl) {
    refreshEvalGallery();
    setInterval(refreshEvalGallery, 2000);
}
