const state = {
    selectedExp: '',
    selectedSequence: '',
    channel: 'bf',
    nRows: 12,
    currentPage: 1,
    clickMode: 'nav', // 'nav', 'start1', 'end1', 'start2', 'end2'
    cells: [],
    septumCache: {}, // global_id -> { has_septum, start_aligned, end_aligned, has_septum_2, start_aligned_2, end_aligned_2, saveStatus }
    aiCache: {}     // global_id -> { aiStart, aiEnd, confidence, source } | null
};

document.addEventListener('DOMContentLoaded', () => {
    initEvents();
    loadExperiments();
});

function initEvents() {
    const expSelect = document.getElementById('experimentSelect');
    if (expSelect) {
        expSelect.addEventListener('change', async (e) => {
            state.selectedExp = e.target.value;
            await loadFilmsAndSequences(state.selectedExp);
        });
    }

    const seqSelect = document.getElementById('sequenceSelect');
    if (seqSelect) {
        seqSelect.addEventListener('change', async (e) => {
            state.selectedSequence = e.target.value;
            await loadSequenceCells(state.selectedExp, state.selectedSequence);
        });
    }

    const channelSelect = document.getElementById('channelSelect');
    if (channelSelect) {
        channelSelect.addEventListener('change', (e) => {
            state.channel = e.target.value;
            renderPage();
        });
    }

    const rowsSelect = document.getElementById('rowsPerPageSelect');
    if (rowsSelect) {
        rowsSelect.addEventListener('change', (e) => {
            state.nRows = parseInt(e.target.value, 10) || 12;
            state.currentPage = 1;
            renderPage();
        });
    }

    // Mode Selector Buttons
    const modeBtns = document.querySelectorAll('.mode-btn');
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode || 'nav';
            setClickMode(mode);
        });
    });

    // Pagination Controls
    const firstBtn = document.getElementById('firstPageBtn');
    if (firstBtn) {
        firstBtn.addEventListener('click', () => {
            if (state.currentPage > 1) {
                state.currentPage = 1;
                renderPage();
            }
        });
    }

    const prevBtn = document.getElementById('prevPageBtn');
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (state.currentPage > 1) {
                state.currentPage--;
                renderPage();
            }
        });
    }

    const nextBtn = document.getElementById('nextPageBtn');
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const maxPage = Math.max(1, Math.ceil(state.cells.length / state.nRows));
            if (state.currentPage < maxPage) {
                state.currentPage++;
                renderPage();
            }
        });
    }

    const lastBtn = document.getElementById('lastPageBtn');
    if (lastBtn) {
        lastBtn.addEventListener('click', () => {
            const maxPage = Math.max(1, Math.ceil(state.cells.length / state.nRows));
            if (state.currentPage < maxPage) {
                state.currentPage = maxPage;
                renderPage();
            }
        });
    }

    const pageSlider = document.getElementById('pageSlider');
    if (pageSlider) {
        pageSlider.addEventListener('input', (e) => {
            state.currentPage = parseInt(e.target.value, 10) || 1;
            renderPage();
        });
    }
}

function setClickMode(mode) {
    state.clickMode = mode;
    const modeBtns = document.querySelectorAll('.mode-btn');
    modeBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
}

async function loadExperiments() {
    try {
        const res = await fetch('/api/list_experiments');
        const data = await res.json();
        const exps = data.experiments || [];
        
        const expSelect = document.getElementById('experimentSelect');
        if (!expSelect) return;

        if (exps.length === 0) {
            expSelect.innerHTML = '<option value="">No experiments</option>';
            return;
        }

        expSelect.innerHTML = exps.map(e => `<option value="${e}">${e}</option>`).join('');
        state.selectedExp = exps[0];
        expSelect.value = state.selectedExp;

        await loadFilmsAndSequences(state.selectedExp);
    } catch (err) {
        console.error('Failed to load experiments:', err);
    }
}

async function loadFilmsAndSequences(exp) {
    try {
        const res = await fetch(`/api/list_films_and_sequences?experiment=${encodeURIComponent(exp)}`);
        const data = await res.json();
        const sequences = data.sequences || [];

        const seqSelect = document.getElementById('sequenceSelect');
        if (!seqSelect) return;

        if (sequences.length === 0) {
            seqSelect.innerHTML = '<option value="">No sequences</option>';
            state.selectedSequence = '';
            state.cells = [];
            renderPage();
            return;
        }

        seqSelect.innerHTML = sequences.map(s => `<option value="${s}">${s}</option>`).join('');
        state.selectedSequence = sequences[0];
        seqSelect.value = state.selectedSequence;

        await loadSequenceCells(exp, state.selectedSequence);
    } catch (err) {
        console.error('Failed to load sequences:', err);
    }
}

async function loadSequenceCells(exp, seq) {
    const grid = document.getElementById('boardGrid');
    if (grid) {
        grid.innerHTML = '<div class="empty-state">Loading global cell strips...</div>';
    }

    try {
        const res = await fetch(`/api/list_cells?experiment=${encodeURIComponent(exp)}&sequence=${encodeURIComponent(seq)}`);
        const data = await res.json();
        state.cells = data.cells || [];
        state.septumCache = {};
        state.aiCache = {};
        state.currentPage = 1;
        renderPage();
    } catch (err) {
        console.error('Failed to load sequence cells:', err);
        if (grid) {
            grid.innerHTML = '<div class="empty-state">Error loading cell data</div>';
        }
    }
}

async function fetchSeptumLabel(cell) {
    if (!cell.film || !cell.cell_id) return null;
    try {
        const res = await fetch(`/api/get_septum_label?experiment=${encodeURIComponent(state.selectedExp)}&film=${encodeURIComponent(cell.film)}&cell_id=${encodeURIComponent(cell.cell_id)}`);
        const data = await res.json();
        if (data.status === 'success' && data.data) {
            const d = data.data;
            const parseVal = (val) => (val !== null && val !== undefined && val !== '' && !isNaN(parseInt(val))) ? parseInt(val) : null;
            return {
                has_septum: !!d.has_septum,
                start_aligned: parseVal(d.start_aligned),
                end_aligned: parseVal(d.end_aligned),
                has_septum_2: !!d.has_septum_2,
                start_aligned_2: parseVal(d.start_aligned_2),
                end_aligned_2: parseVal(d.end_aligned_2),
                saveStatus: 'saved'
            };
        }
    } catch (err) {
        console.error(`Error loading septum for cell ${cell.global_id}:`, err);
    }
    return {
        has_septum: false,
        start_aligned: null,
        end_aligned: null,
        has_septum_2: false,
        start_aligned_2: null,
        end_aligned_2: null,
        saveStatus: 'saved'
    };
}

async function fetchCachedAiSuggestion(cell) {
    try {
        const url = `/api/get_septum_ai_cache?experiment=${encodeURIComponent(state.selectedExp)}` +
            `&sequence=${encodeURIComponent(state.selectedSequence)}` +
            `&global_cell_id=${encodeURIComponent(cell.global_id)}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.status === 'success' && data.cached && data.data) {
            const entry = data.data;
            const parseFrame = (obj) => (obj && obj.sequence_frame !== undefined && obj.sequence_frame !== null) ? parseInt(obj.sequence_frame, 10) : null;
            const aiStart = parseFrame(entry.suggested_start);
            const aiEnd = parseFrame(entry.suggested_end);
            const conf = entry.confidence || entry.peak_confidence || entry.state_confidence || null;
            return {
                aiStart,
                aiEnd,
                confidence: conf,
                source: data.source || 'cached'
            };
        }
    } catch (err) {
        console.error(`Error loading cached AI suggestion for ${cell.global_id}:`, err);
    }
    return null;
}

async function runLiveAiPrediction(cell) {
    const btn = document.getElementById(`runAiBtn_${cell.global_id}`);
    if (btn) {
        btn.disabled = true;
        btn.textContent = '🤖 Predicting...';
    }

    try {
        const body = {
            experiment: state.selectedExp,
            film: cell.film,
            cell_id: parseInt(cell.cell_id, 10),
            sequence: state.selectedSequence,
            global_cell_id: String(cell.global_id)
        };
        const res = await fetch('/api/predict_septum', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.status === 'success') {
            const parseFrame = (obj) => (obj && obj.sequence_frame !== undefined && obj.sequence_frame !== null) ? parseInt(obj.sequence_frame, 10) : null;
            const aiStart = parseFrame(data.suggested_start);
            const aiEnd = parseFrame(data.suggested_end);
            const conf = data.confidence || data.peak_confidence || null;

            state.aiCache[cell.global_id] = {
                aiStart,
                aiEnd,
                confidence: conf,
                source: 'live'
            };

            updateRowAiUI(cell.global_id);
            updateRowTileHighlights(cell.global_id);
        } else {
            alert(data.message || 'AI prediction failed');
        }
    } catch (err) {
        console.error(`Error running live AI prediction for ${cell.global_id}:`, err);
        alert(`AI Prediction error: ${err}`);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🤖 Run AI';
        }
    }
}

async function saveRowSeptum(cell) {
    const sData = state.septumCache[cell.global_id];
    if (!sData) return;

    sData.saveStatus = 'saving';
    updateRowStatusUI(cell.global_id);

    try {
        const body = {
            experiment: state.selectedExp,
            film: cell.film,
            cell_id: String(cell.cell_id),
            sequence: state.selectedSequence,
            global_cell_id: String(cell.global_id),
            has_septum: !!sData.has_septum,
            start_aligned: sData.start_aligned !== null ? sData.start_aligned : null,
            end_aligned: sData.end_aligned !== null ? sData.end_aligned : null,
            has_septum_2: !!sData.has_septum_2,
            start_aligned_2: sData.start_aligned_2 !== null ? sData.start_aligned_2 : null,
            end_aligned_2: sData.end_aligned_2 !== null ? sData.end_aligned_2 : null
        };

        const res = await fetch('/api/save_septum_label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const resData = await res.json();
        if (resData.status === 'success') {
            sData.saveStatus = 'saved';
        } else {
            sData.saveStatus = 'unsaved';
            console.error(`Save error for ${cell.global_id}:`, resData.message);
        }
    } catch (err) {
        sData.saveStatus = 'unsaved';
        console.error(`Save exception for ${cell.global_id}:`, err);
    }

    updateRowStatusUI(cell.global_id);
    updateRowTileHighlights(cell.global_id);
}

function updateRowStatusUI(globalId) {
    const sData = state.septumCache[globalId];
    if (!sData) return;

    const badge = document.getElementById(`statusBadge_${globalId}`);
    if (badge) {
        badge.className = `save-status-badge ${sData.saveStatus}`;
        if (sData.saveStatus === 'saved') badge.textContent = 'Saved ✓';
        else if (sData.saveStatus === 'saving') badge.textContent = 'Saving...';
        else badge.textContent = 'Unsaved ⚠';
    }

    // Update input boxes
    const st1 = document.getElementById(`start1_${globalId}`);
    if (st1) st1.value = sData.start_aligned !== null ? sData.start_aligned : '';

    const en1 = document.getElementById(`end1_${globalId}`);
    if (en1) en1.value = sData.end_aligned !== null ? sData.end_aligned : '';

    const cb1 = document.getElementById(`has1_${globalId}`);
    if (cb1) cb1.checked = !!sData.has_septum;

    const st2 = document.getElementById(`start2_${globalId}`);
    if (st2) st2.value = sData.start_aligned_2 !== null ? sData.start_aligned_2 : '';

    const en2 = document.getElementById(`end2_${globalId}`);
    if (en2) en2.value = sData.end_aligned_2 !== null ? sData.end_aligned_2 : '';

    const cb2 = document.getElementById(`has2_${globalId}`);
    if (cb2) cb2.checked = !!sData.has_septum_2;
}

function updateRowAiUI(globalId) {
    const aiData = state.aiCache[globalId];
    const aiBadgeContainer = document.getElementById(`aiBadgeContainer_${globalId}`);
    if (!aiBadgeContainer) return;

    if (aiData && (aiData.aiStart !== null || aiData.aiEnd !== null)) {
        const startStr = aiData.aiStart !== null ? `g${aiData.aiStart}` : '?';
        const endStr = aiData.aiEnd !== null ? `g${aiData.aiEnd}` : '?';
        const confStr = aiData.confidence !== null ? ` (${(aiData.confidence * 100).toFixed(0)}%)` : '';
        const sourceStr = aiData.source === 'live' ? ' (live)' : '';
        aiBadgeContainer.innerHTML = `<span class="ai-badge" title="AI Suggested Interval">🤖 AI: ${startStr}–${endStr}${confStr}${sourceStr}</span>`;
    } else {
        aiBadgeContainer.innerHTML = '';
    }
}

function updateRowTileHighlights(globalId) {
    const sData = state.septumCache[globalId];
    const aiData = state.aiCache[globalId];

    const container = document.getElementById(`stripContainer_${globalId}`);
    if (!container) return;

    const tiles = container.querySelectorAll('.strip-crop');
    tiles.forEach((tile) => {
        const t = parseInt(tile.dataset.t, 10);
        if (isNaN(t)) return;

        tile.className = 'strip-crop';

        // Human Label Highlights
        if (sData) {
            const has1 = sData.has_septum;
            const start1 = sData.start_aligned;
            const end1 = sData.end_aligned;

            if (has1) {
                if (start1 !== null && t === start1) tile.classList.add('septum-start-frame');
                if (end1 !== null && t === end1) tile.classList.add('septum-end-frame');
                if (start1 !== null && end1 !== null && t >= start1 && t <= end1) tile.classList.add('septum-during-frame');
            }

            const has2 = sData.has_septum_2;
            const start2 = sData.start_aligned_2;
            const end2 = sData.end_aligned_2;

            if (has2) {
                if (start2 !== null && t === start2) tile.classList.add('septum-start-frame-2');
                if (end2 !== null && t === end2) tile.classList.add('septum-end-frame-2');
                if (start2 !== null && end2 !== null && t >= start2 && t <= end2) tile.classList.add('septum-during-frame-2');
            }
        }

        // AI Suggestion Marker Overlays
        if (aiData) {
            const aiStart = aiData.aiStart;
            const aiEnd = aiData.aiEnd;

            if (aiStart !== null && t === aiStart) tile.classList.add('ai-start-marker');
            if (aiEnd !== null && t === aiEnd) tile.classList.add('ai-end-marker');
            if (aiStart !== null && aiEnd !== null && t >= aiStart && t <= aiEnd) tile.classList.add('ai-during-marker');
        }
    });
}

function renderPage() {
    const grid = document.getElementById('boardGrid');
    if (!grid) return;

    const totalCells = state.cells.length;
    const totalPages = Math.max(1, Math.ceil(totalCells / state.nRows));

    if (state.currentPage > totalPages) state.currentPage = totalPages;
    if (state.currentPage < 1) state.currentPage = 1;

    // Update Pagination UI
    const pageInfo = document.getElementById('pageInfo');
    if (pageInfo) pageInfo.textContent = `Page ${state.currentPage} of ${totalPages}`;

    const badge = document.getElementById('totalCellsBadge');
    if (badge) badge.textContent = `${totalCells} cells`;

    const firstBtn = document.getElementById('firstPageBtn');
    if (firstBtn) firstBtn.disabled = (state.currentPage === 1);

    const prevBtn = document.getElementById('prevPageBtn');
    if (prevBtn) prevBtn.disabled = (state.currentPage === 1);

    const nextBtn = document.getElementById('nextPageBtn');
    if (nextBtn) nextBtn.disabled = (state.currentPage === totalPages);

    const lastBtn = document.getElementById('lastPageBtn');
    if (lastBtn) lastBtn.disabled = (state.currentPage === totalPages);

    const pageSlider = document.getElementById('pageSlider');
    if (pageSlider) {
        pageSlider.min = '1';
        pageSlider.max = String(totalPages);
        pageSlider.value = String(state.currentPage);
    }

    if (totalCells === 0) {
        grid.innerHTML = '<div class="empty-state">No cells found in this sequence</div>';
        return;
    }

    const startIdx = (state.currentPage - 1) * state.nRows;
    const endIdx = startIdx + state.nRows;
    const pageCells = state.cells.slice(startIdx, endIdx);

    grid.innerHTML = '';
    const ts = Date.now();

    pageCells.forEach(async (cell) => {
        if (!state.septumCache[cell.global_id]) {
            state.septumCache[cell.global_id] = await fetchSeptumLabel(cell);
        }

        if (state.aiCache[cell.global_id] === undefined) {
            state.aiCache[cell.global_id] = await fetchCachedAiSuggestion(cell);
        }

        const row = document.createElement('div');
        row.className = 'cell-row';
        row.id = `cellRow_${cell.global_id}`;

        const sData = state.septumCache[cell.global_id] || {
            has_septum: false, start_aligned: null, end_aligned: null,
            has_septum_2: false, start_aligned_2: null, end_aligned_2: null,
            saveStatus: 'saved'
        };

        // Left Label & Controls Card
        const labelCard = document.createElement('div');
        labelCard.className = 'cell-label-card';

        const headerDiv = document.createElement('div');
        headerDiv.className = 'cell-card-header';

        const cellName = document.createElement('div');
        cellName.className = 'cell-name';
        cellName.textContent = cell.display_name || `Cell ${cell.global_id}`;

        const statusBadge = document.createElement('span');
        statusBadge.id = `statusBadge_${cell.global_id}`;
        statusBadge.className = `save-status-badge ${sData.saveStatus}`;
        statusBadge.textContent = sData.saveStatus === 'saved' ? 'Saved ✓' : (sData.saveStatus === 'saving' ? 'Saving...' : 'Unsaved ⚠');

        headerDiv.appendChild(cellName);
        headerDiv.appendChild(statusBadge);

        const subRowDiv = document.createElement('div');
        subRowDiv.className = 'cell-card-header';

        const cellSub = document.createElement('div');
        cellSub.className = 'cell-id-sub';
        cellSub.textContent = `ID: ${cell.global_id}`;

        const aiBadgeContainer = document.createElement('div');
        aiBadgeContainer.id = `aiBadgeContainer_${cell.global_id}`;

        subRowDiv.appendChild(cellSub);
        subRowDiv.appendChild(aiBadgeContainer);

        // Septum Controls
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'cell-septum-controls';

        // Septum 1 row
        const row1 = document.createElement('div');
        row1.className = 'septum-row-control';

        const lbl1 = document.createElement('label');
        const cb1 = document.createElement('input');
        cb1.type = 'checkbox';
        cb1.id = `has1_${cell.global_id}`;
        cb1.checked = !!sData.has_septum;
        cb1.addEventListener('change', () => {
            sData.has_septum = cb1.checked;
            saveRowSeptum(cell);
        });
        lbl1.appendChild(cb1);
        lbl1.appendChild(document.createTextNode(' Septum 1'));

        const inputs1 = document.createElement('div');
        inputs1.className = 'septum-inputs';

        const st1 = document.createElement('input');
        st1.type = 'text';
        st1.id = `start1_${cell.global_id}`;
        st1.className = 'endpoint-input';
        st1.placeholder = 'Start';
        st1.value = sData.start_aligned !== null ? sData.start_aligned : '';
        st1.addEventListener('change', () => {
            const v = parseInt(st1.value, 10);
            sData.start_aligned = !isNaN(v) ? v : null;
            if (sData.start_aligned !== null) sData.has_septum = true;
            saveRowSeptum(cell);
        });

        const en1 = document.createElement('input');
        en1.type = 'text';
        en1.id = `end1_${cell.global_id}`;
        en1.className = 'endpoint-input';
        en1.placeholder = 'End';
        en1.value = sData.end_aligned !== null ? sData.end_aligned : '';
        en1.addEventListener('change', () => {
            const v = parseInt(en1.value, 10);
            sData.end_aligned = !isNaN(v) ? v : null;
            if (sData.end_aligned !== null) sData.has_septum = true;
            saveRowSeptum(cell);
        });

        const clr1 = document.createElement('button');
        clr1.className = 'btn-clear';
        clr1.title = 'Clear Septum 1';
        clr1.textContent = '✕';
        clr1.addEventListener('click', () => {
            sData.has_septum = false;
            sData.start_aligned = null;
            sData.end_aligned = null;
            saveRowSeptum(cell);
        });

        inputs1.appendChild(st1);
        inputs1.appendChild(en1);
        inputs1.appendChild(clr1);

        row1.appendChild(lbl1);
        row1.appendChild(inputs1);

        // Septum 2 row
        const row2 = document.createElement('div');
        row2.className = 'septum-row-control';

        const lbl2 = document.createElement('label');
        const cb2 = document.createElement('input');
        cb2.type = 'checkbox';
        cb2.id = `has2_${cell.global_id}`;
        cb2.checked = !!sData.has_septum_2;
        cb2.addEventListener('change', () => {
            sData.has_septum_2 = cb2.checked;
            saveRowSeptum(cell);
        });
        lbl2.appendChild(cb2);
        lbl2.appendChild(document.createTextNode(' Septum 2'));

        const inputs2 = document.createElement('div');
        inputs2.className = 'septum-inputs';

        const st2 = document.createElement('input');
        st2.type = 'text';
        st2.id = `start2_${cell.global_id}`;
        st2.className = 'endpoint-input';
        st2.placeholder = 'Start';
        st2.value = sData.start_aligned_2 !== null ? sData.start_aligned_2 : '';
        st2.addEventListener('change', () => {
            const v = parseInt(st2.value, 10);
            sData.start_aligned_2 = !isNaN(v) ? v : null;
            if (sData.start_aligned_2 !== null) sData.has_septum_2 = true;
            saveRowSeptum(cell);
        });

        const en2 = document.createElement('input');
        en2.type = 'text';
        en2.id = `end2_${cell.global_id}`;
        en2.className = 'endpoint-input';
        en2.placeholder = 'End';
        en2.value = sData.end_aligned_2 !== null ? sData.end_aligned_2 : '';
        en2.addEventListener('change', () => {
            const v = parseInt(en2.value, 10);
            sData.end_aligned_2 = !isNaN(v) ? v : null;
            if (sData.end_aligned_2 !== null) sData.has_septum_2 = true;
            saveRowSeptum(cell);
        });

        const clr2 = document.createElement('button');
        clr2.className = 'btn-clear';
        clr2.title = 'Clear Septum 2';
        clr2.textContent = '✕';
        clr2.addEventListener('click', () => {
            sData.has_septum_2 = false;
            sData.start_aligned_2 = null;
            sData.end_aligned_2 = null;
            saveRowSeptum(cell);
        });

        inputs2.appendChild(st2);
        inputs2.appendChild(en2);
        inputs2.appendChild(clr2);

        row2.appendChild(lbl2);
        row2.appendChild(inputs2);

        // AI Control Row
        const rowAi = document.createElement('div');
        rowAi.className = 'septum-row-control';

        const runAiBtn = document.createElement('button');
        runAiBtn.id = `runAiBtn_${cell.global_id}`;
        runAiBtn.className = 'btn-ai-run';
        runAiBtn.textContent = '🤖 Run Live AI';
        runAiBtn.addEventListener('click', () => runLiveAiPrediction(cell));

        rowAi.appendChild(document.createTextNode('Live Inference'));
        rowAi.appendChild(runAiBtn);

        controlsDiv.appendChild(row1);
        controlsDiv.appendChild(row2);
        controlsDiv.appendChild(rowAi);

        labelCard.appendChild(headerDiv);
        labelCard.appendChild(subRowDiv);
        labelCard.appendChild(controlsDiv);

        // Strip Tiles Container
        const stripContainer = document.createElement('div');
        stripContainer.className = 'cell-strip-container';
        stripContainer.id = `stripContainer_${cell.global_id}`;

        const stripUrl = `/api/cell_strip_image?experiment=${encodeURIComponent(state.selectedExp)}&sequence=${encodeURIComponent(state.selectedSequence)}&cell_id=${encodeURIComponent(cell.global_id)}&channel=${encodeURIComponent(state.channel)}&_ts=${ts}`;

        // Load hidden full strip image to determine numFrames & render tile grid
        const hiddenImg = new Image();
        hiddenImg.src = stripUrl;
        hiddenImg.onload = () => {
            const numFrames = Math.round(hiddenImg.naturalWidth / 100);
            stripContainer.innerHTML = '';

            for (let t = 0; t < numFrames; t++) {
                const tile = document.createElement('div');
                tile.className = 'strip-crop';
                tile.dataset.t = String(t);
                tile.style.backgroundImage = `url('${stripUrl}')`;
                tile.style.backgroundPosition = `-${t * 80}px 0px`;
                tile.style.backgroundSize = `${numFrames * 80}px 80px`;

                // Handle tile click based on active mode
                tile.addEventListener('click', () => {
                    const mode = state.clickMode;
                    const cData = state.septumCache[cell.global_id];
                    if (!cData) return;

                    if (mode === 'start1') {
                        cData.has_septum = true;
                        cData.start_aligned = t;
                        setClickMode('nav');
                        saveRowSeptum(cell);
                    } else if (mode === 'end1') {
                        cData.has_septum = true;
                        cData.end_aligned = t;
                        setClickMode('nav');
                        saveRowSeptum(cell);
                    } else if (mode === 'start2') {
                        cData.has_septum_2 = true;
                        cData.start_aligned_2 = t;
                        setClickMode('nav');
                        saveRowSeptum(cell);
                    } else if (mode === 'end2') {
                        cData.has_septum_2 = true;
                        cData.end_aligned_2 = t;
                        setClickMode('nav');
                        saveRowSeptum(cell);
                    }
                });

                stripContainer.appendChild(tile);
            }

            updateRowAiUI(cell.global_id);
            updateRowTileHighlights(cell.global_id);
        };

        row.appendChild(labelCard);
        row.appendChild(stripContainer);

        grid.appendChild(row);
    });
}
