async function loadCells(exp, target) {
    try {
        const res = await fetch(`/api/list_cells?experiment=${exp}&sequence=${target}`);
        const data = await res.json();
        state.cells = data.cells || [];
        
        state.qc = {};
        
        const cellGrid = document.getElementById('cellGrid');
        const badge = document.getElementById('cellCountBadge');
        if (badge) badge.innerText = `${state.cells.length} cells`;
        
        if (cellGrid) {
            cellGrid.innerHTML = state.cells.map(c => {
                const label = c.display_name.includes('_cell_') ? `Cell ${c.display_name.split('_cell_')[1]}` : c.display_name;
                // Same stable colour the population / mask views use for this cell.
                const sw = (typeof idToColor === 'function')
                    ? `<span class="cell-swatch" style="background:rgb(${idToColor(stableColorKey(String(c.global_id))).join(',')})"></span>`
                    : '';
                return `
                <div class="cell-item" id="cell-item-${c.global_id}" onclick="selectCell('${c.global_id}')" title="${c.display_name}">
                    ${sw}${label}
                </div>
            `}).join('');
        }

        if (state.cells.length > 0) {
            await fetchQC();
            const exists = state.cells.some(c => c.global_id === state.selectedCell);
            if (exists && state.selectedCell !== null) {
                await selectCell(state.selectedCell);
            } else {
                await selectCell(state.cells[0].global_id);
            }
        } else {
            state.selectedCell = null;
            const cellLbl = document.getElementById('cellIdLabel');
            if (cellLbl) cellLbl.innerText = 'None';
            
            // Fallback load keyframe info for empty dataset
            try {
                const res = await fetch(`/api/cell_masks?experiment=${exp}&sequence=${target}&cell_id=0`);
                const maskData = await res.json();
                state.cellMasks = [];
                state.numFrames = maskData.num_frames || 0;
                state.keyframes = maskData.keyframes || [];
                state.imgWidth = maskData.width || 2000;
                state.imgHeight = maskData.height || 2000;
                state.channel = maskData.track_channel || 'bf';
                state.filmBoundaries = maskData.film_boundaries || [];
                
                const slider = document.getElementById('timeSlider');
                if (slider) {
                    slider.max = Math.max(0, state.numFrames - 1);
                    slider.value = 0;
                }
                state.currentFrame = 0;
                const maxTimeLbl = document.getElementById('maxTimeLabel');
                if (maxTimeLbl) maxTimeLbl.innerText = `kf=${Math.max(0, state.numFrames - 1)}`;
                
                resetView();
                await displayFrame();
            } catch (e) {
                if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
    } catch (err) {
        console.error("Error loading cells:", err);
    }
}

async function selectCell(cellId, options = {}) {
    if (state.selectedCell !== null) {
        const prevActive = document.getElementById(`cell-item-${state.selectedCell}`);
        if (prevActive) prevActive.classList.remove('active');
    }
    await flushPendingAutosave();
    state.selectedCell = cellId;
    
    const currActive = document.getElementById(`cell-item-${cellId}`);
    if (currActive) {
        currActive.classList.add('active');
        currActive.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    const cellLbl = document.getElementById('cellIdLabel');
    if (cellLbl) cellLbl.innerText = cellId;
    
    const globalLabel = document.getElementById('currentGlobalCellLabel');
    if (globalLabel) globalLabel.innerText = cellId;
    
    const modeParam = state.selectedSequence ? `sequence=${state.selectedSequence}` : `film=${state.selectedFilm || ''}`;
    const res = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&${modeParam}&cell_id=${cellId}`);
    const data = await res.json();
    
    state.cellMasks = data.masks || [];
    state.numFrames = data.num_frames || 0;
    state.keyframes = data.keyframes || [];
    state.imgWidth = data.width || 2000;
    state.imgHeight = data.height || 2000;
    state.channel = data.track_channel || 'bf';
    state.filmBoundaries = data.film_boundaries || [];
    state.linkageDetails = data.linkage_details || {};
    
    renderFilmBoundaries();
    renderLinkageBoard();
    
    const slider = document.getElementById('timeSlider');
    if (slider) {
        slider.max = Math.max(0, state.numFrames - 1);
    }
    const maxTimeLbl = document.getElementById('maxTimeLabel');
    if (maxTimeLbl) maxTimeLbl.innerText = `kf=${Math.max(0, state.numFrames - 1)}`;
    
    if (!options.preserveView && !state.userHasPanned) {
        resetView();
    }
    await displayFrame();
    updateQCUI();
}


function renderLinkageBoard() {
    const list = document.getElementById('linkageList');
    if (!list) return;
    const globalLabel = document.getElementById('currentGlobalCellLabel');
    if (globalLabel) globalLabel.innerText = state.selectedCell ? `(${state.selectedCell})` : '';
    
    let html = '';
    const films = state.linkageDetails ? state.linkageDetails.films || [] : [];
    const local_ids = state.linkageDetails ? state.linkageDetails.local_ids || [] : [];
    
    if (films.length === 0) {
        list.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 6px;">No sequence linkages</div>';
        return;
    }

    for (let i = 0; i < films.length; i++) {
        const curId = local_ids[i] !== undefined ? local_ids[i] : -1;
        const isLinked = curId !== -1;
        html += `
        <div class="linkage-row" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; padding: 5px 8px; background: rgba(255,255,255,0.03); border-radius: 6px; border-left: 3px solid ${isLinked ? 'var(--accent-primary)' : 'var(--border-color)'};">
            <div style="cursor: pointer;" onclick="jumpToFilm(${i})" title="Jump to this film">
                <div style="color: var(--text-muted); font-size: 0.72rem;">${films[i]}</div>
                <div style="font-weight: 600; font-size: 0.82rem; color: ${isLinked ? 'var(--text-main)' : 'var(--text-muted)'};">
                    ${isLinked ? `Cell: ${curId}` : 'Unlinked (-1)'}
                </div>
            </div>
            <div style="display: flex; gap: 4px;">
                <button onclick="promptAndUpdateLinkage(${i}, '${films[i]}', ${curId})" style="padding: 2px 7px; font-size: 0.72rem; background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-main); border-radius: 4px; cursor: pointer;" title="Enter local cell ID directly">✏ Edit</button>
            </div>
        </div>
        `;
    }
    list.innerHTML = html;
}

function jumpToFilm(filmIdx) {
    if (!state.keyframes || state.keyframes.length === 0) return;
    for (let i = 0; i < state.keyframes.length; i++) {
        if (state.keyframes[i].film_idx === filmIdx) {
            updateCurrentFrame(i);
            break;
        }
    }
}

async function promptAndUpdateLinkage(filmIdx, filmName, currentLocalId) {
    const val = prompt(`Enter Local Cell ID for ${filmName} (enter -1 to unlink):`, currentLocalId === -1 ? "" : currentLocalId);
    if (val === null) return;
    const trimmed = val.trim();
    if (trimmed === "") return;
    const num = parseInt(trimmed);
    if (isNaN(num)) {
        alert("Please enter a valid integer cell ID or -1.");
        return;
    }

    try {
        const res = await fetch('/api/update_linkage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment: state.selectedExp,
                sequence: state.selectedSequence,
                global_id: String(state.selectedCell),
                film_idx: filmIdx,
                new_local_id: num
            })
        });
        const data = await res.json();
        if (data.status === 'success') {
            if (state.linkageDetails && state.linkageDetails.local_ids) {
                state.linkageDetails.local_ids[filmIdx] = num;
            }
            renderLinkageBoard();
            await selectCell(state.selectedCell);
        } else {
            alert("Error updating linkage: " + (data.message || data.error));
        }
    } catch (e) {
        alert("Network error updating linkage: " + e);
    }
}

function renderFilmBoundaries() {
    const ticksContainer = document.getElementById('timelineTicks');
    if (!ticksContainer || !state.filmBoundaries) return;
    
    ticksContainer.innerHTML = '';
    const total = Math.max(1, state.numFrames - 1);
    
    state.filmBoundaries.forEach(b => {
        const pct = (b.frame / total) * 100;
        const tick = document.createElement('div');
        tick.style.position = 'absolute';
        tick.style.left = `${pct}%`;
        tick.style.top = '-6px';
        tick.style.width = '2px';
        tick.style.height = '16px';
        tick.style.backgroundColor = 'var(--accent-primary)';
        tick.style.opacity = '0.7';
        tick.title = `Film: ${b.film}`;
        ticksContainer.appendChild(tick);
    });
}

async function fetchQC() {
    const modeParam = state.selectedSequence ? `sequence=${state.selectedSequence}` : `film=${state.selectedFilm || ''}`;
    try {
        const res = await fetch(`/api/get_qc?experiment=${state.selectedExp}&${modeParam}`);
        const data = await res.json();
        state.qc = data.qc || {};
        updateQCUI();
    } catch (e) {
        console.error("Error fetching QC:", e);
    }
}

function updateQCUI() {
    // Update active cell badge
    const badge = document.getElementById('maskStatusText');
    const numId = state.selectedCell && state.selectedCell.includes('_cell_') ? state.selectedCell.split('_cell_')[1] : state.selectedCell;
    const cellQC = state.selectedCell ? (state.qc[state.selectedCell] || state.qc[numId]) : null;
    const status = cellQC ? cellQC.status : 'unreviewed';
    
    if (badge) {
        badge.className = `status-badge badge-${status}`;
        badge.innerText = status.toUpperCase();
    }
    
    const filterSelect = document.getElementById('cellFilterSelect');
    const filterVal = filterSelect ? filterSelect.value : 'all';
    let visibleCount = 0;

    // Update grid item styling and filter visibility
    state.cells.forEach(c => {
        const item = document.getElementById(`cell-item-${c.global_id}`);
        if (item) {
            item.classList.remove('status-good', 'status-bad', 'status-corrected', 'status-mistracked');
            const cNum = (c.global_id && c.global_id.includes('_cell_')) ? c.global_id.split('_cell_')[1] : String(c.id || '');
            const q = state.qc[c.global_id] || (cNum ? state.qc[cNum] : null);
            const itemStatus = (q && q.status) ? q.status : 'unreviewed';
            if (itemStatus !== 'unreviewed') {
                item.classList.add(`status-${itemStatus}`);
            }

            let show = true;
            if (filterVal === 'good') show = (itemStatus === 'good');
            else if (filterVal === 'bad') show = (itemStatus === 'bad');
            else if (filterVal === 'corrected') show = (itemStatus === 'corrected' || itemStatus === 'review');
            else if (filterVal === 'mistracked') show = (itemStatus === 'mistracked');

            item.style.display = show ? '' : 'none';
            if (show) visibleCount++;
        }
    });

    const countBadge = document.getElementById('cellCountBadge');
    if (countBadge) {
        countBadge.innerText = filterVal === 'all'
            ? `${state.cells.length} cells`
            : `${visibleCount} / ${state.cells.length} cells`;
    }
}

// Bind filter change event
document.addEventListener('DOMContentLoaded', () => {
    const filterSelect = document.getElementById('cellFilterSelect');
    if (filterSelect) {
        filterSelect.addEventListener('change', updateQCUI);
    }
});

async function setQC(status) {
    if (!state.selectedCell) return;
    state.qc[state.selectedCell] = { status: status };
    updateQCUI();
    
    const modeParam = state.selectedSequence ? `sequence=${state.selectedSequence}` : `film=${state.selectedFilm || ''}`;
    await fetch('/api/save_qc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            experiment: state.selectedExp,
            sequence: state.selectedSequence,
            cell_id: state.selectedCell,
            status: status
        })
    });
}

function cancelAutosave() {
    if (state.autosaveTimer) {
        clearTimeout(state.autosaveTimer);
        state.autosaveTimer = null;
    }
    const tag = document.getElementById('autosaveStatus');
    if (tag) tag.innerText = 'Idle';
}

async function flushPendingAutosave() {
    if (state.autosaveTimer) {
        clearTimeout(state.autosaveTimer);
        state.autosaveTimer = null;
        await saveCurrentCell();
    }
}

function triggerAutosave(targetFrame = null) {
    const frameToSave = (targetFrame !== null && targetFrame !== undefined) ? targetFrame : state.currentFrame;
    const tag = document.getElementById('autosaveStatus');
    if (tag) {
        tag.innerText = 'Saving...';
        tag.style.color = 'var(--accent-yellow)';
    }
    
    if (state.autosaveTimer) clearTimeout(state.autosaveTimer);
    state.autosaveTimer = setTimeout(async () => {
        state.autosaveTimer = null;
        await saveCurrentCell(frameToSave);
    }, 500);
}

async function saveCurrentCell(targetFrame = null) {
    if (!state.selectedCell) return;
    const fIdx = (targetFrame !== null && targetFrame !== undefined) ? targetFrame : state.currentFrame;
    const tag = document.getElementById('autosaveStatus');
    
    try {
        const payload = {
            experiment: state.selectedExp,
            sequence: state.selectedSequence,
            cell_id: state.selectedCell,
            channel: state.channel,
            masks: state.cellMasks,
            changes: [
                {
                    time_point: fIdx,
                    new_rle: state.cellMasks[fIdx] || ""
                }
            ]
        };

        const res = await fetch('/api/save_mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        if (data.status === 'success') {
            if (tag) {
                tag.innerText = 'Synced to Cellpose ✅';
                tag.style.color = 'var(--accent-green)';
            }
            // Mark cell as corrected if not good
            if (!state.qc[state.selectedCell] || state.qc[state.selectedCell].status !== 'good') {
                await setQC('corrected');
            }
        }
    } catch (err) {
        console.error("Save error:", err);
        if (tag) {
            tag.innerText = 'Save Error ⚠️';
            tag.style.color = 'var(--accent-red)';
        }
    }
}

async function activateCellAtCoords(x, y) {
    const statusText = document.getElementById('autosaveStatus') || document.getElementById('cellIdLabel');
    if (statusText) {
        statusText.innerText = 'Activating cell...';
        statusText.style.color = 'var(--accent-secondary)';
    }

    const payload = {
        experiment: state.selectedExp,
        sequence: state.selectedSequence,
        film: state.selectedFilm,
        t: state.currentFrame,
        x: Math.round(x),
        y: Math.round(y)
    };

    try {
        const res = await fetch('/api/identify_cell', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.status === 'success' && data.cell_id) {
            if (state.viewMode === 'population') {
                state.viewMode = 'single';
                const viewBtn = document.getElementById('viewModeBtn');
                if (viewBtn) {
                    viewBtn.innerText = 'Single Cell';
                    viewBtn.style.backgroundColor = 'var(--accent-primary)';
                }
                const brushBtn = document.getElementById('toolBrushBtn');
                const eraserBtn = document.getElementById('toolEraserBtn');
                if (brushBtn) { brushBtn.disabled = false; brushBtn.style.opacity = '1.0'; }
                if (eraserBtn) { eraserBtn.disabled = false; eraserBtn.style.opacity = '1.0'; }
            }

            await selectCell(data.cell_id, { preserveView: true });
            if (statusText) {
                statusText.innerText = `Selected ${data.cell_id}`;
                statusText.style.color = 'var(--accent-green)';
            }
        } else {
            if (statusText) {
                statusText.innerText = 'No cell mask found at click';
                statusText.style.color = 'var(--accent-red)';
            }
        }
    } catch (err) {
        console.error("Error activating cell:", err);
    }
}

