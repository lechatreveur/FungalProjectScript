async function loadCells(exp, target) {
    const res = await fetch(`/api/list_cells?experiment=${exp}&sequence=${target}`);
    const data = await res.json();
    state.cells = data.cells || [];
    state.lineageTree = data.lineage || {};
    
    state.qc = {};
    state.suspicious = {};
    
    const cellGrid = document.getElementById('cellGrid');
    if (cellGrid) {
        cellGrid.innerHTML = state.cells.map(c => `
            <div class="cell-item" id="cell-item-${c.global_id}" onclick="selectCell('${c.global_id}')">${c.display_name}</div>
        `).join('');
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
        const stripContainer = document.getElementById('stripContainer');
        if (stripContainer) stripContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No cells available to display.</span>';
        const linkageList = document.getElementById('linkageList');
        if (linkageList) linkageList.innerHTML = '';
        const globalLabel = document.getElementById('currentGlobalCellLabel');
        if (globalLabel) globalLabel.innerText = '';
        if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    fetch(`/api/suspicious_cells?experiment=${exp}&sequence=${target}`)
        .then(r => r.json())
        .then(suspData => {
            state.suspicious = suspData.suspicious || {};
            updateQCUI();
            renderSuspiciousTicks();
        })
        .catch(e => console.error("Error fetching suspicious cells:", e));
}

async function selectCell(cellId) {
    if (state.selectedCell !== null) {
        const prevActive = document.getElementById(`cell-item-${state.selectedCell}`);
        if (prevActive) prevActive.classList.remove('active');
    }
    cancelAutosave();
    state.selectedCell = cellId;
    state.isEditingLink = false;
    state.isLocalEdit = false;
    
    const trackNewBtn = document.getElementById('trackNewCellBtn');
    if (trackNewBtn) trackNewBtn.style.display = 'none';
    const exitBtn = document.getElementById('btnExitLocalEdit');
    if (exitBtn) exitBtn.style.display = 'none';
    
    const modeLbl = document.getElementById('modeLabel');
    if (modeLbl) {
        modeLbl.style.color = 'var(--accent-primary)';
        if (state.tool === 'select') modeLbl.innerText = 'Click-Select';
    }
    
    const activeItem = document.getElementById(`cell-item-${cellId}`);
    if (activeItem) activeItem.classList.add('active');
    
    const cellLbl = document.getElementById('cellIdLabel');
    if (cellLbl) cellLbl.innerText = cellId;
    
    const modeParam = `sequence=${state.selectedSequence}`;
    const res = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&${modeParam}&cell_id=${cellId}`);
    const data = await res.json();
    
    state.cellMasks = data.masks || [];
    state.numFrames = data.num_frames || 0;
    state.imgWidth = data.width || 2000;
    state.imgHeight = data.height || 2000;
    state.channel = data.track_channel || 'bf';
    state.filmBoundaries = data.film_boundaries || [];
    state.localFilmId = data.local_film;
    
    state.linkageDetails = data.linkage_details || {};
    renderLinkageBoard();
    renderFilmBoundaries();
    
    const channelLbl = document.getElementById('cellChannelLabel');
    if (channelLbl) channelLbl.innerText = (data.track_channel || 'bf').toUpperCase();
    
    let startFrame = 0;
    if (state.linkageDetails && state.linkageDetails.local_ids && state.filmBoundaries && state.filmBoundaries.length > 0) {
        const local_ids = state.linkageDetails.local_ids;
        for (let i = 0; i < local_ids.length; i++) {
            if (local_ids[i] !== -1 && i < state.filmBoundaries.length) {
                startFrame = state.filmBoundaries[i];
                break;
            }
        }
    }

    const slider = document.getElementById('timeSlider');
    if (slider) {
        slider.max = Math.max(0, state.numFrames - 1);
        slider.value = startFrame;
    }
    state.currentFrame = startFrame;
    const maxTimeLbl = document.getElementById('maxTimeLabel');
    if (maxTimeLbl) maxTimeLbl.innerText = `t=${Math.max(0, state.numFrames - 1)}`;
    
    updateChannelButtons();
    resetView();
    await displayFrame();
    updateQCUI();
    renderGallery();
    await loadSeptumLabels(cellId);
    renderSuspiciousTicks();
}

async function selectLocalCell(cellId, filmId) {
    if (state.selectedCell !== null) {
        const prevActive = document.getElementById(`cell-item-${state.selectedCell}`);
        if (prevActive) prevActive.classList.remove('active');
    }
    cancelAutosave();
    state.selectedCell = cellId;
    state.isLocalEdit = true;
    state.localFilmId = filmId;
    
    const exitBtn = document.getElementById('btnExitLocalEdit');
    if (exitBtn) exitBtn.style.display = 'block';
    
    const modeLbl = document.getElementById('modeLabel');
    if (modeLbl) {
        modeLbl.innerText = `Editing ${filmId} Cell #${cellId}`;
        modeLbl.style.color = '#38bdf8';
    }
    
    const res = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&film=${filmId}&cell_id=${cellId}`);
    const data = await res.json();
    
    state.cellMasks = data.masks || [];
    state.numFrames = data.num_frames || 0;
    state.imgWidth = data.width || 2000;
    state.imgHeight = data.height || 2000;
    state.channel = data.track_channel || 'bf';
    
    const cellLbl = document.getElementById('cellIdLabel');
    if (cellLbl) cellLbl.innerText = `${cellId} (${filmId})`;
    
    const slider = document.getElementById('timeSlider');
    if (slider) {
        slider.max = Math.max(0, state.numFrames - 1);
        slider.value = 0;
    }
    state.currentFrame = 0;
    
    renderFilmBoundaries();
    updateChannelButtons();
    resetView();
    await displayFrame();
    updateQCUI();
    renderGallery();
    await loadSeptumLabels(cellId);
}

async function exitLocalCellEdit() {
    state.isLocalEdit = false;
    const exitBtn = document.getElementById('btnExitLocalEdit');
    if (exitBtn) exitBtn.style.display = 'none';
    
    if (state.prevGlobalCell) {
        const targetGlobalCell = state.prevGlobalCell;
        const targetFilmIdx = state.prevLinkEditFilmIdx;
        const targetFilmName = state.prevLinkEditFilmName;
        
        state.prevGlobalCell = null;
        state.prevLinkEditFilmIdx = null;
        state.prevLinkEditFilmName = null;
        
        await selectCell(targetGlobalCell);
        openLinkageModal(targetFilmIdx, targetFilmName);
    } else {
        await loadCells(state.selectedExp, state.selectedSequence);
    }
}

async function activateCellAtCoords(x, y) {
    const modeLbl = document.getElementById('modeLabel');
    if (modeLbl) {
        modeLbl.innerText = 'Activating cell...';
        modeLbl.style.color = 'var(--accent-green)';
    }
    
    const active = getActiveFilmAndLocalCell();
    const body = {
        experiment: state.selectedExp,
        sequence: state.selectedSequence,
        film: active.film || state.localFilmId,
        t: state.currentFrame,
        x: Math.round(x),
        y: Math.round(y)
    };
    
    try {
        const res = await fetch('/api/identify_cell', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        
        if (data.status === 'success' && data.cell_id) {
            if (state.viewMode === 'population') {
                state.viewMode = 'single';
                const btn = document.getElementById('viewModeBtn');
                if (btn) {
                    btn.innerText = 'Single Cell';
                    btn.style.backgroundColor = '#581c87';
                }
                const brushBtn = document.getElementById('toolBrushBtn');
                const eraserBtn = document.getElementById('toolEraserBtn');
                if (brushBtn) { brushBtn.disabled = false; brushBtn.style.opacity = 1.0; }
                if (eraserBtn) { eraserBtn.disabled = false; eraserBtn.style.opacity = 1.0; }
            }
            
            await selectCell(data.cell_id);
            if (modeLbl) {
                modeLbl.innerText = `Activated Cell #${data.cell_id}`;
                modeLbl.style.color = 'var(--accent-green)';
            }
        } else if (modeLbl) {
            modeLbl.innerText = 'No cell mask found at click';
            modeLbl.style.color = 'var(--accent-red)';
        }
    } catch (err) {
        console.error('Error activating cell:', err);
        if (modeLbl) modeLbl.innerText = 'Error identifying cell';
    }
}

function markDirty() {
    const status = document.getElementById('autosaveStatus');
    if (status) {
        status.innerText = "Unsaved changes";
        status.style.background = "#7f1d1d";
        status.style.color = "#fee2e2";
    }
    cancelAutosave();
    state.autosaveTimer = setTimeout(() => {
        saveCorrectedMasks(true);
    }, 2000);
}

function cancelAutosave() {
    if (state.autosaveTimer) {
        clearTimeout(state.autosaveTimer);
        state.autosaveTimer = null;
    }
}

async function saveCorrectedMasks(isAutosave = false) {
    cancelAutosave();
    const status = document.getElementById('autosaveStatus');
    if (status) {
        status.innerText = isAutosave ? "Autosaving..." : "Saving...";
        status.style.background = "#0c4a6e";
        status.style.color = "#e0f2fe";
    }
    
    if (state.selectedCell === null) return;
    
    const activeInfo = getActiveFilmAndLocalCell();
    if (!activeInfo.film || activeInfo.cellId === null || activeInfo.cellId === -1) {
        if (status) {
            status.innerText = "Error: Cell not in film";
            status.style.background = "#7f1d1d";
        }
        return;
    }
    
    const body = {
        experiment: state.selectedExp,
        film: activeInfo.film,
        cell_id: activeInfo.cellId.toString(),
        channel: state.channel,
        masks: state.cellMasks
    };
    
    try {
        const res = await fetch('/api/save_masks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.status === 'success') {
            if (status) {
                status.innerText = "Saved";
                status.style.background = "#064e3b";
                status.style.color = "#d1fae5";
            }
            if (state.qc[state.selectedCell] === undefined || state.qc[state.selectedCell] === 'pending') {
                await setQC('corrected');
            }
        } else {
            if (status) {
                status.innerText = "Save failed";
                status.style.background = "#7f1d1d";
                status.style.color = "#fee2e2";
            }
            if (!isAutosave) alert("Failed to save masks: " + data.message);
        }
    } catch (e) {
        if (status) {
            status.innerText = "Save error";
            status.style.background = "#7f1d1d";
            status.style.color = "#fee2e2";
        }
        if (!isAutosave) alert("Error saving masks: " + e);
    }
}
