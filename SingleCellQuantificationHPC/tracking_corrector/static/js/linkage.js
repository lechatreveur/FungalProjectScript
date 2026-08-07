function renderLinkageBoard() {
    const list = document.getElementById('linkageList');
    if (!list) return;
    const globalLabel = document.getElementById('currentGlobalCellLabel');
    if (globalLabel) globalLabel.innerText = `(${state.selectedCell})`;
    
    let html = '';
    const films = state.linkageDetails ? state.linkageDetails.films || [] : [];
    const local_ids = state.linkageDetails ? state.linkageDetails.local_ids || [] : [];
    
    for (let i = 0; i < films.length; i++) {
        const curId = local_ids[i] !== undefined ? local_ids[i] : -1;
        html += `
        <div class="linkage-row" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-radius: 6px;">
            <div>
                <div style="color: var(--text-muted); font-size: 0.75rem;">${films[i]}</div>
                <div style="font-weight: 600;">Local Cell: ${curId === -1 ? 'None (-1)' : curId}</div>
            </div>
            <div style="display: flex; gap: 4px;">
                <button onclick="promptAndUpdateLinkage(${i}, '${films[i]}', ${curId})" style="padding: 3px 6px; font-size: 0.75rem; background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-main); border-radius: 4px; cursor: pointer;" title="Enter local cell ID directly">✏ Enter ID</button>
                <button onclick="openLinkageModal(${i}, '${films[i]}')" style="padding: 3px 6px; font-size: 0.75rem; background: var(--accent-primary); color: white; border: none; border-radius: 4px; cursor: pointer;" title="Click on canvas to select cell">🎯 Pick Canvas</button>
            </div>
        </div>
        `;
    }
    list.innerHTML = html;
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
    state.linkEditFilmIdx = filmIdx;
    state.linkEditFilmName = filmName;
    await updateLinkage(num);
}

function renderFilmBoundaries() {
    const container = document.getElementById('filmBoundaries');
    if (!container) return;
    container.innerHTML = '';
    if (state.isLocalEdit || !state.filmBoundaries || state.filmBoundaries.length === 0 || !state.linkageDetails) {
        return;
    }
    if (state.numFrames <= 1) return;
    
    const films = state.linkageDetails.films || [];
    for (let i = 0; i < state.filmBoundaries.length; i++) {
        const startFrame = state.filmBoundaries[i];
        const pct = (startFrame / (state.numFrames - 1)) * 100;
        
        const tick = document.createElement('div');
        tick.className = 'film-boundary-tick';
        tick.style.left = `calc(var(--tick-inset) + (100% - 2 * var(--tick-inset)) * ${pct / 100})`;
        
        const lbl = document.createElement('div');
        lbl.className = 'film-boundary-label';
        lbl.style.left = `calc(var(--tick-inset) + (100% - 2 * var(--tick-inset)) * ${pct / 100})`;
        lbl.innerText = films[i];
        
        container.appendChild(tick);
        container.appendChild(lbl);
    }
}

function openLinkageModal(filmIdx, filmName) {
    state.isEditingLink = true;
    state.linkEditFilmIdx = filmIdx;
    state.linkEditFilmName = filmName;
    
    let targetFrame = 0;
    if (filmIdx > 0 && state.filmBoundaries.length > filmIdx) {
        targetFrame = state.filmBoundaries[filmIdx];
    }
    state.currentFrame = targetFrame;
    const slider = document.getElementById('timeSlider');
    if (slider) slider.value = targetFrame;
    
    const modeLbl = document.getElementById('modeLabel');
    if (modeLbl) {
        modeLbl.innerText = `Pick Link Cell for ${filmName}`;
        modeLbl.style.color = '#f59e0b';
    }
    const newBtn = document.getElementById('trackNewCellBtn');
    if (newBtn) newBtn.style.display = 'block';
    
    displayFrame();
}

function closeModal() {
    const modal = document.getElementById('modalOverlay');
    if (modal) modal.style.display = 'none';
}

async function updateLinkage(newLocalId) {
    const body = {
        experiment: state.selectedExp,
        sequence: state.selectedSequence,
        global_cell_id: state.selectedCell,
        film_idx: state.linkEditFilmIdx,
        new_local_cell: parseInt(newLocalId)
    };
    const res = await fetch('/api/update_linkage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.status === 'success') {
        state.isEditingLink = false;
        const modeLbl = document.getElementById('modeLabel');
        if (modeLbl) {
            modeLbl.innerText = 'Click-Select';
            modeLbl.style.color = 'var(--accent-primary)';
        }
        selectCell(state.selectedCell);
    } else {
        alert("Failed to update linkage: " + data.message);
    }
}

async function identifyAndLinkCell(x, y) {
    const modeLbl = document.getElementById('modeLabel');
    if (modeLbl) modeLbl.innerText = 'Searching...';
    
    let localT = 0;
    if (state.filmBoundaries && state.filmBoundaries.length > state.linkEditFilmIdx) {
        localT = state.currentFrame - state.filmBoundaries[state.linkEditFilmIdx];
        if (localT < 0) localT = 0;
    } else {
        localT = state.currentFrame;
    }
    const body = {
        experiment: state.selectedExp,
        film: state.linkEditFilmName,
        t: localT,
        x: x,
        y: y
    };
    const res = await fetch('/api/identify_cell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.status === 'success') {
        const localId = data.local_cell_id !== undefined ? data.local_cell_id : data.cell_id;
        updateLinkage(localId);
    } else if (data.status === 'untracked') {
        const conf = confirm(`This cell (Label #${data.label_id}) is not currently tracked. Do you want to quantify it locally on the laptop now? (This will take a moment)`);
        if (conf) {
            await quantifyCellLocally(data.label_id);
        } else if (modeLbl) {
            modeLbl.innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
        }
    } else {
        alert(data.message);
        if (modeLbl) modeLbl.innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
    }
}

async function quantifyCellLocally(labelId) {
    const modeLbl = document.getElementById('modeLabel');
    if (modeLbl) modeLbl.innerText = `Quantifying Cell #${labelId}...`;
    
    try {
        const res = await fetch('/api/quantify_on_hpc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment: state.selectedExp,
                film: state.linkEditFilmName,
                label_id: labelId,
                seed_from_csv: true,
                track_channel: state.channel
            })
        });
        const data = await res.json();
        if (data.status === 'success') {
            await updateLinkage(labelId);
        } else {
            alert("Quantification failed: " + data.message);
            if (modeLbl) modeLbl.innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
        }
    } catch (e) {
        alert(e);
        if (modeLbl) modeLbl.innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
    }
}
