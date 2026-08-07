async function fetchQC() {
    let url = `/api/get_qc?experiment=${state.selectedExp}&sequence=${state.selectedSequence}`;
    const res = await fetch(url);
    const data = await res.json();
    state.qc = data.qc || {};
    updateQCUI();
}

async function setQC(status) {
    if (state.selectedCell === null) return;
    
    const currentStatus = state.qc[state.selectedCell];
    if (currentStatus === status) {
        status = 'pending';
    }
    
    const body = { experiment: state.selectedExp, cell_id: state.selectedCell, status: status, sequence: state.selectedSequence };
    const res = await fetch('/api/save_qc', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json();
    if (data.status === 'success') {
        if (status === 'pending') {
            delete state.qc[state.selectedCell];
        } else {
            state.qc[state.selectedCell] = status;
        }
        updateQCUI();
    } else alert("Failed to save QC");
}

function updateQCUI() {
    state.cells.forEach(c => {
        const btn = document.getElementById(`cell-item-${c.global_id}`);
        if (btn) {
            const st = state.qc[c.global_id] || 'pending';
            btn.classList.remove('qc-good', 'qc-bad', 'qc-corrected', 'qc-review');
            if (st !== 'pending') {
                btn.classList.add(`qc-${st}`);
            }
            
            const isSuspicious = !!(state.suspicious[c.global_id] && state.suspicious[c.global_id].length > 0);
            if (isSuspicious) {
                btn.classList.add('suspicious');
            } else {
                btn.classList.remove('suspicious');
            }
        }
    });

    if (state.selectedCell === null) {
        const maskTxt = document.getElementById('maskStatusText');
        if (maskTxt) {
            maskTxt.innerText = 'None';
            maskTxt.className = 'status-badge badge-pending';
        }
        return;
    }
    const status = state.qc[state.selectedCell] || 'pending';
    const maskTxt = document.getElementById('maskStatusText');
    if (maskTxt) {
        maskTxt.innerText = status.charAt(0).toUpperCase() + status.slice(1);
        let color = 'var(--text-muted)';
        let badgeClass = 'badge-pending';
        if (status === 'good') { color = '#10b981'; badgeClass = 'badge-good'; }
        if (status === 'bad') { color = '#ef4444'; badgeClass = 'badge-bad'; }
        if (status === 'corrected' || status === 'review') { color = '#f59e0b'; badgeClass = 'badge-pending'; }
        maskTxt.style.color = color;
        maskTxt.className = `status-badge ${badgeClass}`;
    }
}

function renderSuspiciousTicks() {
    const container = document.getElementById('suspiciousTicks');
    if (!container) return;
    container.innerHTML = '';
    
    const cellId = state.selectedCell;
    const list = document.getElementById('suspiciousFrameList');
    const section = document.getElementById('suspiciousSection');
    
    if (cellId === null) {
        if (section) section.style.display = 'none';
        if (list) list.innerHTML = '';
        return;
    }
    
    const frames = state.suspicious[cellId] || [];
    if (frames.length === 0) {
        if (section) section.style.display = 'none';
        if (list) list.innerHTML = '';
        return;
    }
    
    if (section) section.style.display = 'block';
    if (list) {
        list.innerHTML = frames.map(t => 
            `<button class="suspicious-frame-btn" onclick="goToFrame(${t})">t=${t}</button>`
        ).join('');
    }
    
    if (state.numFrames <= 1) return;
    
    frames.forEach(t => {
        const pct = (t / (state.numFrames - 1)) * 100;
        const tick = document.createElement('div');
        tick.className = 'suspicious-tick';
        tick.style.left = `calc(var(--tick-inset) + (100% - 2 * var(--tick-inset)) * ${pct / 100})`;
        container.appendChild(tick);
    });
}

function goToFrame(t) {
    state.currentFrame = t;
    const slider = document.getElementById('timeSlider');
    if (slider) slider.value = t;
    displayFrame();
}
