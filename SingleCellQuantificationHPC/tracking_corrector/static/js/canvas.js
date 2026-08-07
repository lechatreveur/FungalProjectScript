function updateChannelButtons() {
    const bfBtn = document.getElementById('chanBfBtn');
    const gfpBtn = document.getElementById('chanGfpBtn');
    if (bfBtn) bfBtn.classList.toggle('active', state.channel === 'bf');
    if (gfpBtn) gfpBtn.classList.toggle('active', state.channel === 'gfp');
}

async function displayFrame() {
    const timeLbl = document.getElementById('currentTimeLabel');
    if (timeLbl) timeLbl.innerText = `t=${state.currentFrame}`;
    updateGalleryHighlight();
    
    const current = getActiveFilmAndLocalCell();
    if (current.film !== state.lastActiveFilm || current.cellId !== state.lastActiveCellId) {
        state.lastActiveFilm = current.film;
        state.lastActiveCellId = current.cellId;
        // Only refetch septum labels from the server when we're in plain
        // navigation mode. While the user is mid-way through setting a
        // septum endpoint (galleryClickMode is 'start1'/'end1'/'start2'/
        // 'end2'), a gallery click can legitimately jump the playhead into a
        // *different film* - e.g. the septum's other endpoint already set
        // sits in the previous film - and reloading here would overwrite the
        // in-progress, not-yet-saved input values with whatever is still on
        // disk, making the endpoint the user just set appear to vanish. See
        // septum.js header comment for the full cross-film septum rule.
        if (state.galleryClickMode === 'nav') {
            await loadSeptumLabels(state.selectedCell);
        }
    }

    const localFilmRow = document.getElementById('localFilmRow');
    const localFilmLbl = document.getElementById('localFilmLabel');
    if (state.isLocalEdit) {
        if (localFilmRow) localFilmRow.style.display = 'flex';
        if (localFilmLbl) localFilmLbl.innerText = `${state.localFilmId} (Local Edit)`;
    } else if (state.filmBoundaries && state.filmBoundaries.length > 0) {
        if (localFilmRow) localFilmRow.style.display = 'flex';
        let fIdx = 0;
        for (let i = 0; i < state.filmBoundaries.length; i++) {
            if (state.currentFrame >= state.filmBoundaries[i]) {
                fIdx = i;
            }
        }
        const films = state.linkageDetails ? state.linkageDetails.films || [] : [];
        const local_ids = state.linkageDetails ? state.linkageDetails.local_ids || [] : [];
        if (localFilmLbl) localFilmLbl.innerText = `${films[fIdx]} (Cell ${local_ids[fIdx]})`;
    } else if (localFilmRow) {
        localFilmRow.style.display = 'none';
    }
    
    const modeParam = state.isLocalEdit ? `film=${state.localFilmId}` : `sequence=${state.selectedSequence}`;
    
    const loadImage = (src) => new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = src;
    });
    
    const ts = Date.now();
    if (state.viewMode === 'population') {
        const popUrl = `/api/population_frame?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&_ts=${ts}`;
        const img = await loadImage(popUrl);
        if (!img || !canvas || !ctx) return;
        
        canvas.width = state.imgWidth;
        canvas.height = state.imgHeight;
        canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        return;
    }
    
    const imgUrl = `/api/frame_image?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&channel=${state.channel}&_ts=${ts}`;
    const promises = [loadImage(imgUrl)];
    
    if (state.tool === 'select') {
        const boundsUrl = `/api/frame_boundaries?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&_ts=${ts}`;
        promises.push(loadImage(boundsUrl));
    } else {
        promises.push(Promise.resolve(null));
    }
    
    const [img, boundariesImg] = await Promise.all(promises);
    
    if (!img || !canvas || !ctx) return;
    
    canvas.width = state.imgWidth;
    canvas.height = state.imgHeight;
    canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    
    if (boundariesImg) {
        ctx.drawImage(boundariesImg, 0, 0);
    }
    
    drawMask();
}

function drawMask() {
    if (!ctx || !canvas) return;
    const currentRle = state.cellMasks[state.currentFrame];
    if (!currentRle) return;
    const maskArr = decodeRle(currentRle, state.imgWidth, state.imgHeight);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    for (let i = 0; i < maskArr.length; i++) {
        if (maskArr[i]) {
            const idx = i * 4;
            data[idx] = Math.min(255, data[idx] + 20);      // R
            data[idx + 1] = Math.min(255, data[idx + 1] + 160); // G
            data[idx + 2] = Math.min(255, data[idx + 2] + 40);  // B
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function decodeRle(rleStr, W, H) {
    const flat = new Uint8Array(W * H);
    if (!rleStr || rleStr.trim() === "") return flat;
    const nums = rleStr.trim().split(/\s+/).map(Number);
    for (let i = 0; i < nums.length; i += 2) {
        const start = nums[i] - 1;
        const length = nums[i + 1];
        for (let j = 0; j < length; j++) {
            const idx = start + j;
            if (idx < flat.length) flat[idx] = 1;
        }
    }
    const rowMajor = new Uint8Array(W * H);
    let k = 0;
    for (let x = 0; x < W; x++) {
        for (let y = 0; y < H; y++) {
            rowMajor[y * W + x] = flat[k++];
        }
    }
    return rowMajor;
}

function rleEncode(rowMajorArr, W, H) {
    const flat = new Uint8Array(W * H);
    let k = 0;
    for (let x = 0; x < W; x++) {
        for (let y = 0; y < H; y++) {
            flat[k++] = rowMajorArr[y * W + x];
        }
    }
    const starts = [];
    const lengths = [];
    let prev = 0;
    for (let i = 0; i <= flat.length; i++) {
        const v = (i < flat.length) ? flat[i] : 0;
        const diff = v - prev;
        if (diff === 1) starts.push(i + 1);
        else if (diff === -1) lengths.push(i + 1 - starts[starts.length - 1]);
        prev = v;
    }
    const out = [];
    for (let i = 0; i < starts.length; i++) {
        out.push(starts[i]);
        out.push(lengths[i]);
    }
    return out.join(" ");
}

function drawStroke(x, y, isAdding) {
    const currentRle = state.cellMasks[state.currentFrame] || "";
    const maskArr = decodeRle(currentRle, state.imgWidth, state.imgHeight);
    const r = state.brushSize;
    
    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            if (dx*dx + dy*dy <= r*r) {
                const px = x + dx;
                const py = y + dy;
                if (px >= 0 && px < state.imgWidth && py >= 0 && py < state.imgHeight) {
                    maskArr[py * state.imgWidth + px] = isAdding ? 1 : 0;
                }
            }
        }
    }
    
    state.cellMasks[state.currentFrame] = rleEncode(maskArr, state.imgWidth, state.imgHeight);
    displayFrame();
    markDirty();
}

function undoStroke() {
    if (state.drawingHistory.length > 0) {
        state.cellMasks[state.currentFrame] = state.drawingHistory.pop();
        displayFrame();
        markDirty();
    }
}

async function clickSelectSegment(x, y, isShift) {
    const body = {
        experiment: state.selectedExp,
        t: state.currentFrame,
        channel: state.channel,
        cell_id: state.selectedCell,
        x: x,
        y: y
    };
    if (state.isLocalEdit) {
        body.film = state.localFilmId;
    } else {
        body.sequence = state.selectedSequence;
    }
    
    try {
        const res = await fetch('/api/click_segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        
        if (data.status === 'success' && data.rle) {
            state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
            if (state.drawingHistory.length > 20) state.drawingHistory.shift();

            if (isShift) {
                const W = state.imgWidth;
                const H = state.imgHeight;
                const maskArr = decodeRle(state.cellMasks[state.currentFrame] || "", W, H);
                const newMaskArr = decodeRle(data.rle, W, H);
                for (let i = 0; i < maskArr.length; i++) {
                    if (newMaskArr[i]) maskArr[i] = 1;
                }
                state.cellMasks[state.currentFrame] = rleEncode(maskArr, W, H);
            } else {
                state.cellMasks[state.currentFrame] = data.rle;
            }
            displayFrame();
            markDirty();
        }
    } catch (e) {
        console.error("Error in click select segment:", e);
    }
}

async function runAutofix() {
    const startInput = document.getElementById('autofixStartInput');
    const endInput = document.getElementById('autofixEndInput');
    const btn = document.getElementById('runAutofixBtn');
    
    if (!startInput || !endInput || !startInput.value || !endInput.value) {
        return alert("Please enter both start and end frame numbers.");
    }
    
    const startT = parseInt(startInput.value);
    const endT = parseInt(endInput.value);
    if (isNaN(startT) || isNaN(endT) || startT > endT) {
        return alert("Invalid frame range.");
    }
    
    if (btn) {
        btn.innerText = "Fixing...";
        btn.disabled = true;
    }
    
    try {
        const body = {
            experiment: state.selectedExp,
            cell_id: state.selectedCell,
            start_t: startT,
            end_t: endT,
            is_local: state.isLocalEdit,
            channel: state.channel
        };
        if (state.isLocalEdit) {
            body.film = state.localFilmId;
        } else {
            body.sequence = state.selectedSequence;
        }
        const res = await fetch('/api/autofix_masks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const contentType = res.headers.get('content-type') || '';
        const data = contentType.includes('application/json')
            ? await res.json()
            : { status: 'error', message: `Server returned HTTP ${res.status} instead of JSON.` };
        if (!res.ok) {
            throw new Error(data.message || `Auto-fix request failed (HTTP ${res.status}).`);
        }
        if (data.status === 'success') {
            for (let t = startT; t <= endT; t++) {
                if (data.masks && data.masks[t] !== undefined) {
                    state.cellMasks[t] = data.masks[t];
                }
            }
            displayFrame();
            alert(`Successfully auto-fixed ${data.fixed_count} mask(s) from frame ${startT} to ${endT}.`);
        } else {
            alert("Auto-fix failed: " + data.message);
        }
    } catch (e) {
        alert(e.message || e);
    } finally {
        if (btn) {
            btn.innerText = "Run Auto-Fix";
            btn.disabled = false;
        }
    }
}
