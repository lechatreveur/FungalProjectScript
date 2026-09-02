function loadImage(url) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = url;
    });
}

function updateKeyframeBadge() {
    const badge = document.getElementById('keyframeBadge');
    const timeLbl = document.getElementById('keyframeTimeLabel');
    const kf = (state.keyframes && state.keyframes[state.currentFrame]) ? state.keyframes[state.currentFrame] : null;
    
    if (kf) {
        const text = `Film: ${kf.film} | ${kf.keyframe_pos} (t=${kf.local_t})`;
        if (badge) badge.innerText = text;
        if (timeLbl) timeLbl.innerText = `t=${kf.local_t} (${kf.keyframe_pos})`;
    } else {
        if (badge) badge.innerText = `Keyframe ${state.currentFrame + 1}/${state.numFrames || 1}`;
        if (timeLbl) timeLbl.innerText = `-`;
    }
}

let renderToken = 0;

async function displayFrame() {
    const token = ++renderToken;
    const timeLbl = document.getElementById('currentTimeLabel');
    if (timeLbl) timeLbl.innerText = `kf=${state.currentFrame}`;
    
    updateKeyframeBadge();
    
    const kf = (state.keyframes && state.keyframes[state.currentFrame]) ? state.keyframes[state.currentFrame] : null;
    const localFilmLbl = document.getElementById('localFilmLabel');
    if (localFilmLbl && kf) {
        localFilmLbl.innerText = kf.film;
    }

    const modeParam = state.isLocalEdit
        ? `film=${state.localFilmId}`
        : (state.selectedSequence ? `sequence=${state.selectedSequence}` : `film=${state.selectedFilm || ''}`);

    const ver = state.cacheVer || 1;
    if (state.viewMode === 'population') {
        const popUrl = `/api/population_frame?experiment=${state.selectedExp}&${modeParam}&t=${state.currentFrame}&v=${ver}`;
        const img = await loadImage(popUrl);
        if (token !== renderToken || !img || !canvas || !ctx) return;
        
        canvas.width = 1000;
        canvas.height = 1000;
        if (!state.userHasPanned) {
            resetView();
        }
        canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        prefetchNeighbors(modeParam, ver);
        return;
    }

    const imgUrl = `/api/frame_image?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell || ''}&t=${state.currentFrame}&channel=${state.channel}&v=${ver}`;
    const promises = [loadImage(imgUrl)];

    if (state.tool === 'select') {
        const boundsUrl = `/api/frame_boundaries?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell || ''}&t=${state.currentFrame}&v=${ver}`;
        promises.push(loadImage(boundsUrl));
    } else {
        promises.push(Promise.resolve(null));
    }

    const [img, boundariesImg] = await Promise.all(promises);
    if (token !== renderToken || !img || !canvas || !ctx) return;

    canvas.width = state.imgWidth;
    canvas.height = state.imgHeight;
    if (!state.userHasPanned) {
        resetView();
    }
    canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (boundariesImg) {
        ctx.drawImage(boundariesImg, 0, 0, canvas.width, canvas.height);
    }

    drawMask();
    prefetchNeighbors(modeParam, ver);
}

function prefetchNeighbors(modeParam, ver) {
    const cur = state.currentFrame;
    const maxF = state.numFrames - 1;
    const targets = [];
    if (cur > 0) targets.push(cur - 1);
    if (cur < maxF) targets.push(cur + 1);

    for (const t of targets) {
        if (state.viewMode === 'population') {
            const preImg = new Image();
            preImg.src = `/api/population_frame?experiment=${state.selectedExp}&${modeParam}&t=${t}&v=${ver}`;
        } else {
            const preImg = new Image();
            preImg.src = `/api/frame_image?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell || ''}&t=${t}&channel=${state.channel}&v=${ver}`;
        }
    }
}

function drawMask() {
    if (!ctx || !canvas) return;
    const currentRle = state.cellMasks[state.currentFrame];
    if (!currentRle) return;

    const rleNums = currentRle.trim().split(/\s+/).map(Number);
    if (rleNums.length < 2) return;

    const W = state.imgWidth;
    const H = state.imgHeight;
    const imgData = ctx.getImageData(0, 0, W, H);
    const data = imgData.data;

    // Colour the selected cell's mask by its stable identity so it matches the
    // population / boundary views; fall back to the old blue if no cell id.
    const rgb = (typeof selectedCellColorRGB === 'function' && selectedCellColorRGB()) || [59, 130, 246];
    const [cr, cg, cb] = rgb;

    const alpha = (state.maskOpacity !== undefined) ? state.maskOpacity : 0.40;
    const invAlpha = 1.0 - alpha;

    for (let i = 0; i < rleNums.length; i += 2) {
        const start = rleNums[i] - 1;
        const length = rleNums[i + 1];
        for (let j = 0; j < length; j++) {
            const idx = start + j;
            const x = Math.floor(idx / H);
            const y = idx % H;
            if (x < W && y < H) {
                const pixelIdx = (y * W + x) * 4;
                data[pixelIdx]     = Math.round(data[pixelIdx] * invAlpha + cr * alpha);
                data[pixelIdx + 1] = Math.round(data[pixelIdx + 1] * invAlpha + cg * alpha);
                data[pixelIdx + 2] = Math.round(data[pixelIdx + 2] * invAlpha + cb * alpha);
                data[pixelIdx + 3] = 255;
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function getCanvasMouseCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / state.scale;
    const y = (e.clientY - rect.top) / state.scale;
    return { x: Math.max(0, Math.min(state.imgWidth - 1, x)), y: Math.max(0, Math.min(state.imgHeight - 1, y)) };
}

function resetView() {
    const viewport = document.getElementById('canvasViewport');
    if (!viewport) return;
    const vpW = viewport.clientWidth;
    const vpH = viewport.clientHeight;
    if (vpW <= 0 || vpH <= 0) return;

    const imgW = state.viewMode === 'population' ? 1000 : (state.imgWidth || 2000);
    const imgH = state.viewMode === 'population' ? 1000 : (state.imgHeight || 2000);

    const scale = Math.min(vpW / imgW, vpH / imgH) * 0.96;
    state.scale = scale;
    state.panX = (vpW - imgW * scale) / 2;
    state.panY = (vpH - imgH * scale) / 2;
    state.userHasPanned = false;
    if (canvas) {
        canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
    }
}

window.addEventListener('resize', () => {
    if (!state.userHasPanned) {
        resetView();
    }
});
