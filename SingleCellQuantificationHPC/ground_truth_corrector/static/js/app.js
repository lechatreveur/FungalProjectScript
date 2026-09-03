document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupCanvasInteractions();
    loadExperiments();
});

function setupEventListeners() {
    // Experiment & Sequence selection
    const expSelect = document.getElementById('expSelect');
    if (expSelect) {
        expSelect.onchange = (e) => {
            state.selectedExp = e.target.value;
            loadFilmsAndSequences(state.selectedExp);
        };
    }

    const seqSelect = document.getElementById('sequenceSelect');
    if (seqSelect) {
        seqSelect.onchange = (e) => {
            state.selectedSequence = e.target.value;
            loadCells(state.selectedExp, state.selectedSequence);
        };
    }

    // Timeline Slider
    const timeSlider = document.getElementById('timeSlider');
    if (timeSlider) {
        timeSlider.oninput = (e) => {
            state.currentFrame = parseInt(e.target.value);
            displayFrame();
        };
    }

    // Navigation buttons
    const firstBtn = document.getElementById('firstFrameBtn');
    const lastBtn = document.getElementById('lastFrameBtn');
    const prevBtn = document.getElementById('prevFrameBtn');
    const nextBtn = document.getElementById('nextFrameBtn');
    const playBtn = document.getElementById('playBtn');

    if (firstBtn) firstBtn.onclick = async () => { await flushPendingAutosave(); state.currentFrame = 0; if (timeSlider) timeSlider.value = 0; displayFrame(); };
    if (lastBtn) lastBtn.onclick = async () => { await flushPendingAutosave(); state.currentFrame = Math.max(0, state.numFrames - 1); if (timeSlider) timeSlider.value = state.currentFrame; displayFrame(); };
    if (prevBtn) prevBtn.onclick = async () => {
        if (state.currentFrame > 0) {
            await flushPendingAutosave();
            state.currentFrame--;
            if (timeSlider) timeSlider.value = state.currentFrame;
            displayFrame();
        }
    };
    if (nextBtn) nextBtn.onclick = async () => {
        if (state.currentFrame < state.numFrames - 1) {
            await flushPendingAutosave();
            state.currentFrame++;
            if (timeSlider) timeSlider.value = state.currentFrame;
            displayFrame();
        }
    };

    if (playBtn) {
        playBtn.onclick = () => {
            state.isPlaying = !state.isPlaying;
            playBtn.innerText = state.isPlaying ? '⏸' : '▶';
            if (state.isPlaying) {
                state.playInterval = setInterval(() => {
                    if (state.currentFrame < state.numFrames - 1) {
                        state.currentFrame++;
                    } else {
                        state.currentFrame = 0;
                    }
                    if (timeSlider) timeSlider.value = state.currentFrame;
                    displayFrame();
                }, 350);
            } else {
                clearInterval(state.playInterval);
            }
        };
    }

    // Channel toggles
    const chanBfBtn = document.getElementById('chanBfBtn');
    const chanGfpBtn = document.getElementById('chanGfpBtn');
    if (chanBfBtn) {
        chanBfBtn.onclick = () => {
            state.channel = 'bf';
            chanBfBtn.classList.add('active');
            if (chanGfpBtn) chanGfpBtn.classList.remove('active');
            displayFrame();
        };
    }
    if (chanGfpBtn) {
        chanGfpBtn.onclick = () => {
            state.channel = 'gfp';
            chanGfpBtn.classList.add('active');
            if (chanBfBtn) chanBfBtn.classList.remove('active');
            displayFrame();
        };
    }

    // View Mode toggle
    const viewModeBtn = document.getElementById('viewModeBtn');
    if (viewModeBtn) {
        viewModeBtn.onclick = () => {
            state.viewMode = state.viewMode === 'single' ? 'population' : 'single';
            state.userHasPanned = false;
            resetView();
            if (state.viewMode === 'population') {
                viewModeBtn.innerText = 'Population';
                viewModeBtn.classList.add('active');
                selectTool('select');
                const brushBtn = document.getElementById('toolBrushBtn');
                const eraserBtn = document.getElementById('toolEraserBtn');
                if (brushBtn) { brushBtn.disabled = true; brushBtn.style.opacity = 0.5; }
                if (eraserBtn) { eraserBtn.disabled = true; eraserBtn.style.opacity = 0.5; }
            } else {
                viewModeBtn.innerText = 'Single Cell';
                viewModeBtn.classList.remove('active');
                const brushBtn = document.getElementById('toolBrushBtn');
                const eraserBtn = document.getElementById('toolEraserBtn');
                if (brushBtn) { brushBtn.disabled = false; brushBtn.style.opacity = 1.0; }
                if (eraserBtn) { eraserBtn.disabled = false; eraserBtn.style.opacity = 1.0; }
            }
            displayFrame();
        };
    }

    // Tool Selection
    const selectTool = (t) => {
        state.tool = t;
        const toolSelectBtn = document.getElementById('toolSelectBtn');
        const toolBrushBtn = document.getElementById('toolBrushBtn');
        const toolEraserBtn = document.getElementById('toolEraserBtn');
        const brushControls = document.getElementById('brushControls');
        
        [toolSelectBtn, toolBrushBtn, toolEraserBtn].forEach(b => { if (b) b.classList.remove('active'); });
        if (t === 'select' && toolSelectBtn) toolSelectBtn.classList.add('active');
        if (t === 'brush' && toolBrushBtn) toolBrushBtn.classList.add('active');
        if (t === 'eraser' && toolEraserBtn) toolEraserBtn.classList.add('active');
        
        if (brushControls) brushControls.style.display = (t === 'brush' || t === 'eraser') ? 'flex' : 'none';
        displayFrame();
    };

    const toolSelectBtn = document.getElementById('toolSelectBtn');
    const toolBrushBtn = document.getElementById('toolBrushBtn');
    const toolEraserBtn = document.getElementById('toolEraserBtn');
    if (toolSelectBtn) toolSelectBtn.onclick = () => selectTool('select');
    if (toolBrushBtn) toolBrushBtn.onclick = () => selectTool('brush');
    if (toolEraserBtn) toolEraserBtn.onclick = () => selectTool('eraser');

    // Brush Size
    const brushSlider = document.getElementById('brushSizeSlider');
    const brushLbl = document.getElementById('brushSizeLabel');
    if (brushSlider) {
        brushSlider.oninput = (e) => {
            state.brushSize = parseInt(e.target.value);
            if (brushLbl) brushLbl.innerText = `${state.brushSize}px`;
        };
    }

    // Mask Opacity
    const opacitySlider = document.getElementById('maskOpacitySlider');
    const opacityLbl = document.getElementById('maskOpacityLabel');
    if (opacitySlider) {
        opacitySlider.oninput = (e) => {
            const val = parseInt(e.target.value, 10);
            state.maskOpacity = val / 100.0;
            if (opacityLbl) opacityLbl.innerText = `${val}%`;
            displayFrame();
        };
    }

    // Editing Actions
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.onclick = async () => {
            if (!state.selectedCell) return;
            pushHistory();
            const cur = state.currentFrame;
            state.cellMasks[cur] = "";
            displayFrame();
            await saveCurrentCell(cur);
        };
    }

    const undoBtn = document.getElementById('undoBtn');
    if (undoBtn) {
        undoBtn.onclick = async () => {
            if (state.drawingHistory.length > 0) {
                const cur = state.currentFrame;
                state.cellMasks[cur] = state.drawingHistory.pop();
                displayFrame();
                await saveCurrentCell(cur);
            }
        };
    }

    const usePrevBtn = document.getElementById('usePrevSegmentBtn');
    if (usePrevBtn) {
        usePrevBtn.onclick = async () => {
            if (!state.selectedCell || state.currentFrame === 0) return;
            const prevMask = state.cellMasks[state.currentFrame - 1];
            if (prevMask) {
                pushHistory();
                const cur = state.currentFrame;
                state.cellMasks[cur] = prevMask;
                displayFrame();
                await saveCurrentCell(cur);
            }
        };
    }

    // QC buttons
    const markGoodBtn = document.getElementById('btnMarkGood');
    const markBadBtn = document.getElementById('btnMarkBad');
    const markCorrBtn = document.getElementById('btnMarkCorrected');
    const markMisBtn = document.getElementById('btnMarkMistracked');
    if (markGoodBtn) markGoodBtn.onclick = () => setQC('good');
    if (markBadBtn) markBadBtn.onclick = () => setQC('bad');
    if (markCorrBtn) markCorrBtn.onclick = () => setQC('corrected');
    if (markMisBtn) markMisBtn.onclick = () => setQC('mistracked');

    // Manual Save Button
    const saveBtn = document.getElementById('btnSaveCell');
    if (saveBtn) saveBtn.onclick = () => saveCurrentCell();

    // Export All Keyframes to Training Set
    const exportBtn = document.getElementById('btnExportAllTraining');
    if (exportBtn) {
        exportBtn.onclick = async () => {
            exportBtn.disabled = true;
            exportBtn.innerText = 'Exporting... ⏳';
            try {
                const res = await fetch('/api/export_training_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        experiment: state.selectedExp,
                        sequence: state.selectedSequence
                    })
                });
                const data = await res.json();
                alert(`Export Complete!\nExported ${data.total_keyframes_exported} keyframes to:\n${data.destination}`);
            } catch (e) {
                alert(`Export Error: ${e}`);
            } finally {
                exportBtn.disabled = false;
                exportBtn.innerText = '📦 Export Training Set';
            }
        };
    }

let isSpaceKeyDown = false;

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' || e.key === ' ') {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                isSpaceKeyDown = true;
                const vp = document.getElementById('canvasViewport');
                if (vp) vp.style.cursor = 'crosshair';
            }
        }
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        
        if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
            if (prevBtn) prevBtn.click();
        } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
            if (nextBtn) nextBtn.click();
        } else if (e.key === 's' || e.key === 'S') {
            selectTool('select');
        } else if (e.key === 'b' || e.key === 'B') {
            if (state.viewMode === 'single') selectTool('brush');
        } else if (e.key === 'e' || e.key === 'E') {
            if (state.viewMode === 'single') selectTool('eraser');
        } else if (e.key === 'p' || e.key === 'P') {
            if (usePrevBtn) usePrevBtn.click();
        } else if (e.key === 'f' || e.key === 'F') {
            resetView();
        } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
            if (undoBtn) undoBtn.click();
        }
    });

    window.addEventListener('keyup', (e) => {
        if (e.code === 'Space' || e.key === ' ') {
            isSpaceKeyDown = false;
            const vp = document.getElementById('canvasViewport');
            if (vp) vp.style.cursor = '';
        }
    });
}

function pushHistory() {
    state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
    if (state.drawingHistory.length > 20) state.drawingHistory.shift();
}

function setupCanvasInteractions() {
    const viewport = document.getElementById('canvasViewport');
    if (!viewport) return;

    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        state.userHasPanned = true;
        const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
        const newScale = Math.max(0.05, Math.min(10.0, state.scale * zoomFactor));
        
        const rect = viewport.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        state.panX = mouseX - (mouseX - state.panX) * (newScale / state.scale);
        state.panY = mouseY - (mouseY - state.panY) * (newScale / state.scale);
        state.scale = newScale;
        
        if (canvas) {
            canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
        }
    });

    viewport.addEventListener('mousedown', async (e) => {
        const coords = getCanvasMouseCoords(e);
        if (isSpaceKeyDown || e.code === 'Space' || e.key === ' ') {
            e.preventDefault();
            e.stopPropagation();
            const scaleFactor = state.viewMode === 'population' ? ((state.imgWidth || 2000) / (canvas.width || 1000)) : 1.0;
            const imgX = coords.x * scaleFactor;
            const imgY = coords.y * scaleFactor;
            await activateCellAtCoords(imgX, imgY);
            return;
        }

        if (e.button === 1 || e.shiftKey || state.tool === 'pan') {
            state.isPanning = true;
            state.startX = e.clientX - state.panX;
            state.startY = e.clientY - state.panY;
            return;
        }

        if (e.button === 0) {
            const coords = getCanvasMouseCoords(e);
            
            // Segment click in select tool mode
            if (state.tool === 'select') {
                const modeParam = state.selectedSequence ? `sequence=${state.selectedSequence}` : `film=${state.selectedFilm || ''}`;
                try {
                    const res = await fetch(`/api/segment_at_coords?experiment=${state.selectedExp}&${modeParam}&t=${state.currentFrame}&x=${coords.x}&y=${coords.y}`);

                    const segData = await res.json();
                    if (segData.rle && segData.label > 0) {
                        pushHistory();
                        const cur = state.currentFrame;
                        if ((e.altKey || e.ctrlKey || e.metaKey) && state.cellMasks[cur]) {
                            state.cellMasks[cur] = unionRle(
                                state.cellMasks[cur],
                                segData.rle,
                                state.imgWidth,
                                state.imgHeight
                            );
                        } else {
                            state.cellMasks[cur] = segData.rle;
                        }
                        displayFrame();
                        await saveCurrentCell(cur);
                    }
                } catch (err) {
                    console.error("Segment click error:", err);
                }
                return;
            }

            // Brush / Eraser drawing in Single Cell mode
            if (state.viewMode === 'single' && (state.tool === 'brush' || state.tool === 'eraser')) {
                state.isDrawing = true;
                pushHistory();
                applyBrush(coords.x, coords.y, state.tool === 'brush');
            }
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (state.isPanning) {
            state.panX = e.clientX - state.startX;
            state.panY = e.clientY - state.startY;
            if (canvas) {
                canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
            }
            return;
        }

        if (state.isDrawing && state.viewMode === 'single') {
            const coords = getCanvasMouseCoords(e);
            applyBrush(coords.x, coords.y, state.tool === 'brush');
        }
    });

    window.addEventListener('mouseup', () => {
        if (state.isPanning) state.isPanning = false;
        if (state.isDrawing) {
            state.isDrawing = false;
            triggerAutosave(state.currentFrame);
        }
    });
}

function applyBrush(cx, cy, isAdd) {
    const W = state.imgWidth;
    const H = state.imgHeight;
    const radius = Math.max(1, Math.floor(state.brushSize / 2));
    
    // Decode current mask to boolean array
    let mask = new Uint8Array(W * H);
    const curRle = state.cellMasks[state.currentFrame];
    if (curRle) {
        const nums = curRle.trim().split(/\s+/).map(Number);
        for (let i = 0; i < nums.length; i += 2) {
            const start = nums[i] - 1;
            const length = nums[i + 1];
            for (let j = 0; j < length; j++) {
                if (start + j < mask.length) mask[start + j] = 1;
            }
        }
    }

    const minX = Math.max(0, Math.floor(cx - radius));
    const maxX = Math.min(W - 1, Math.ceil(cx + radius));
    const minY = Math.max(0, Math.floor(cy - radius));
    const maxY = Math.min(H - 1, Math.ceil(cy + radius));
    const r2 = radius * radius;

    for (let x = minX; x <= maxX; x++) {
        for (let y = minY; y <= maxY; y++) {
            const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if (d2 <= r2) {
                // Fortran order index: idx = x * H + y
                const idx = x * H + y;
                mask[idx] = isAdd ? 1 : 0;
            }
        }
    }

    // Re-encode to Fortran RLE
    state.cellMasks[state.currentFrame] = encodeRle(mask);
    displayFrame();
}

function encodeRle(mask) {
    const pairs = [];
    let inRun = false;
    let runStart = 0;

    for (let i = 0; i < mask.length; i++) {
        if (mask[i] === 1 && !inRun) {
            inRun = true;
            runStart = i + 1; // 1-indexed
        } else if (mask[i] === 0 && inRun) {
            inRun = false;
            pairs.push(`${runStart} ${i + 1 - runStart}`);
        }
    }
    if (inRun) {
        pairs.push(`${runStart} ${mask.length + 1 - runStart}`);
    }
    return pairs.join(' ');
}

function unionRle(rle1, rle2, W, H) {
    if (!rle1) return rle2 || "";
    if (!rle2) return rle1 || "";
    const arr = new Uint8Array(W * H);
    const decodeToArr = (rleStr) => {
        const nums = rleStr.trim().split(/\s+/).map(Number);
        for (let i = 0; i < nums.length; i += 2) {
            const start = nums[i] - 1;
            const length = nums[i + 1];
            for (let j = 0; j < length; j++) {
                if (start + j < arr.length) arr[start + j] = 1;
            }
        }
    };
    decodeToArr(rle1);
    decodeToArr(rle2);
    return encodeRle(arr);
}
