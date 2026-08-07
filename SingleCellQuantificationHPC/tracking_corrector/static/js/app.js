window.addEventListener('DOMContentLoaded', async () => {
    await loadExperiments();
    setupEventListeners();
});

function setupEventListeners() {
    const slider = document.getElementById('timeSlider');
    if (slider) {
        slider.addEventListener('input', (e) => {
            state.currentFrame = parseInt(e.target.value);
            displayFrame();
        });
    }

    const prevBtn = document.getElementById('prevBtn');
    if (prevBtn) {
        prevBtn.onclick = () => {
            if (state.currentFrame > 0) {
                state.currentFrame--;
                if (slider) slider.value = state.currentFrame;
                displayFrame();
            }
        };
    }

    const nextBtn = document.getElementById('nextBtn');
    if (nextBtn) {
        nextBtn.onclick = () => {
            if (state.currentFrame < state.numFrames - 1) {
                state.currentFrame++;
                if (slider) slider.value = state.currentFrame;
                displayFrame();
            }
        };
    }

    const playBtn = document.getElementById('playBtn');
    if (playBtn) {
        playBtn.onclick = () => {
            if (state.isPlaying) {
                clearInterval(state.playInterval);
                state.isPlaying = false;
                playBtn.innerText = 'Play';
            } else {
                state.isPlaying = true;
                playBtn.innerText = 'Pause';
                state.playInterval = setInterval(() => {
                    if (state.currentFrame < state.numFrames - 1) {
                        state.currentFrame++;
                        if (slider) slider.value = state.currentFrame;
                        displayFrame();
                    } else {
                        state.currentFrame = 0;
                        if (slider) slider.value = 0;
                        displayFrame();
                    }
                }, 150);
            }
        };
    }

    const expSelect = document.getElementById('experimentSelect');
    if (expSelect) {
        expSelect.onchange = async (e) => {
            state.selectedExp = e.target.value;
            await loadFilmsAndSequences(state.selectedExp);
        };
    }

    const seqSelect = document.getElementById('sequenceSelect');
    if (seqSelect) {
        seqSelect.onchange = (e) => {
            state.selectedSequence = e.target.value;
            state.selectedCell = null;
            loadCells(state.selectedExp, state.selectedSequence);
        };
    }
    
    const trackNewBtn = document.getElementById('trackNewCellBtn');
    if (trackNewBtn) {
        trackNewBtn.onclick = async () => {
            const targetFilm = state.isEditingLink ? state.linkEditFilmName : state.localFilmId;
            if (!targetFilm) return alert("Select a film first.");
            
            try {
                const res = await fetch('/api/create_new_cell', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        experiment: state.selectedExp,
                        film: targetFilm
                    })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    state.prevGlobalCell = state.selectedCell;
                    state.prevLinkEditFilmIdx = state.linkEditFilmIdx;
                    state.prevLinkEditFilmName = state.linkEditFilmName;
                    state.isEditingLink = false;
                    
                    await selectLocalCell(data.cell_id, targetFilm);
                    alert(`Created custom local cell #${data.cell_id} in ${targetFilm}.\n\nInstructions:\n1. Paint the seed mask on Frame 0 using the Brush tool.\n2. Click "Save Changes" (or wait for autosave).\n3. Click "💻 Quantify Locally (Seed from CSV)" to track the cell.\n4. Once finished, click "🔙 Return to Linkage" to link it!`);
                } else {
                    alert(data.message);
                }
            } catch(e) {
                alert(e);
            }
        };
    }

    const quantHpcBtn = document.getElementById('btnQuantifyHpc');
    if (quantHpcBtn) {
        quantHpcBtn.onclick = async () => {
            if (!state.selectedCell) return alert("Select a cell first.");
            
            let targetFilm, targetCellId;
            if (state.isLocalEdit) {
                targetFilm = state.localFilmId;
                targetCellId = state.selectedCell;
            } else {
                const activeInfo = getActiveFilmAndLocalCell();
                if (!activeInfo || !activeInfo.film || activeInfo.cellId === null || activeInfo.cellId === -1) {
                    return alert("The selected cell is not tracked or mapped in the currently viewed film.");
                }
                targetFilm = activeInfo.film;
                targetCellId = activeInfo.cellId;
            }
            
            quantHpcBtn.innerText = '⏳ Quantifying...';
            quantHpcBtn.disabled = true;
            
            try {
                const res = await fetch('/api/quantify_on_hpc', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        experiment: state.selectedExp,
                        film: targetFilm,
                        label_id: targetCellId,
                        seed_from_csv: true,
                        track_channel: state.channel
                    })
                });
                const data = await res.json();
                alert(data.message);
                if (data.status === 'success') {
                    if (state.isLocalEdit) {
                        const masksRes = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&film=${targetFilm}&cell_id=${targetCellId}`);
                        const masksData = await masksRes.json();
                        state.cellMasks = masksData.masks || [];
                    } else {
                        await loadCells(state.selectedExp, state.selectedSequence);
                        const curBtn = document.getElementById(`cell-item-${state.selectedCell}`);
                        if (curBtn) curBtn.click();
                    }
                    displayFrame();
                }
            } catch (e) {
                alert(e);
            } finally {
                quantHpcBtn.innerText = '💻 Quantify Locally (Seed from CSV)';
                quantHpcBtn.disabled = false;
            }
        };
    }

    const exitBtn = document.getElementById('btnExitLocalEdit');
    if (exitBtn) {
        exitBtn.onclick = async () => {
            await exitLocalCellEdit();
        };
    }

    const chanBfBtn = document.getElementById('chanBfBtn');
    const chanGfpBtn = document.getElementById('chanGfpBtn');
    if (chanBfBtn) chanBfBtn.onclick = () => { state.channel = 'bf'; updateChannelButtons(); displayFrame(); renderGallery(); };
    if (chanGfpBtn) chanGfpBtn.onclick = () => { state.channel = 'gfp'; updateChannelButtons(); displayFrame(); renderGallery(); };

    const viewModeBtn = document.getElementById('viewModeBtn');
    if (viewModeBtn) {
        viewModeBtn.onclick = () => {
            state.viewMode = state.viewMode === 'single' ? 'population' : 'single';
            if (state.viewMode === 'population') {
                viewModeBtn.innerText = 'Population';
                viewModeBtn.style.backgroundColor = 'var(--accent-primary)';
                selectTool('select');
                const brushBtn = document.getElementById('toolBrushBtn');
                const eraserBtn = document.getElementById('toolEraserBtn');
                if (brushBtn) { brushBtn.disabled = true; brushBtn.style.opacity = 0.5; }
                if (eraserBtn) { eraserBtn.disabled = true; eraserBtn.style.opacity = 0.5; }
            } else {
                viewModeBtn.innerText = 'Single Cell';
                viewModeBtn.style.backgroundColor = '#581c87';
                const brushBtn = document.getElementById('toolBrushBtn');
                const eraserBtn = document.getElementById('toolEraserBtn');
                if (brushBtn) { brushBtn.disabled = false; brushBtn.style.opacity = 1.0; }
                if (eraserBtn) { eraserBtn.disabled = false; eraserBtn.style.opacity = 1.0; }
            }
            displayFrame();
        };
    }

    const toolBtns = {
        select: document.getElementById('toolSelectBtn'),
        brush: document.getElementById('toolBrushBtn'),
        eraser: document.getElementById('toolEraserBtn')
    };

    const selectTool = (t) => {
        state.tool = t;
        const modeLbl = document.getElementById('modeLabel');
        if (state.isEditingLink) {
            state.isEditingLink = false;
            const newBtn = document.getElementById('trackNewCellBtn');
            if (newBtn) newBtn.style.display = 'none';
            if (modeLbl) {
                modeLbl.innerText = t === 'select' ? 'Click-Select' : (t === 'brush' ? 'Brush Draw' : 'Eraser');
                modeLbl.style.color = 'var(--accent-primary)';
            }
        } else if (modeLbl) {
            modeLbl.innerText = t === 'select' ? 'Click-Select' : (t === 'brush' ? 'Brush Draw' : 'Eraser');
        }
        
        Object.keys(toolBtns).forEach(k => {
            if (toolBtns[k]) toolBtns[k].classList.toggle('active', k === t);
        });
        const brushControls = document.getElementById('brushControls');
        if (brushControls) brushControls.style.display = (t === 'select' ? 'none' : 'flex');
        displayFrame();
    };

    if (toolBtns.select) toolBtns.select.onclick = () => selectTool('select');
    if (toolBtns.brush) toolBtns.brush.onclick = () => selectTool('brush');
    if (toolBtns.eraser) toolBtns.eraser.onclick = () => selectTool('eraser');

    const brushSizeSlider = document.getElementById('brushSizeSlider');
    if (brushSizeSlider) {
        brushSizeSlider.addEventListener('input', (e) => {
            state.brushSize = parseInt(e.target.value);
            const lbl = document.getElementById('brushSizeLabel');
            if (lbl) lbl.innerText = `${state.brushSize}px`;
        });
    }

    const undoBtn = document.getElementById('undoBtn');
    if (undoBtn) undoBtn.onclick = undoStroke;
    
    const usePrevBtn = document.getElementById('usePrevSegmentBtn');
    if (usePrevBtn) {
        usePrevBtn.onclick = () => {
            if (state.currentFrame > 0 && state.cellMasks && state.cellMasks.length > 0) {
                const prevRle = state.cellMasks[state.currentFrame - 1] || "";
                state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
                if (state.drawingHistory.length > 20) state.drawingHistory.shift();
                
                state.cellMasks[state.currentFrame] = prevRle;
                displayFrame();
                markDirty();
            } else {
                alert("Cannot copy. No previous frame mask exists.");
            }
        };
    }
    
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.onclick = () => {
            state.cellMasks[state.currentFrame] = "";
            displayFrame();
            markDirty();
        };
    }

    const saveCellBtn = document.getElementById('btnSaveCell');
    if (saveCellBtn) saveCellBtn.onclick = () => saveCorrectedMasks(false);

    const markGoodBtn = document.getElementById('btnMarkGood');
    const markBadBtn = document.getElementById('btnMarkBad');
    const markCorrBtn = document.getElementById('btnMarkCorrected');
    if (markGoodBtn) markGoodBtn.onclick = () => setQC('good');
    if (markBadBtn) markBadBtn.onclick = () => setQC('bad');
    if (markCorrBtn) markCorrBtn.onclick = () => setQC('corrected');

    const hasSeptumCb1 = document.getElementById('hasSeptumCheckbox');
    if (hasSeptumCb1) {
        hasSeptumCb1.onchange = (e) => {
            const divCont1 = document.getElementById('divisionIntervalContainer');
            if (divCont1) divCont1.style.display = e.target.checked ? 'flex' : 'none';
            saveSeptumLabels();
        };
    }
    const startIn1 = document.getElementById('septumStartInput');
    const endIn1 = document.getElementById('septumEndInput');
    const whiteCb1 = document.getElementById('whiteSeptumCheckbox');
    const predSeptBtn = document.getElementById('predictSeptumBtn');
    if (startIn1) startIn1.onchange = () => saveSeptumLabels();
    if (endIn1) endIn1.onchange = () => saveSeptumLabels();
    if (whiteCb1) whiteCb1.onchange = () => saveSeptumLabels();
    if (predSeptBtn) predSeptBtn.onclick = () => runSeptumAi();
    
    const setStartBtn1 = document.getElementById('setSeptumStartBtn');
    const setEndBtn1 = document.getElementById('setSeptumEndBtn');
    if (setStartBtn1) {
        setStartBtn1.onclick = () => {
            if (startIn1) startIn1.value = state.currentFrame;
            saveSeptumLabels();
        };
    }
    if (setEndBtn1) {
        setEndBtn1.onclick = () => {
            if (endIn1) endIn1.value = state.currentFrame;
            saveSeptumLabels();
        };
    }
    
    const hasSeptumCb2 = document.getElementById('hasSeptumCheckbox2');
    if (hasSeptumCb2) {
        hasSeptumCb2.onchange = (e) => {
            const divCont2 = document.getElementById('divisionIntervalContainer2');
            if (divCont2) divCont2.style.display = e.target.checked ? 'flex' : 'none';
            saveSeptumLabels();
        };
    }
    const startIn2 = document.getElementById('septumStartInput2');
    const endIn2 = document.getElementById('septumEndInput2');
    const whiteCb2 = document.getElementById('whiteSeptumCheckbox2');
    if (startIn2) startIn2.onchange = () => saveSeptumLabels();
    if (endIn2) endIn2.onchange = () => saveSeptumLabels();
    if (whiteCb2) whiteCb2.onchange = () => saveSeptumLabels();
    
    const setStartBtn2 = document.getElementById('setSeptumStartBtn2');
    const setEndBtn2 = document.getElementById('setSeptumEndBtn2');
    if (setStartBtn2) {
        setStartBtn2.onclick = () => {
            if (startIn2) startIn2.value = state.currentFrame;
            saveSeptumLabels();
        };
    }
    if (setEndBtn2) {
        setEndBtn2.onclick = () => {
            if (endIn2) endIn2.value = state.currentFrame;
            saveSeptumLabels();
        };
    }

    const galleryClickNavBtn = document.getElementById('galleryClickNavBtn');
    const galleryClickStartBtn = document.getElementById('galleryClickStartBtn');
    const galleryClickEndBtn = document.getElementById('galleryClickEndBtn');
    const galleryClickStartBtn2 = document.getElementById('galleryClickStartBtn2');
    const galleryClickEndBtn2 = document.getElementById('galleryClickEndBtn2');

    if (galleryClickNavBtn) {
        galleryClickNavBtn.onclick = () => {
            state.galleryClickMode = 'nav';
            updateGalleryClickModeButtons();
        };
    }
    if (galleryClickStartBtn) {
        galleryClickStartBtn.onclick = () => {
            state.galleryClickMode = 'start1';
            updateGalleryClickModeButtons();
        };
    }
    if (galleryClickEndBtn) {
        galleryClickEndBtn.onclick = () => {
            state.galleryClickMode = 'end1';
            updateGalleryClickModeButtons();
        };
    }
    if (galleryClickStartBtn2) {
        galleryClickStartBtn2.onclick = () => {
            state.galleryClickMode = 'start2';
            updateGalleryClickModeButtons();
        };
    }
    if (galleryClickEndBtn2) {
        galleryClickEndBtn2.onclick = () => {
            state.galleryClickMode = 'end2';
            updateGalleryClickModeButtons();
        };
    }

    const setAutofixStartBtn = document.getElementById('setAutofixStartBtn');
    const setAutofixEndBtn = document.getElementById('setAutofixEndBtn');
    const runAutofixBtn = document.getElementById('runAutofixBtn');
    if (setAutofixStartBtn) {
        setAutofixStartBtn.onclick = () => {
            const autofixStartInput = document.getElementById('autofixStartInput');
            if (autofixStartInput) autofixStartInput.value = state.currentFrame;
        };
    }
    if (setAutofixEndBtn) {
        setAutofixEndBtn.onclick = () => {
            const autofixEndInput = document.getElementById('autofixEndInput');
            if (autofixEndInput) autofixEndInput.value = state.currentFrame;
        };
    }
    if (runAutofixBtn) runAutofixBtn.onclick = () => runAutofix();

    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' || e.key === ' ') {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT' && e.target.tagName !== 'TEXTAREA') {
                isSpaceKeyDown = true;
            }
        }
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowRight') {
            const nBtn = document.getElementById('nextBtn');
            if (nBtn) nBtn.click();
        } else if (e.key === 'ArrowLeft') {
            const pBtn = document.getElementById('prevBtn');
            if (pBtn) pBtn.click();
        } else if (e.key.toLowerCase() === 's') selectTool('select');
        else if (e.key.toLowerCase() === 'b') selectTool('brush');
        else if (e.key.toLowerCase() === 'e') selectTool('eraser');
        else if (e.key.toLowerCase() === 'p') {
            const uBtn = document.getElementById('usePrevSegmentBtn');
            if (uBtn) uBtn.click();
        } else if (e.key.toLowerCase() === 'z' && e.ctrlKey) undoStroke();
    });

    window.addEventListener('keyup', (e) => {
        if (e.code === 'Space' || e.key === ' ') {
            isSpaceKeyDown = false;
        }
    });

    if (canvasContainer) {
        canvasContainer.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomIntensity = 0.1;
            const containerRect = canvasContainer.getBoundingClientRect();
            const mx = e.clientX - containerRect.left;
            const my = e.clientY - containerRect.top;
            const canvasMouseX = (mx - state.panX) / state.scale;
            const canvasMouseY = (my - state.panY) / state.scale;
            const zoomFactor = e.deltaY < 0 ? (1 + zoomIntensity) : (1 - zoomIntensity);
            state.scale = Math.min(20, Math.max(0.1, state.scale * zoomFactor));
            state.panX = mx - canvasMouseX * state.scale;
            state.panY = my - canvasMouseY * state.scale;
            if (canvas) canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
            updateTransformLabels();
        });

        canvasContainer.addEventListener('mousedown', async (e) => {
            const coords = getCanvasMouseCoords(e);
            if (isSpaceKeyDown || e.code === 'Space' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                await activateCellAtCoords(coords.x, coords.y);
                return;
            }
            if (e.ctrlKey || e.button === 1 || state.tool === 'pan') {
                state.isPanning = true;
                state.startX = e.clientX - state.panX;
                state.startY = e.clientY - state.panY;
            } else if (state.isEditingLink) {
                identifyAndLinkCell(coords.x, coords.y);
            } else if (state.tool === 'select') {
                clickSelectSegment(coords.x, coords.y, e.shiftKey);
            } else if (state.tool === 'brush' || state.tool === 'eraser') {
                state.isDrawing = true;
                state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
                if (state.drawingHistory.length > 20) state.drawingHistory.shift();
                drawStroke(coords.x, coords.y, state.tool === 'brush');
            }
        });
    }

    window.addEventListener('mousemove', (e) => {
        if (state.isPanning) {
            state.panX = e.clientX - state.startX;
            state.panY = e.clientY - state.startY;
            if (canvas) canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
            updateTransformLabels();
        } else if (state.isDrawing) {
            const coords = getCanvasMouseCoords(e);
            drawStroke(coords.x, coords.y, state.tool === 'brush');
        }
    });

    window.addEventListener('mouseup', () => {
        state.isPanning = false;
        state.isDrawing = false;
    });
}
