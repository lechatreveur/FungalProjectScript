function renderGallery() {
    const container = document.getElementById('stripContainer');
    if (!container) return;
    container.innerHTML = '';
    
    if (state.selectedCell === null || state.numFrames === 0) {
        container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Select a cell to view strip crops...</span>';
        return;
    }
    
    const start1 = parseInt(document.getElementById('septumStartInput') ? document.getElementById('septumStartInput').value : NaN);
    const end1 = parseInt(document.getElementById('septumEndInput') ? document.getElementById('septumEndInput').value : NaN);
    const has1 = document.getElementById('hasSeptumCheckbox') ? document.getElementById('hasSeptumCheckbox').checked : false;
    
    const start2 = parseInt(document.getElementById('septumStartInput2') ? document.getElementById('septumStartInput2').value : NaN);
    const end2 = parseInt(document.getElementById('septumEndInput2') ? document.getElementById('septumEndInput2').value : NaN);
    const has2 = document.getElementById('hasSeptumCheckbox2') ? document.getElementById('hasSeptumCheckbox2').checked : false;
    
    const modeParam = state.isLocalEdit ? `film=${state.localFilmId}` : `sequence=${state.selectedSequence}`;
    const ts = Date.now();
    const stripUrl = `/api/cell_strip_image?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&channel=${state.channel}&_ts=${ts}`;
    
    for (let t = 0; t < state.numFrames; t++) {
        const img = document.createElement('div');
        img.className = 'strip-crop';
        if (t === state.currentFrame) img.classList.add('active');
        
        if (has1) {
            if (!isNaN(start1) && t === start1) img.classList.add('septum-start-frame');
            if (!isNaN(end1) && t === end1) img.classList.add('septum-end-frame');
            if (!isNaN(start1) && !isNaN(end1) && t >= start1 && t <= end1) img.classList.add('septum-during-frame');
        }
        
        if (has2) {
            if (!isNaN(start2) && t === start2) img.classList.add('septum-start-frame-2');
            if (!isNaN(end2) && t === end2) img.classList.add('septum-end-frame-2');
            if (!isNaN(start2) && !isNaN(end2) && t >= start2 && t <= end2) img.classList.add('septum-during-frame-2');
        }
        
        img.style.backgroundImage = `url('${stripUrl}')`;
        img.style.backgroundPosition = `-${t * 60}px 0px`;
        img.style.backgroundSize = `${state.numFrames * 60}px 60px`;
        img.style.width = '60px';
        img.style.height = '60px';
        img.style.flexShrink = '0';
        img.style.display = 'inline-block';
        
        img.onclick = async () => {
            if (state.galleryClickMode === 'start1') {
                state.currentFrame = t;
                const slider = document.getElementById('timeSlider');
                if (slider) slider.value = t;
                await displayFrame();
                
                const cb1 = document.getElementById('hasSeptumCheckbox');
                if (cb1) cb1.checked = true;
                const divCont1 = document.getElementById('divisionIntervalContainer');
                if (divCont1) divCont1.style.display = 'flex';
                const st1 = document.getElementById('septumStartInput');
                if (st1) st1.value = t;
                
                await saveSeptumLabels();
                state.galleryClickMode = 'nav';
                updateGalleryClickModeButtons();
                renderGallery();
            } else if (state.galleryClickMode === 'end1') {
                state.currentFrame = t;
                const slider = document.getElementById('timeSlider');
                if (slider) slider.value = t;
                await displayFrame();
                
                const cb1 = document.getElementById('hasSeptumCheckbox');
                if (cb1) cb1.checked = true;
                const divCont1 = document.getElementById('divisionIntervalContainer');
                if (divCont1) divCont1.style.display = 'flex';
                const en1 = document.getElementById('septumEndInput');
                if (en1) en1.value = t;
                
                await saveSeptumLabels();
                state.galleryClickMode = 'nav';
                updateGalleryClickModeButtons();
                renderGallery();
            } else if (state.galleryClickMode === 'start2') {
                state.currentFrame = t;
                const slider = document.getElementById('timeSlider');
                if (slider) slider.value = t;
                await displayFrame();
                
                const cb2 = document.getElementById('hasSeptumCheckbox2');
                if (cb2) cb2.checked = true;
                const divCont2 = document.getElementById('divisionIntervalContainer2');
                if (divCont2) divCont2.style.display = 'flex';
                const st2 = document.getElementById('septumStartInput2');
                if (st2) st2.value = t;
                
                await saveSeptumLabels();
                state.galleryClickMode = 'nav';
                updateGalleryClickModeButtons();
                renderGallery();
            } else if (state.galleryClickMode === 'end2') {
                state.currentFrame = t;
                const slider = document.getElementById('timeSlider');
                if (slider) slider.value = t;
                await displayFrame();
                
                const cb2 = document.getElementById('hasSeptumCheckbox2');
                if (cb2) cb2.checked = true;
                const divCont2 = document.getElementById('divisionIntervalContainer2');
                if (divCont2) divCont2.style.display = 'flex';
                const en2 = document.getElementById('septumEndInput2');
                if (en2) en2.value = t;
                
                await saveSeptumLabels();
                state.galleryClickMode = 'nav';
                updateGalleryClickModeButtons();
                renderGallery();
            } else {
                state.currentFrame = t;
                const slider = document.getElementById('timeSlider');
                if (slider) slider.value = t;
                await displayFrame();
            }
        };
        container.appendChild(img);
    }
}

function updateGalleryHighlight() {
    const crops = document.querySelectorAll('.strip-crop');
    crops.forEach((crop, idx) => {
        crop.classList.toggle('active', idx === state.currentFrame);
    });
}

function updateGalleryClickModeButtons() {
    const navBtn = document.getElementById('galleryClickNavBtn');
    const start1Btn = document.getElementById('galleryClickStartBtn');
    const end1Btn = document.getElementById('galleryClickEndBtn');
    const start2Btn = document.getElementById('galleryClickStartBtn2');
    const end2Btn = document.getElementById('galleryClickEndBtn2');
    
    if (navBtn) navBtn.classList.toggle('active', state.galleryClickMode === 'nav');
    if (start1Btn) start1Btn.classList.toggle('active', state.galleryClickMode === 'start1');
    if (end1Btn) end1Btn.classList.toggle('active', state.galleryClickMode === 'end1');
    if (start2Btn) start2Btn.classList.toggle('active', state.galleryClickMode === 'start2');
    if (end2Btn) end2Btn.classList.toggle('active', state.galleryClickMode === 'end2');
}
