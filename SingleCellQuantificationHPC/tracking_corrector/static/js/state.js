const state = {
    experiments: [],
    films: [],
    sequences: [],
    cells: [],
    selectedExp: '',
    selectedSequence: '',
    isLocalEdit: false,
    prevGlobalCell: null,
    prevLinkEditFilmIdx: null,
    prevLinkEditFilmName: null,
    isEditingLink: false,
    linkEditFilmIdx: -1,
    linkEditFilmName: '',
    selectedCell: null,
    cellMasks: [],
    numFrames: 0,
    currentFrame: 0,
    channel: 'bf',
    viewMode: 'population',
    tool: 'select',
    brushSize: 10,
    isPlaying: false,
    playInterval: null,
    imgWidth: 2000,
    imgHeight: 2000,
    scale: 1.0,
    panX: 0,
    panY: 0,
    isPanning: false,
    isDrawing: false,
    startX: 0,
    startY: 0,
    drawingHistory: [],
    qc: {},
    suspicious: {},
    autosaveTimer: null,
    
    // Sequence specific
    linkageDetails: {},
    filmBoundaries: [],
    
    // Septum specific
    lastActiveFilm: null,
    lastActiveCellId: null,
    septumOffset: 0
};

let isSpaceKeyDown = false;
let activeModalFilmIdx = -1;
let activeModalFilmName = "";

Object.defineProperty(window, 'canvas', {
    get: () => document.getElementById('imageCanvas'),
    configurable: true
});
Object.defineProperty(window, 'ctx', {
    get: () => {
        const c = document.getElementById('imageCanvas');
        return c ? c.getContext('2d') : null;
    },
    configurable: true
});
Object.defineProperty(window, 'canvasContainer', {
    get: () => document.getElementById('canvasContainer'),
    configurable: true
});

function getCanvasMouseCoords(e) {
    if (!canvasContainer) return { x: 0, y: 0 };
    const containerRect = canvasContainer.getBoundingClientRect();
    const mx = e.clientX - containerRect.left;
    const my = e.clientY - containerRect.top;
    const mouseX = Math.round((mx - state.panX) / state.scale);
    const mouseY = Math.round((my - state.panY) / state.scale);
    return { x: mouseX, y: mouseY };
}

function updateTransformLabels() {
    const scaleLbl = document.getElementById('scaleLabel');
    const panXLbl = document.getElementById('panXLabel');
    const panYLbl = document.getElementById('panYLabel');
    if (scaleLbl) scaleLbl.innerText = state.scale.toFixed(1);
    if (panXLbl) panXLbl.innerText = Math.round(state.panX);
    if (panYLbl) panYLbl.innerText = Math.round(state.panY);
}

function resetView() {
    if (!canvasContainer) return;
    const availableWidth = canvasContainer.clientWidth;
    const availableHeight = canvasContainer.clientHeight;
    if (availableWidth <= 0 || availableHeight <= 0) {
        return;
    }
    const iw = state.viewMode === 'population' ? 1000 : (state.imgWidth || 2000);
    const ih = state.viewMode === 'population' ? 1000 : (state.imgHeight || 2000);
    state.scale = Math.min(
        availableWidth / iw,
        availableHeight / ih
    ) * 0.96;
    state.panX = (canvasContainer.clientWidth - iw * state.scale) / 2;
    state.panY = (canvasContainer.clientHeight - ih * state.scale) / 2;
    if (canvas) {
        canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
    }
    updateTransformLabels();
}

window.addEventListener('resize', () => {
    resetView();
});

function getActiveFilmAndLocalCell() {
    if (state.isLocalEdit) {
        return { film: state.localFilmId, cellId: state.selectedCell, filmIdx: 0 };
    }
    let fIdx = 0;
    if (state.filmBoundaries && state.filmBoundaries.length > 0) {
        for (let i = 0; i < state.filmBoundaries.length; i++) {
            if (state.currentFrame >= state.filmBoundaries[i]) {
                fIdx = i;
            }
        }
    }
    const film = state.linkageDetails && state.linkageDetails.films ? state.linkageDetails.films[fIdx] : null;
    const cellId = state.linkageDetails && state.linkageDetails.local_ids ? state.linkageDetails.local_ids[fIdx] : null;
    return { film, cellId, filmIdx: fIdx };
}

function getFilmSequenceBounds(filmIdx) {
    if (state.isLocalEdit || !state.filmBoundaries || state.filmBoundaries.length === 0) {
        return { start: 0, end: state.numFrames };
    }
    const start = state.filmBoundaries[filmIdx] || 0;
    const end = filmIdx + 1 < state.filmBoundaries.length
        ? state.filmBoundaries[filmIdx + 1]
        : state.numFrames;
    return { start, end };
}

function localFrameToSequenceFrame(localFrame, filmIdx) {
    if (localFrame === null || localFrame === undefined || Number.isNaN(localFrame)) return null;
    return getFilmSequenceBounds(filmIdx).start + Number(localFrame);
}

function sequenceFrameToLocalFrame(sequenceFrame, filmIdx) {
    if (sequenceFrame === null || sequenceFrame === undefined || Number.isNaN(sequenceFrame)) return null;
    const bounds = getFilmSequenceBounds(filmIdx);
    const value = Number(sequenceFrame);
    if (value < bounds.start || value >= bounds.end) return null;
    return value - bounds.start;
}

// ---------------------------------------------------------------------------
// CROSS-FILM SEPTUM RULE
//
// A single cell's linked/stitched sequence can span multiple films (a cell
// division can straddle a film boundary), so the two endpoints of one septum
// interval (start / end) are NOT guaranteed to live in the same film. Do NOT
// write code that assumes "the film the playhead is currently on" is the film
// either endpoint belongs to.
//
// `isLinkedSequenceCell()` tells you whether the active cell is a multi-film
// linked sequence (as opposed to a single-film / local-edit cell, where every
// frame trivially belongs to filmIdx 0 and none of this matters).
//
// IMPORTANT - two different bugs came from getting this wrong, in order:
//   1. Forcing both endpoints through the CURRENTLY ACTIVE film's local
//      bounds when *saving* rejected/mis-saved the endpoint that belonged to
//      the other film (it would appear to vanish right after being set).
//   2. After fixing (1) by sending the frontend's endpoint as a plain
//      sequence-wide frame number, a second bug showed up when *loading*:
//      the backend was decoding each film's own stored value with whichever
//      film's offset happened to belong to the query's entry point, instead
//      of that value's own film's offset. Since offsets are stored
//      independently per film, this made the same cell's endpoints appear to
//      change depending on which film was active when the reload fired.
//
// The fix for (2) lives server-side now: SeptumService.get_septum_alignment()
// (septum_service.py) resolves each film's own local value with that film's
// own stored offset and that film's own position in the sequence
// (`_sequence_film_bounds`), and returns an already-correct, query-entry-
// point-independent sequence frame number, signaled by `offset: 0` in the
// response. So the frontend does NOT need to (and must not try to) re-guess
// which film an endpoint belongs to - see septum.js's `toSequenceFrame` for
// where that value is consumed as-is for linked cells.
// ---------------------------------------------------------------------------

function isLinkedSequenceCell() {
    return !state.isLocalEdit && !!(state.filmBoundaries && state.filmBoundaries.length > 1);
}
