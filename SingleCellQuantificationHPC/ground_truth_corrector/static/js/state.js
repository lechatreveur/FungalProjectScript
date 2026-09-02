const state = {
    experiments: [],
    selectedExp: null,
    sequences: [],
    films: [],
    selectedSequence: null,
    selectedFilm: null,
    cells: [],
    selectedCell: null,
    cellMasks: [],
    keyframes: [],           // 3-keyframe metadata map [{global_t, film, local_t, keyframe_pos, ...}]
    numFrames: 0,
    currentFrame: 0,
    channel: 'bf',
    viewMode: 'population',  // Default Population mode
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
    autosaveTimer: null,
    
    // Linkage / Sequence details
    filmBoundaries: [],
    localFilmId: null,
    isLocalEdit: false,
    linkageDetails: {},
    cacheVer: 1,
    maskOpacity: 0.40
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
