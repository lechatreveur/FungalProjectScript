// Stable per-cell colour. Ported to match the server:
//   fnv1a32      == gt_frames_service.fnv1a_32
//   idToColor    == gt_frames_service.id_to_color (server returns BGR, here RGB)
// so a cell shows the same colour in the client mask overlay and in the
// server-rendered population / boundary views. See docs/FLASK_APPS.md
// "Shared UI conventions".

function fnv1a32(s) {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);           // global_cell_id values are ASCII
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h >>> 0;
}

function idToColor(id) {
    // Knuth multiplicative hash -> hue; s=0.8, v=0.95. Returns [r, g, b] 0-255.
    const val = ((id >>> 0) * 2654435761) % 4294967296;
    const h = (val % 360) / 360.0;
    const s = 0.8, v = 0.95;
    let i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    i = ((i % 6) + 6) % 6;
    let r, g, b;
    if (i === 0) { r = v; g = t; b = p; }
    else if (i === 1) { r = q; g = v; b = p; }
    else if (i === 2) { r = p; g = v; b = t; }
    else if (i === 3) { r = p; g = q; b = v; }
    else if (i === 4) { r = t; g = p; b = v; }
    else { r = v; g = p; b = q; }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// identity: a number (single-film tracked local id) or a string (global_cell_id).
function stableColorKey(identity) {
    if (typeof identity === 'number' && Number.isFinite(identity)) return identity >>> 0;
    return fnv1a32(String(identity));
}

// The stable colour for the currently selected cell, or null if none.
// state.selectedCell is already the global_cell_id (see cells.js selectCell()).
function selectedCellColorRGB() {
    if (state.selectedCell === null || state.selectedCell === undefined || state.selectedCell === '') {
        return null;
    }
    return idToColor(stableColorKey(String(state.selectedCell)));
}
