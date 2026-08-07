// ===========================================================================
// SEPTUM ENDPOINT COORDINATE SYSTEMS - READ BEFORE EDITING
//
// Three different frame numberings are in play for a septum endpoint, and
// mixing them up is exactly what caused endpoints to silently disappear:
//
//   sequence frame  - 0-based index into the UI's stitched timeline for the
//                      *whole linked sequence* (all films concatenated in
//                      film order). This is what the gallery strip, the time
//                      slider, and the septumStartInput/septumEndInput text
//                      boxes all display and accept. `state.filmBoundaries`
//                      marks where each film starts within this numbering.
//
//   local frame     - 0-based index into a *single film's* own frame list
//                      (row index into that film's cell_..._masks.csv). This
//                      is what the backend actually stores/operates on
//                      per-film (`start_frame`/`end_frame`).
//
//   aligned frame   - `local frame + a per-film, per-local-cell-id offset`
//                      stored in that film's own JSON (`offsets` dict). This
//                      is what's actually persisted to disk as
//                      `start_aligned`/`end_aligned` per film. Different
//                      films can (and do) carry DIFFERENT stored offsets for
//                      what is conceptually "the same" linked cell - there is
//                      no single per-cell offset, only per-film ones.
//
// RULE: cell division can span multiple stitched films, so a septum's start
// and end endpoints are NOT guaranteed to fall in the same film. Never assume
// "the film the playhead is on right now" (`getActiveFilmAndLocalCell().filmIdx`)
// is the film either endpoint belongs to, and never decode one film's stored
// aligned value using a DIFFERENT film's offset.
//
// This module intentionally does NOT try to resolve "which film does this
// endpoint belong to" itself for linked cells - that was tried and is
// fundamentally ambiguous client-side (see git history / PR discussion if
// curious). Instead:
//   - SAVE: for linked cells we send the plain sequence-wide frame number
//     (what the gallery/slider already display) as `start_aligned`/
//     `end_aligned`, with no local/offset conversion. The backend
//     (`SeptumService.save_septum_label`, septum_service.py) splits that one
//     sequence-wide interval across every film it touches and stores each
//     film's own local frame + that film's own offset in that film's own
//     JSON.
//   - LOAD: for linked cells, the backend (`SeptumService.get_septum_alignment`)
//     reassembles the interval itself - it knows each film's own offset and
//     each film's own position in the sequence, so it can return an
//     already-correct sequence-wide frame number, and signals this with
//     `offset: 0` in the response. This module just uses that value as-is;
//     see `toSequenceFrame` below.
//
// Single-film cells (state.isLocalEdit, or a sequence with only one film)
// have no ambiguity - sequence frame === local frame there - so the code
// below short-circuits to the original local+offset path via
// `isLinkedSequenceCell()` (state.js).
// ===========================================================================

async function loadSeptumLabels(cellId) {
    if (cellId === null) return;
    const current = getActiveFilmAndLocalCell();
    if (!current.film || current.cellId === null || current.cellId === -1) return;
    
    try {
        const res = await fetch(`/api/get_septum_label?experiment=${state.selectedExp}&film=${current.film}&cell_id=${current.cellId}`);
        const data = await res.json();
        
        const hasBox1 = document.getElementById('hasSeptumCheckbox');
        const startIn1 = document.getElementById('septumStartInput');
        const endIn1 = document.getElementById('septumEndInput');
        const whiteBox1 = document.getElementById('whiteSeptumCheckbox');
        const divContainer1 = document.getElementById('divisionIntervalContainer');
        
        const hasBox2 = document.getElementById('hasSeptumCheckbox2');
        const startIn2 = document.getElementById('septumStartInput2');
        const endIn2 = document.getElementById('septumEndInput2');
        const whiteBox2 = document.getElementById('whiteSeptumCheckbox2');
        const divContainer2 = document.getElementById('divisionIntervalContainer2');
        const divSec = document.getElementById('divisionSection');
        
        if (divSec) divSec.style.display = 'block';
        
        if (data.status === 'success' && data.data) {
            const d = data.data;
            const offset = Number.isFinite(Number(d.offset)) ? Number(d.offset) : 0;
            state.septumOffset = offset;

            // Convert a stored value back to a sequence frame for display.
            //
            // For LINKED (multi-film) cells, the backend's
            // SeptumService.get_septum_alignment() already reassembles each
            // endpoint into a true sequence-wide frame number before sending
            // it to us - it has access to each film's own locally-stored
            // offset and each film's own position in the sequence, so it can
            // resolve the "which film does this endpoint belong to"
            // ambiguity server-side instead of us guessing client-side. It
            // signals this by returning `offset: 0`. Do NOT re-apply local
            // per-film math here for linked cells - `alignedValue` already
            // IS the sequence frame.
            //
            // (An earlier version of this function tried to guess the owning
            // film client-side via findFilmIdxForLocalFrame(). That broke
            // because different films can carry different stored offsets, so
            // decoding with only one film's offset made a cell's endpoints
            // appear to change depending on which film happened to be active
            // when the reload fired. Resolving it once, server-side, with
            // each film's own offset is the correct fix - see
            // septum_service.py's get_septum_alignment() for the full
            // explanation.)
            //
            // Single-film cells never had this ambiguity (sequence frame ===
            // local frame there), so they still go through the plain
            // local+offset conversion against the one film that exists.
            const toSequenceFrame = (alignedValue) => {
                if (alignedValue === null || alignedValue === undefined || alignedValue === '') return '';
                if (isLinkedSequenceCell()) {
                    const sequenceFrame = Number(alignedValue);
                    return sequenceFrame >= 0 && sequenceFrame < state.numFrames ? sequenceFrame : '';
                }
                const localFrame = Number(alignedValue) - offset;
                const sequenceFrame = localFrameToSequenceFrame(localFrame, current.filmIdx);
                const bounds = getFilmSequenceBounds(current.filmIdx);
                return sequenceFrame !== null && sequenceFrame >= bounds.start && sequenceFrame < bounds.end
                    ? sequenceFrame
                    : '';
            };
            if (hasBox1) hasBox1.checked = !!d.has_septum;
            if (divContainer1) divContainer1.style.display = d.has_septum ? 'flex' : 'none';
            if (startIn1) startIn1.value = toSequenceFrame(d.start_aligned);
            if (endIn1) endIn1.value = toSequenceFrame(d.end_aligned);
            if (whiteBox1) whiteBox1.checked = !!d.is_white_septum;

            if (hasBox2) hasBox2.checked = !!d.has_septum_2;
            if (divContainer2) divContainer2.style.display = d.has_septum_2 ? 'flex' : 'none';
            if (startIn2) startIn2.value = toSequenceFrame(d.start_aligned_2);
            if (endIn2) endIn2.value = toSequenceFrame(d.end_aligned_2);
            if (whiteBox2) whiteBox2.checked = !!d.is_white_septum_2;
        } else {
            state.septumOffset = 0;
            if (hasBox1) hasBox1.checked = false;
            if (divContainer1) divContainer1.style.display = 'none';
            if (startIn1) startIn1.value = '';
            if (endIn1) endIn1.value = '';
            if (whiteBox1) whiteBox1.checked = false;
            
            if (hasBox2) hasBox2.checked = false;
            if (divContainer2) divContainer2.style.display = 'none';
            if (startIn2) startIn2.value = '';
            if (endIn2) endIn2.value = '';
            if (whiteBox2) whiteBox2.checked = false;
        }
        renderGallery();
        loadCachedSeptumSuggestion();
    } catch (e) {
        console.error("Error loading septum labels:", e);
    }
}

async function saveSeptumLabels() {
    if (state.selectedCell === null) return;
    const current = getActiveFilmAndLocalCell();
    if (!current.film || current.cellId === null || current.cellId === -1) return;
    
    const has1 = document.getElementById('hasSeptumCheckbox').checked;
    const start1 = document.getElementById('septumStartInput').value;
    const end1 = document.getElementById('septumEndInput').value;
    const white1 = document.getElementById('whiteSeptumCheckbox').checked;
    
    const has2 = document.getElementById('hasSeptumCheckbox2').checked;
    const start2 = document.getElementById('septumStartInput2').value;
    const end2 = document.getElementById('septumEndInput2').value;
    const white2 = document.getElementById('whiteSeptumCheckbox2').checked;

    // IMPORTANT: a septum's start and end endpoint do not have to be in the
    // same film - cell division can span a film boundary within a linked
    // sequence. So we must NOT force every endpoint through the film the
    // playhead currently happens to be on (`current.filmIdx`); that film is
    // only meaningful for whichever endpoint was *just* clicked, not for
    // whichever endpoint you set earlier in a different film.
    //
    // For linked/multi-film cells we instead send the plain sequence-wide
    // frame number as `start_aligned`/`end_aligned`. The backend
    // (SeptumService.save_septum_label) already knows how to split one
    // sequence-wide interval across every film it crosses - see
    // services/septum_service.py, which compares these values against each
    // film's cumulative sequence bounds and re-derives each film's own local
    // frame itself. `start_frame`/`end_frame` (local) are left null here
    // because the backend ignores them on this multi-film path anyway.
    //
    // For single-film cells (isLocalEdit, or a sequence with exactly one
    // film) sequence frame === local frame for the one film that exists, so
    // we keep the original local/offset conversion - there's no ambiguity to
    // worry about there.
    const parseSequenceEndpoint = (rawValue, label) => {
        if (rawValue === '') return { local: null, aligned: null };
        const sequenceFrame = parseInt(rawValue);

        if (isLinkedSequenceCell()) {
            if (isNaN(sequenceFrame) || sequenceFrame < 0 || sequenceFrame >= state.numFrames) {
                throw new Error(
                    `${label} frame ${sequenceFrame} is outside the sequence range ` +
                    `0–${Math.max(0, state.numFrames - 1)}.`
                );
            }
            // No single film owns this value - it's resolved per-film on the
            // backend once we know which film(s) the interval overlaps.
            return { local: null, aligned: sequenceFrame };
        }

        const localFrame = sequenceFrameToLocalFrame(sequenceFrame, current.filmIdx);
        if (localFrame === null) {
            const bounds = getFilmSequenceBounds(current.filmIdx);
            throw new Error(
                `${label} frame ${sequenceFrame} is outside the active film range ` +
                `${bounds.start}–${Math.max(bounds.start, bounds.end - 1)}.`
            );
        }
        const offset = Number.isFinite(Number(state.septumOffset)) ? Number(state.septumOffset) : 0;
        return { local: localFrame, aligned: localFrame + offset };
    };

    let endpoint1Start, endpoint1End, endpoint2Start, endpoint2End;
    try {
        endpoint1Start = parseSequenceEndpoint(start1, 'Start 1');
        endpoint1End = parseSequenceEndpoint(end1, 'End 1');
        endpoint2Start = parseSequenceEndpoint(start2, 'Start 2');
        endpoint2End = parseSequenceEndpoint(end2, 'End 2');
    } catch (e) {
        alert(e.message || e);
        return;
    }
    
    // `film`/`cell_id` here just need to be *any* film+local-cell-id that
    // belongs to this cell's linked sequence - the backend uses them purely
    // to look up the full film list via sequence_linkage.json
    // (_find_sequence_linkage), then fans the save out across every film the
    // start_aligned/end_aligned interval actually touches. They do NOT need
    // to be "the film the endpoints are in" (there may not be a single such
    // film). Same for `film_index`: it's informational only and unused by
    // the multi-film split path in septum_service.py.
    const body = {
        experiment: state.selectedExp,
        film: current.film,
        cell_id: String(current.cellId),
        has_septum: has1,
        start_frame: endpoint1Start.local,
        end_frame: endpoint1End.local,
        start_aligned: endpoint1Start.aligned,
        end_aligned: endpoint1End.aligned,
        is_white_septum: white1,
        has_septum_2: has2,
        start_frame_2: endpoint2Start.local,
        end_frame_2: endpoint2End.local,
        start_aligned_2: endpoint2Start.aligned,
        end_aligned_2: endpoint2End.aligned,
        is_white_septum_2: white2,
        offset: Number.isFinite(Number(state.septumOffset)) ? Number(state.septumOffset) : 0,
        sequence: state.selectedSequence || null,
        global_cell_id: state.isLocalEdit ? null : String(state.selectedCell),
        film_index: current.filmIdx
    };
    
    try {
        const res = await fetch('/api/save_septum_label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success') {
            throw new Error(data.message || 'Failed to save septum annotation');
        }
        if (data.training_export && data.training_export.status === 'error') {
            alert(`Annotation saved, but training export failed: ${data.training_export.message}`);
        }
        renderGallery();
    } catch (e) {
        console.error("Error saving septum labels:", e);
    }
}

async function runSeptumAi() {
    if (state.selectedCell === null) return;
    const current = getActiveFilmAndLocalCell();
    if (!current.film || current.cellId === null || current.cellId === -1) return;
    
    const btn = document.getElementById('predictSeptumBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerText = '🤖 Predicting...';
    }
    
    try {
        const body = {
            experiment: state.selectedExp,
            film: current.film,
            cell_id: current.cellId
        };
        if (!state.isLocalEdit) {
            body.sequence = state.selectedSequence;
            body.global_cell_id = String(state.selectedCell);
        }
        const res = await fetch('/api/predict_septum', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.status === 'success') {
            if (data.probs && data.probs.length > 0) {
                renderSeptumSparkline(data);
            }
        } else {
            alert(data.message || "Septum AI prediction failed");
        }
    } catch (e) {
        alert(e);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerText = '🤖 Run AI';
        }
    }
}

function renderSeptumSparkline(data, isCached = false) {
    const chart = document.getElementById('septumAiChart');
    const spark = document.getElementById('septumAiSparkline');
    const txt = document.getElementById('septumAiPeakText');
    const warning = document.getElementById('septumAiWarning');
    if (!chart || !spark) return;

    const probs = data.probs || [];
    const sequenceIndices = data.sequence_indices || probs.map((_, idx) => idx);
    const startFrame = data.suggested_start ? data.suggested_start.sequence_frame : null;
    const endFrame = data.suggested_end ? data.suggested_end.sequence_frame : null;
    chart.style.display = 'flex';
    if (txt) {
        const startLabel = data.suggested_start
            ? `start g${startFrame} (${data.suggested_start.film} t${data.suggested_start.local_frame})`
            : 'start unavailable';
        const endLabel = data.suggested_end
            ? `end g${endFrame} (${data.suggested_end.film} t${data.suggested_end.local_frame})`
            : 'end unavailable';
        const prefix = isCached ? '(cached batch) ' : '';
        txt.innerText = `${prefix}${startLabel} · ${endLabel}`;
    }
    if (warning) {
        // Cached suggestions come from an offline batch run and have no
        // per-frame probability array to draw (see get_septum_ai_cache /
        // predict_m156_septum.py-style batch scripts) - only the summary
        // fields (peak/start/end/confidences) were kept to keep the cache
        // file small. Say so explicitly so nobody mistakes the empty
        // sparkline below for "no signal" rather than "no chart data saved".
        const cachedNote = isCached
            ? ' (cached from an offline batch run - click Run AI for the full live chart)'
            : '';
        warning.innerText = (data.warning || 'AI suggestion only—verify before saving.') + cachedNote;
        warning.style.display = 'block';
    }

    spark.innerHTML = probs.map((p, idx) => {
        const hPct = Math.max(5, p * 100);
        const sequenceFrame = sequenceIndices[idx];
        const isStart = sequenceFrame === startFrame;
        const isEnd = sequenceFrame === endFrame;
        const isPeak = sequenceFrame === data.peak_t;
        const color = isStart ? '#22c55e' : (isEnd ? '#ef4444' : (isPeak ? '#a855f7' : (p > 0.4 ? '#3b82f6' : '#334155')));
        return `<div style="flex: 1; height: ${hPct}%; background-color: ${color}; min-width: 1px;" title="sequence g${sequenceFrame}: state ${(p * 100).toFixed(1)}%"></div>`;
    }).join('');
}

async function loadCachedSeptumSuggestion() {
    // Cheap, automatic counterpart to the manual "Run AI" button: on every
    // cell load, ask the backend for a pre-computed (offline batch)
    // suggestion for this cell, if one exists. This never triggers model
    // inference (see /api/get_septum_ai_cache in septum_bp.py) - it's a
    // pure file lookup - so it's safe to fire on every cell switch instead
    // of only on an explicit click. If the user then clicks "Run AI", that
    // live result (with its full per-frame chart) simply overwrites this.
    const chart = document.getElementById('septumAiChart');
    if (state.isLocalEdit || state.selectedCell === null || !state.selectedSequence) {
        if (chart) chart.style.display = 'none';
        return;
    }
    try {
        const url = `/api/get_septum_ai_cache?experiment=${encodeURIComponent(state.selectedExp)}` +
            `&sequence=${encodeURIComponent(state.selectedSequence)}` +
            `&global_cell_id=${encodeURIComponent(state.selectedCell)}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.status === 'success' && data.cached && data.data) {
            renderSeptumSparkline(data.data, true);
        } else if (chart) {
            chart.style.display = 'none';
        }
    } catch (e) {
        console.error("Error loading cached septum AI suggestion:", e);
        if (chart) chart.style.display = 'none';
    }
}
