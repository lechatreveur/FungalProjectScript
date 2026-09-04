(function() {
    "use strict";

    var dimMode = "3D";
    var is3D = true;
    var lastKey = "";
    var plotDiv = document.getElementById('plot-div');
    var selectedGID = null;
    var selectedGCID = null;
    
    // Color thresholds for classification
    var defaultThresholds = {
        pol1: 4.04,
        pol2: 2.0,
        mono: 5.0,
        bi: 6.5
    };

    // 5-category discrete colorscale: 0=Non-polarized, 1=Monopolar, 2=Monopolar Osc, 3=Bipolar, 4=Bipolar Osc
    var dynamicColorscale = [
        [0.0,  '#94a3b8'], [0.2,  '#94a3b8'],
        [0.2,  '#f59e0b'], [0.4,  '#f59e0b'],
        [0.4,  '#ef4444'], [0.6,  '#ef4444'],
        [0.6,  '#10b981'], [0.8,  '#10b981'],
        [0.8,  '#3b82f6'], [1.0,  '#3b82f6']
    ];

    var customColorscale = [
        [0.0, 'rgb(0,0,131)'],
        [0.125, 'rgb(0,60,170)'],
        [0.375, 'rgb(5,255,255)'],
        [0.625, 'rgb(255,255,0)'],
        [0.875, 'rgb(250,0,0)'],
        [1.0, 'rgb(128,0,0)']
    ];

    // Safe text helper to prevent innerHTML injection
    function el(tag, attrs, children) {
        var element = document.createElement(tag);
        if (attrs) {
            for (var key in attrs) {
                if (key === 'className') {
                    element.className = attrs[key];
                } else if (key === 'style' && typeof attrs[key] === 'object') {
                    Object.assign(element.style, attrs[key]);
                } else {
                    element.setAttribute(key, attrs[key]);
                }
            }
        }
        if (children) {
            if (!Array.isArray(children)) children = [children];
            children.forEach(function(child) {
                if (typeof child === 'string' || typeof child === 'number') {
                    element.appendChild(document.createTextNode(child));
                } else if (child) {
                    element.appendChild(child);
                }
            });
        }
        return element;
    }

    // Debounce wrapper
    function debounce(fn, delay) {
        var timer;
        return function() {
            var context = this, args = arguments;
            clearTimeout(timer);
            timer = setTimeout(function() {
                fn.apply(context, args);
            }, delay);
        };
    }

    // Dynamic mode classification: returns integer 0-4
    function getCategory(pol1_mid, pol2_mid, periodicity) {
        var t = defaultThresholds;
        if (pol1_mid < t.pol1) {
            return 0; // Non-polarized
        } else if (pol2_mid < t.pol2) {
            return periodicity > t.mono ? 2 : 1; // Monopolar Osc vs Monopolar
        } else {
            return periodicity > t.bi ? 4 : 3; // Bipolar Osc vs Bipolar
        }
    }

    function categoryLabel(cat) {
        var labels = ["Non-polarized", "Monopolar", "Monopolar Osc", "Bipolar", "Bipolar Osc"];
        var badgeColors = ["#94a3b8", "#f59e0b", "#ef4444", "#10b981", "#3b82f6"];
        return { txt: labels[cat], col: badgeColors[cat] };
    }

    // Helper: decode binary-typed arrays from plotly if needed
    function getCoordsArray(val) {
        if (!val) return null;
        if (val.bdata) {
            var binaryString = atob(val.bdata);
            var bytes = new Uint8Array(binaryString.length);
            for (var i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
            return new Float32Array(bytes.buffer);
        }
        return val;
    }

    function syncMobileQC() {
        var expKeys = {};
        for (var gid in trajData) {
            var cell = trajData[gid];
            if (cell) {
                var exp = cell.experiment || 'M156';
                var seq = cell.sequence || cell.field || cell.gfp_film;
                if (exp && seq) {
                    if (!expKeys[exp]) expKeys[exp] = {};
                    expKeys[exp][seq] = true;
                }
            }
        }
        var promises = [];
        for (var expKey in expKeys) {
            for (var sKey in expKeys[expKey]) {
                var url = 'http://127.0.0.1:5001/api/get_qc?experiment=' + encodeURIComponent(expKey) + '&sequence=' + encodeURIComponent(sKey);
                promises.push(
                    fetch(url)
                    .then(function(res) { return res.ok ? res.json() : null; })
                    .then(function(data) {
                        return (data && data.status === 'success' && data.qc) ? data.qc : null;
                    })
                    .catch(function() { return null; })
                );
            }
        }
        Promise.all(promises).then(function(results) {
            var mergedQC = {};
            results.forEach(function(qcObj) {
                if (qcObj) Object.assign(mergedQC, qcObj);
            });
            var countUpdated = 0;
            for (var gid in trajData) {
                var cell = trajData[gid];
                if (!cell) continue;

                var matchedStatus = null;
                if (mergedQC[gid]) matchedStatus = mergedQC[gid];
                if (!matchedStatus && cell.global_cell_id && mergedQC[cell.global_cell_id]) matchedStatus = mergedQC[cell.global_cell_id];

                var numMatch = String(cell.global_cell_id || cell.table_row_id || cell.local_gfp_id || gid).match(/\d+$/);
                var numId = numMatch ? numMatch[0] : null;
                if (!matchedStatus && numId) {
                    if (mergedQC[numId]) matchedStatus = mergedQC[numId];
                    if (!matchedStatus && cell.sequence && mergedQC[cell.sequence + '_cell_' + numId]) matchedStatus = mergedQC[cell.sequence + '_cell_' + numId];
                    if (!matchedStatus) {
                        for (var k in mergedQC) {
                            if (k.endsWith('_cell_' + numId) || k.endsWith('_' + numId) || k === numId) {
                                matchedStatus = mergedQC[k];
                                break;
                            }
                        }
                    }
                }

                if (matchedStatus) {
                    if (cell.qc_marked !== matchedStatus) {
                        cell.qc_marked = matchedStatus;
                        cell.qc_status = matchedStatus;
                        cell.status = matchedStatus;
                        cell.qc = matchedStatus;
                        countUpdated++;
                    }
                }
            }
            var statusEl = document.getElementById('qc-sync-status');
            if (statusEl) {
                statusEl.textContent = '• Synced with Mobile QC';
                statusEl.style.color = '#10b981';
            }
            var hideCb = document.getElementById('hide-mistracked-checkbox');
            var showGoodCb = document.getElementById('show-only-good-checkbox');
            if (countUpdated > 0 || (hideCb && hideCb.checked) || (showGoodCb && showGoodCb.checked)) {
                renderPlot();
            }
        }).catch(function() {
            var statusEl = document.getElementById('qc-sync-status');
            if (statusEl) {
                statusEl.textContent = '(Offline QC mode)';
                statusEl.style.color = '#64748b';
            }
        });
    }

    function renderPlot() {
        var colorKey = document.getElementById('color-select').value;
        var hideMistracked = document.getElementById('hide-mistracked-checkbox') && document.getElementById('hide-mistracked-checkbox').checked;
        var showOnlyGood = document.getElementById('show-only-good-checkbox') && document.getElementById('show-only-good-checkbox').checked;

        var activePlotData;
        if (dimMode === "3D") {
            activePlotData = plotData3D;
        } else if (dimMode === "3D_latent") {
            activePlotData = plotDataLatent;
        } else {
            activePlotData = plotData2D;
        }

        // Capture current per-trace visibility before rebuilding (so legend toggles survive re-renders)
        var savedVisibility = {};
        if (plotDiv && plotDiv.data) {
            for (var vi = 0; vi < plotDiv.data.length; vi++) {
                var vt = plotDiv.data[vi];
                if (vt.name) savedVisibility[vt.name] = vt.visible;
            }
        }
        
        var showThresholds = (colorKey === "Dynamic modes");
        document.getElementById('thresholds-container').style.display = showThresholds ? "flex" : "none";
        
        // Show pre-div checkbox only if we have linkages or multi-part cell lineages
        var hasLinkages = false;
        var gcidCountMap = {};
        var gcidMistrackedMap = {};
        var maxConnectedParts = 1;

        for (var gidCheck in trajData) {
            var cCheck = trajData[gidCheck];
            if (!cCheck || !cCheck.gcid) continue;
            var gK = cCheck.gcid;
            
            var rSt = cCheck.qc_marked || cCheck.qc_status || cCheck.status || cCheck.qc;
            var sSt = rSt ? String(rSt).toLowerCase().trim() : 'pending';
            
            if (sSt === 'mistracked' || sSt === 'bad' || sSt === 'false_positive') {
                gcidMistrackedMap[gK] = true;
            } else {
                gcidCountMap[gK] = (gcidCountMap[gK] || 0) + 1;
                if (gcidCountMap[gK] >= 2) hasLinkages = true;
                if (gcidCountMap[gK] > maxConnectedParts) maxConnectedParts = gcidCountMap[gK];
            }
        }
        document.getElementById('link-pairs-container').style.display = hasLinkages ? "flex" : "none";

        // Level 2 QC: Target connected parts for a complete good cell (6 for M156 multi-part, 2 for pair linkages)
        var targetConnectedParts = hasLinkages ? Math.max(6, maxConnectedParts) : 1;

        // Read thresholds from inputs
        if (showThresholds) {
            defaultThresholds.pol1 = parseFloat(document.getElementById('thresh-pol1-mid').value) || 4.04;
            defaultThresholds.pol2 = parseFloat(document.getElementById('thresh-pol2-mid').value) || 2.0;
            defaultThresholds.mono = parseFloat(document.getElementById('thresh-mono-osc').value) || 5.0;
            defaultThresholds.bi = parseFloat(document.getElementById('thresh-bi-osc').value) || 6.5;
        }

        // Deep copy container structure but NOT coordinates to avoid memory waste
        var currentPlot = {
            data: [],
            layout: JSON.parse(JSON.stringify(activePlotData.layout))
        };

        // Determine min/max range overrides
        var cmin_override = parseFloat(document.getElementById('color-range-min').value);
        var cmax_override = parseFloat(document.getElementById('color-range-max').value);

        // Process color values
        var colorValues = colorArrays[colorKey] || [];
        var validVals = colorValues.filter(function(v) { return v !== null && !isNaN(v); });
        
        var lo = Infinity, hi = -Infinity;
        for (var idx = 0; idx < validVals.length; idx++) {
            if (validVals[idx] < lo) lo = validVals[idx];
            if (validVals[idx] > hi) hi = validVals[idx];
        }
        if (lo === Infinity) { lo = 0; hi = 1; }
        
        var chosen_cmin = isNaN(cmin_override) ? lo : cmin_override;
        var chosen_cmax = isNaN(cmax_override) ? hi : cmax_override;

        var pointsMap = {}; // observation_id -> {x, y, z, val}

        var lineTraces = [];
        var markerTraces = [];

        var linkPairsActive = document.getElementById('link-pairs-checkbox').checked;

        for (var tIdx = 0; tIdx < activePlotData.data.length; tIdx++) {
            var srcTrace = activePlotData.data[tIdx];
            if (srcTrace.type !== "scatter3d" && srcTrace.type !== "scatter") {
                continue;
            }
            if (!srcTrace.customdata) {
                markerTraces.push(srcTrace);
                continue;
            }

            var gids = srcTrace.customdata.map(function(d) { return d[0]; });
            var mappedColors = [];
            var pointSizes = [];
            var pointLineWidths = [];
            var pointLineColors = [];
            var filteredX = [];
            var filteredY = [];
            var filteredZ = [];
            var filteredCustomdata = [];

            var origX = getCoordsArray(srcTrace.x);
            var origY = getCoordsArray(srcTrace.y);
            var origZ = is3D ? getCoordsArray(srcTrace.z) : null;
            
            for (var pIdx = 0; pIdx < gids.length; pIdx++) {
                var gid = gids[pIdx];
                var cell = trajData[gid];

                var rawStatus = cell ? (cell.qc_marked || cell.qc_status || cell.status || cell.qc) : null;
                var statusVal = rawStatus ? String(rawStatus).toLowerCase().trim() : 'pending';

                // Level 1 QC: Local part mistracked status
                var isMistracked = (statusVal === 'mistracked' || statusVal === 'bad' || statusVal === 'false_positive');

                // Level 2 QC: "Good" means status is explicitly good, cell has 6 connected local parts, and NO parts are mistracked!
                var gcidCount = (cell && cell.gcid) ? (gcidCountMap[cell.gcid] || 0) : 0;
                var gcidHasMistracked = (cell && cell.gcid) ? (gcidMistrackedMap[cell.gcid] || false) : false;

                var isGood = (statusVal === 'good' || statusVal === 'ok' || statusVal === 'usable' || statusVal === 'corrected')
                             && (gcidCount >= targetConnectedParts)
                             && (!gcidHasMistracked);

                if (hideMistracked && isMistracked) {
                    continue; // Skip mis-tracked point when "Hide mis-tracked" is active
                }

                if (showOnlyGood && !isGood) {
                    continue; // Skip non-good point when "Show only good" toggle is active
                }

                var val = null;

                if (cell) {
                    if (colorKey === "Dynamic modes") {
                        val = getCategory(cell.raw_feats.pol1_mid, cell.raw_feats.pol2_mid, cell.raw_feats.periodicity);
                    } else {
                        val = colorValues[cell.idx];
                    }
                }
                mappedColors.push(val === null ? NaN : val);

                var isSelectedPoint = (selectedGCID !== null && cell && (cell.gcid === selectedGCID || cell.global_cell_id === selectedGCID));
                var isSelectedGid = (selectedGID !== null && gid === selectedGID);
                
                if (linkPairsActive && selectedGCID && isSelectedPoint) {
                    if (isSelectedGid) {
                        pointSizes.push(Math.max((srcTrace.marker.size || 5) * 2.0, 12.0));
                        pointLineWidths.push(3.0);
                        pointLineColors.push('#ffffff');
                    } else {
                        pointSizes.push(Math.max((srcTrace.marker.size || 5) * 1.4, 8.5));
                        pointLineWidths.push(1.8);
                        pointLineColors.push('#0f172a');
                    }
                } else if (linkPairsActive) {
                    pointSizes.push(srcTrace.marker.size || 5);
                    pointLineWidths.push(0.5);
                    pointLineColors.push('#64748b');
                } else {
                    pointSizes.push(srcTrace.marker.size || 5);
                    pointLineWidths.push(0.5);
                    pointLineColors.push('#475569');
                }

                filteredX.push(origX[pIdx]);
                filteredY.push(origY[pIdx]);
                if (is3D && origZ) filteredZ.push(origZ[pIdx]);
                filteredCustomdata.push(srcTrace.customdata[pIdx]);

                // Save point coords for pair linking
                pointsMap[gid] = {
                    x: origX[pIdx],
                    y: origY[pIdx],
                    z: is3D && origZ ? origZ[pIdx] : 0,
                    val: val
                };
            }

            var cscale = (colorKey === "Dynamic modes") ? dynamicColorscale : customColorscale;
            var cmin_val = (colorKey === "Dynamic modes") ? -0.5 : chosen_cmin;
            var cmax_val = (colorKey === "Dynamic modes") ? 4.5 : chosen_cmax;

            var newTrace = Object.assign({}, srcTrace, {
                x: filteredX,
                y: filteredY,
                customdata: filteredCustomdata,
                marker: Object.assign({}, srcTrace.marker, {
                    color: mappedColors,
                    cmin: cmin_val,
                    cmax: cmax_val,
                    colorscale: cscale,
                    showscale: (tIdx === 0),
                    size: pointSizes,
                    line: {
                        color: pointLineColors,
                        width: pointLineWidths
                    },
                    colorbar: (tIdx === 0) ? {
                        x: -0.1,
                        xanchor: 'right',
                        titleside: 'top',
                        ticks: 'outside',
                        thickness: 15,
                        len: 0.6,
                        tickvals: (colorKey === "Dynamic modes") ? [0, 1, 2, 3, 4] : undefined,
                        ticktext: (colorKey === "Dynamic modes") ? ["Non-polarized", "Monopolar", "Monopolar Osc", "Bipolar", "Bipolar Osc"] : undefined,
                        title: {
                            text: (colorKey === "Dynamic modes") ? "Dynamic Mode" : colorKey,
                            font: { color: '#1e293b', size: 10 }
                        },
                        tickfont: { color: '#1e293b', size: 9 }
                    } : undefined
                })
            });
            if (is3D && origZ) newTrace.z = filteredZ;

            // Reapply saved visibility so legend toggles survive re-renders
            if (srcTrace.name && savedVisibility[srcTrace.name] !== undefined) {
                newTrace.visible = savedVisibility[srcTrace.name];
            }
            markerTraces.push(newTrace);
        }

        // Draw pair links if enabled
        var linkPairs = document.getElementById('link-pairs-checkbox').checked;
        if (linkPairs && hasLinkages) {
            // Snapshot current trace visibility from the live plot
            var traceVisibility = {};
            if (plotDiv && plotDiv.data) {
                for (var vi = 0; vi < plotDiv.data.length; vi++) {
                    var vt = plotDiv.data[vi];
                    if (vt.name) traceVisibility[vt.name] = vt.visible;
                }
            }

            // Build coord map only from visible traces
            var coordsMap = {};
            for (var tIdx2 = 0; tIdx2 < markerTraces.length; tIdx2++) {
                var t2 = markerTraces[tIdx2];
                if (!t2.customdata || !t2.x || !t2.y) continue;
                var vis = traceVisibility[t2.name];
                if (vis === 'legendonly' || vis === false) continue;

                var xs2 = getCoordsArray(t2.x);
                var ys2 = getCoordsArray(t2.y);
                var zs2 = is3D ? getCoordsArray(t2.z) : null;
                for (var j2 = 0; j2 < t2.customdata.length; j2++) {
                    var gid2 = t2.customdata[j2][0];
                    if (is3D && zs2 && j2 < zs2.length) {
                        coordsMap[gid2] = { x: xs2[j2], y: ys2[j2], z: zs2[j2] };
                    } else if (!is3D) {
                        coordsMap[gid2] = { x: xs2[j2], y: ys2[j2] };
                    }
                }
            }

            // Group gids by global cell id (gcid)
            var gcidGroups = {};
            for (var gid3 in trajData) {
                var cell3 = trajData[gid3];
                if (cell3 && cell3.gcid && coordsMap[gid3]) {
                    if (!gcidGroups[cell3.gcid]) gcidGroups[cell3.gcid] = [];
                    gcidGroups[cell3.gcid].push(gid3);
                }
            }

            // Build N=10 gradient segments for Dull Lines and Active Selected Line
            var N = 10;
            var segX_dull = [], segY_dull = [], segZ_dull = [];
            var segX_active = [], segY_active = [], segZ_active = [];
            for (var si = 0; si < N; si++) {
                segX_dull.push([]); segY_dull.push([]); if (is3D) segZ_dull.push([]);
                segX_active.push([]); segY_active.push([]); if (is3D) segZ_active.push([]);
            }
            var hasDullData = false;
            var hasActiveData = false;

            for (var gcid in gcidGroups) {
                var gidsInGroup = gcidGroups[gcid];
                if (gidsInGroup.length >= 2) {
                    gidsInGroup.sort(function(a, b) {
                        var ca = trajData[a], cb = trajData[b];
                        var flA = ca && (ca.gfp_film || ca.sequence || a) ? (ca.gfp_film || ca.sequence || a).match(/FL(\d+)/i) : null;
                        var flB = cb && (cb.gfp_film || cb.sequence || b) ? (cb.gfp_film || cb.sequence || b).match(/FL(\d+)/i) : null;
                        if (flA && flB) {
                            var numA = parseInt(flA[1], 10);
                            var numB = parseInt(flB[1], 10);
                            if (numA !== numB) return numA - numB;
                        }
                        var ta = (ca && ca.f && ca.f[4] !== null) ? ca.f[4] : 0;
                        var tb = (cb && cb.f && cb.f[4] !== null) ? cb.f[4] : 0;
                        return ta - tb;
                    });

                    // Check if this cell group matches the currently clicked global cell
                    var isSelectedCellGroup = (selectedGCID !== null && gcid === selectedGCID);

                    var curX = isSelectedCellGroup ? segX_active : segX_dull;
                    var curY = isSelectedCellGroup ? segY_active : segY_dull;
                    var curZ = isSelectedCellGroup ? segZ_active : segZ_dull;

                    if (isSelectedCellGroup) hasActiveData = true;
                    else hasDullData = true;

                    for (var pi = 0; pi < gidsInGroup.length - 1; pi++) {
                        var gid1 = gidsInGroup[pi];
                        var gid2 = gidsInGroup[pi + 1];
                        var c1 = coordsMap[gid1];
                        var c2 = coordsMap[gid2];
                        if (c1 && c2) {
                            var start = c1;
                            var end   = c2;
                            for (var si2 = 0; si2 < N; si2++) {
                                var fStart = si2 / N, fEnd = (si2 + 1) / N;
                                curX[si2].push(start.x + fStart * (end.x - start.x), start.x + fEnd * (end.x - start.x), null);
                                curY[si2].push(start.y + fStart * (end.y - start.y), start.y + fEnd * (end.y - start.y), null);
                                if (is3D) curZ[si2].push(start.z + fStart * (end.z - start.z), start.z + fEnd * (end.z - start.z), null);
                            }
                        }
                    }
                }
            }

            // 1. Render dull lines (background)
            if (hasDullData) {
                for (var si3 = 0; si3 < N; si3++) {
                    var opacityD = 0.10 + 0.15 * (si3 / (N - 1));
                    var segTraceD = {
                        type: is3D ? 'scatter3d' : 'scatter',
                        mode: 'lines',
                        x: segX_dull[si3],
                        y: segY_dull[si3],
                        line: { color: 'rgba(160, 174, 192, ' + opacityD.toFixed(3) + ')', width: 1.2 },
                        hoverinfo: 'none',
                        showlegend: false
                    };
                    if (is3D) { segTraceD.z = segZ_dull[si3]; segTraceD.scene = 'scene'; }
                    lineTraces.push(segTraceD);
                }
            }

            // 2. Render ACTIVE selected cell lines on top (bold sky blue)
            if (hasActiveData) {
                for (var si4 = 0; si4 < N; si4++) {
                    var opacityA = 0.50 + 0.50 * (si4 / (N - 1));
                    var segTraceA = {
                        type: is3D ? 'scatter3d' : 'scatter',
                        mode: 'lines',
                        x: segX_active[si4],
                        y: segY_active[si4],
                        line: { color: 'rgba(2, 132, 199, ' + opacityA.toFixed(3) + ')', width: is3D ? 4.8 : 3.8 },
                        hoverinfo: 'none',
                        showlegend: false
                    };
                    if (is3D) { segTraceA.z = segZ_active[si4]; segTraceA.scene = 'scene'; }
                    lineTraces.push(segTraceA);
                }
            }
        }

        // IMPORTANT: Line traces FIRST (drawn under), Marker traces SECOND (drawn on top)
        currentPlot.data = lineTraces.concat(markerTraces);

        Plotly.react(plotDiv, currentPlot.data, currentPlot.layout, { responsive: true, displayModeBar: true });
        
        // Re-bind click handlers
        bindClick();
    }

    function bindClick() {
        plotDiv.removeAllListeners("plotly_click");
        plotDiv.removeAllListeners("plotly_legendclick");
        plotDiv.removeAllListeners("plotly_legenddoubleclick");
        plotDiv.on('plotly_legendclick', function() {
            setTimeout(renderPlot, 50);
        });
        plotDiv.on('plotly_click', function(eventData) {
            if (!eventData || !eventData.points || eventData.points.length === 0) return;
            var pt = eventData.points[0];
            if (pt.customdata && pt.customdata[0]) {
                var gid = pt.customdata[0];
                showDetail(gid);
            }
        });
    }

    function showDetail(gid) {
        var cell = trajData[gid];
        if (!cell) return;

        var prevGCID = selectedGCID;
        selectedGID = gid;
        selectedGCID = cell.gcid || cell.global_cell_id || null;

        if (prevGCID !== selectedGCID) {
            renderPlot();
        }

        var sidebar = document.getElementById('content');
        sidebar.innerHTML = "";

        // Header Title
        var header = el('div', { className: 'card' }, [
            el('h1', null, cell.gcid || gid),
            el('div', { className: 'subtitle' }, cell.experiment + " | " + cell.gfp_film)
        ]);
        sidebar.appendChild(header);

        // Metadata Card
        var mode = getCategory(cell.raw_feats.pol1_mid, cell.raw_feats.pol2_mid, cell.raw_feats.periodicity);
        var modeLabel = categoryLabel(mode);
        var metadataCard = el('div', { className: 'card' }, [
            el('h2', null, "Cell Metadata"),
            el('div', { className: 'stat' }, ["Global Cell ID", el('span', { className: 'val' }, cell.global_cell_id)]),
            el('div', { className: 'stat' }, ["Table Row ID", el('span', { className: 'val' }, cell.table_row_id !== undefined ? cell.table_row_id : cell.local_gfp_id)]),
            el('div', { className: 'stat' }, ["Estimated Cycle Stage", el('span', { className: 'val' }, cell.f[3] !== null ? cell.f[3] : "N/A")]),
            el('div', { className: 'stat' }, ["Time relative to div (min)", el('span', { className: 'val' }, cell.f[4] !== null ? cell.f[4] + " min" : "N/A")]),
            el('div', { className: 'stat' }, ["Time Alignment Method", el('span', { className: 'val' }, cell.f[5] !== null ? cell.f[5] : "N/A")]),
            el('div', { className: 'stat' }, ["Dynamic Mode", el('span', { className: 'val' }, [
                el('span', { className: 'cycle-badge', style: { background: modeLabel.col, marginLeft: '0' } }, modeLabel.txt)
            ])])
        ]);

        // Mark mistracked button & QC feedback
        var isCurrentlyMistracked = cell && (
            cell.qc_marked === 'mistracked' || 
            cell.status === 'mistracked' || 
            cell.qc === 'mistracked' || 
            cell.qc_status === 'mistracked'
        );

        var markBtn = el('button', { id: 'btn-mark-mistracked', className: 'btn-mistracked' }, isCurrentlyMistracked ? "✓ Marked as Mistracked (Click to unmark)" : "Mark mistracked");
        var feedbackMsg = el('div', { id: 'qc-feedback-msg', style: { marginTop: '6px', fontSize: '0.8rem', textAlign: 'center' } });

        if (isCurrentlyMistracked) {
            markBtn.style.background = "#059669";
        }

        markBtn.addEventListener('click', function() {
            var currentlyMistracked = cell && (
                cell.qc_marked === 'mistracked' || 
                cell.status === 'mistracked' || 
                cell.qc === 'mistracked' || 
                cell.qc_status === 'mistracked'
            );
            var targetStatus = currentlyMistracked ? 'pending' : 'mistracked';

            markBtn.disabled = true;
            markBtn.textContent = 'Saving...';
            feedbackMsg.style.color = '#64748b';
            feedbackMsg.textContent = '';

            var exp = cell.experiment || 'M156';
            var seq = cell.sequence || cell.field || cell.gfp_film;
            var targetId = cell.global_cell_id || gid;

            fetch('http://127.0.0.1:5001/api/save_qc', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    experiment: exp,
                    sequence: seq,
                    cell_id: targetId,
                    status: targetStatus
                })
            })
            .then(function(res) {
                if (!res.ok) {
                    throw new Error('HTTP ' + res.status + ': ' + res.statusText);
                }
                return res.json();
            })
            .then(function(data) {
                if (data.status === 'success') {
                    markBtn.disabled = false;
                    if (targetStatus === 'mistracked') {
                        cell.qc_marked = 'mistracked';
                        cell.status = 'mistracked';
                        cell.qc = 'mistracked';
                        cell.qc_status = 'mistracked';
                        markBtn.textContent = '✓ Marked as Mistracked (Click to unmark)';
                        markBtn.style.background = '#059669';
                        feedbackMsg.style.color = '#059669';
                        feedbackMsg.textContent = 'Saved to qc_' + seq + '.json!';
                    } else {
                        cell.qc_marked = 'pending';
                        cell.status = 'pending';
                        cell.qc = 'pending';
                        cell.qc_status = 'pending';
                        markBtn.textContent = 'Mark mistracked';
                        markBtn.style.background = '';
                        feedbackMsg.style.color = '#10b981';
                        feedbackMsg.textContent = 'Reverted to pending in qc_' + seq + '.json!';
                    }
                    renderPlot();
                } else {
                    throw new Error(data.message || 'Save failed');
                }
            })
            .catch(function(err) {
                console.warn('Mark mistracked request failed:', err);
                markBtn.disabled = false;
                var currState = cell && (
                    cell.qc_marked === 'mistracked' || 
                    cell.status === 'mistracked' || 
                    cell.qc === 'mistracked' || 
                    cell.qc_status === 'mistracked'
                );
                markBtn.textContent = currState ? '✓ Marked as Mistracked (Click to unmark)' : 'Mark mistracked';
                markBtn.style.background = currState ? '#059669' : '';
                feedbackMsg.style.color = '#dc2626';
                feedbackMsg.textContent = '⚠️ Failed: ' + err.message + ' (Check tracking_corrector on :5001)';
            });
        });

        metadataCard.appendChild(el('div', { style: { marginTop: '12px', paddingTop: '10px', borderTop: '1px dashed #cbd5e1' } }, [
            markBtn,
            feedbackMsg
        ]));

        sidebar.appendChild(metadataCard);

        // 2. Trajectory Chart Card
        var chartCard = el('div', { className: 'card sticky-card' }, [
            el('h2', null, "Intensity Trajectories"),
            el('div', { id: 'traj-div' }),
            el('div', { className: 'legend' }, "Red: Pol1 (computed) | Blue: Pol2")
        ]);
        sidebar.appendChild(chartCard);

        // 3. Numeric parameters card
        var statsCard = el('div', { className: 'card' }, [
            el('h2', null, "Extracted Features"),
            el('div', { className: 'stat' }, ["Pol1 Mid Intensity", el('span', { className: 'val' }, cell.raw_feats.pol1_mid)]),
            el('div', { className: 'stat' }, ["Pol2 Mid Intensity", el('span', { className: 'val' }, cell.raw_feats.pol2_mid)]),
            el('div', { className: 'stat' }, ["Periodicity Score", el('span', { className: 'val' }, cell.raw_feats.periodicity)])
        ]);
        sidebar.appendChild(statsCard);

        // 4. ACF fit & formula detail card (collapsible)
        var acorCard = el('div', { className: 'card' });
        var acorHeader = el('div', { 
            style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' } 
        }, [
            el('h2', { style: { margin: 0 } }, "Autocorrelation & Fit Details"),
            el('span', { id: 'acor-toggle-indicator', style: { color: '#64748b', fontSize: '0.9rem' } }, '►')
        ]);
        var acorBody = el('div', { id: 'acor-collapsible-body', style: { display: 'none', marginTop: '10px' } });
        
        var acorPlotDiv = el('div', { id: 'acor-div', style: { marginBottom: '10px' } });
        var formulaDiv = el('div', { style: { fontSize: '0.75rem', color: '#334155', background: '#f8fafc', padding: '8px', borderRadius: '4px', border: '1px solid #e2e8f0' } });
        
        acorBody.appendChild(acorPlotDiv);
        acorBody.appendChild(formulaDiv);
        
        acorCard.appendChild(acorHeader);
        acorCard.appendChild(acorBody);

        acorHeader.addEventListener('click', function() {
            var body = document.getElementById('acor-collapsible-body');
            if (body.style.display === 'none') {
                body.style.display = 'block';
                document.getElementById('acor-toggle-indicator').textContent = '▼';
            } else {
                body.style.display = 'none';
                document.getElementById('acor-toggle-indicator').textContent = '►';
            }
        });
        sidebar.appendChild(acorCard);

        // Populate fit formula details
        var fp = cell.fit_params || {};
        var env1 = (fp.p1_A1 || 0).toFixed(3) + '·e<sup>-t/' + (fp.p1_tau1 || 1).toFixed(1) + '</sup> + ' + 
                   Math.min(Math.max((fp.p1_acf0 || 0) - (fp.p1_A1 || 0) - (fp.p1_C || 0), 0.0), 1.0).toFixed(3) + '·e<sup>-t/' + (fp.p1_tau2 || 1).toFixed(1) + '</sup> + ' + 
                   (fp.p1_C || 0).toFixed(3);
        var fit1 = 'env · cos(2π·' + (fp.p1_f || 0).toFixed(4) + '·t + ' + (fp.p1_phi || 0).toFixed(2) + ')';
        
        var env2 = (fp.p2_A1 || 0).toFixed(3) + '·e<sup>-t/' + (fp.p2_tau1 || 1).toFixed(1) + '</sup> + ' + 
                   Math.min(Math.max((fp.p2_acf0 || 0) - (fp.p2_A1 || 0) - (fp.p2_C || 0), 0.0), 1.0).toFixed(3) + '·e<sup>-t/' + (fp.p2_tau2 || 1).toFixed(1) + '</sup> + ' + 
                   (fp.p2_C || 0).toFixed(3);
        var fit2 = 'env · cos(2π·' + (fp.p2_f || 0).toFixed(4) + '·t + ' + (fp.p2_phi || 0).toFixed(2) + ')';

        formulaDiv.innerHTML = 
            '<div style="margin-bottom: 8px;"><b>Pol1 ACF Fit:</b><br>' +
            'env = ' + env1 + '<br>' +
            'fit = ' + fit1 + '<br>' +
            '<span style="color:#ef4444;font-weight:bold;">y_f1_pen = ' + (fp.p1_penalized || 0).toFixed(4) + '</span> = ' +
            (fp.p1_y_f || 0).toFixed(4) + ' (raw peak) · ' + (fp.p1_freq_prior || 0).toFixed(3) + ' (freq prior)</div>' +
            
            '<div style="margin-bottom: 8px; border-top: 1px dashed #e2e8f0; padding-top: 6px;"><b>Pol2 ACF Fit:</b><br>' +
            'env = ' + env2 + '<br>' +
            'fit = ' + fit2 + '<br>' +
            '<span style="color:#2563eb;font-weight:bold;">y_f2_pen = ' + (fp.p2_penalized || 0).toFixed(4) + '</span> = ' +
            (fp.p2_y_f || 0).toFixed(4) + ' (raw peak) · ' + (fp.p2_freq_prior || 0).toFixed(3) + ' (freq prior)</div>' +
            
            '<div style="border-top: 1px solid #cbd5e1; padding-top: 6px; font-size:0.7rem; line-height: 1.4; color: #475569;">' +
            '<b>Periodicity Components:</b><br>' +
            'raw peaks sum (unpenalized): ' + (fp.raw_precision_sum || 0).toFixed(4) + '<br>' +
            'penalized peaks sum (raw): ' + Math.exp(fp.log_precision_sum || 0).toFixed(4) + '<br>' +
            'penalized peaks sum (log): ' + (fp.log_precision_sum || 0).toFixed(4) + '<br>' +
            'sse_zero (envelope distance): ' + (fp.sse_zero || 0).toFixed(4) + '<br>' +
            'log_zero (penalty): ' + (fp.log_zero || 0).toFixed(4) + '<br>' +
            '<span style="color:#ef4444;font-weight:bold;">Periodicity = log_prec - log_zero = ' + 
            ((fp.log_precision_sum || 0) - (fp.log_zero || 0)).toFixed(4) + '</span></div>';

        // 5. Image Strip Card
        if (cell.strip && cell.strip.length > 50) {
            var imgCard = el('div', { className: 'card' }, [
                el('h2', null, "Vertical Image Strip (Bleaching/Dynamics)"),
                el('div', { className: 'image-strip-container' }, [
                    el('img', { src: cell.strip, alt: "Image strip for " + gid })
                ])
            ]);
            sidebar.appendChild(imgCard);
        }

        // Render mini-plots
        var frames = Array.from({length: cell.p1.length}, function(_, i) { return i; });
        Plotly.newPlot('traj-div', [
            { x: frames, y: cell.p1, name: "Pol1", line: { color: "red", width: 2 } },
            { x: frames, y: cell.p2, name: "Pol2", line: { color: "blue", width: 2 } }
        ], {
            margin: { l: 30, r: 10, b: 30, t: 10 },
            showlegend: false,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            xaxis: { gridcolor: "#cbd5e1", tickcolor: "#475569", font: { color: "#475569" } },
            yaxis: { gridcolor: "#cbd5e1", tickcolor: "#475569", font: { color: "#475569" } }
        });

        // Compute fit curves
        function fit_model(lag, A1, tau1, tau2, f, phi, C, acf0) {
            var A2 = Math.min(Math.max((acf0 || 0) - (A1 || 0) - (C || 0), 0.0), 1.0);
            var envelope = A1 * Math.exp(-lag / tau1) + A2 * Math.exp(-lag / (tau2 || 1.0)) + C;
            return envelope * Math.cos(2 * Math.PI * f * lag + phi);
        }

        var acorFrames = Array.from({length: cell.acor1.length}, function(_, i) { return i; });
        var pol1Fit = [], pol2Fit = [];
        var fp = cell.fit_params || {};
        
        for (var lag = 0; lag < cell.acor1.length; lag++) {
            pol1Fit.push(fit_model(lag, fp.p1_A1, fp.p1_tau1, fp.p1_tau2, fp.p1_f, fp.p1_phi, fp.p1_C, fp.p1_acf0));
            pol2Fit.push(fit_model(lag, fp.p2_A1, fp.p2_tau1, fp.p2_tau2, fp.p2_f, fp.p2_phi, fp.p2_C, fp.p2_acf0));
        }

        Plotly.newPlot('acor-div', [
            { x: acorFrames, y: cell.acor1, name: "Pol1 Raw", line: { color: "rgba(239, 68, 68, 0.3)", width: 2 } },
            { x: acorFrames, y: pol1Fit, name: "Pol1 Fit", line: { color: "#ef4444", width: 2, dash: "dash" } },
            { x: acorFrames, y: cell.acor2, name: "Pol2 Raw", line: { color: "rgba(59, 130, 246, 0.3)", width: 2 } },
            { x: acorFrames, y: pol2Fit, name: "Pol2 Fit", line: { color: "#3b82f6", width: 2, dash: "dash" } }
        ], {
            margin: { l: 30, r: 10, b: 30, t: 10 },
            showlegend: false,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            xaxis: { gridcolor: "#cbd5e1", tickcolor: "#475569", font: { color: "#475569" } },
            yaxis: { gridcolor: "#cbd5e1", tickcolor: "#475569", font: { color: "#475569" } }
        });
    }

    // Attach control listeners
    document.getElementById('dim-select').addEventListener('change', function() {
        dimMode = this.value;
        is3D = (dimMode === '3D' || dimMode === '3D_latent');
        renderPlot();
    });

    document.getElementById('color-select').addEventListener('change', renderPlot);

    var debouncedRender = debounce(renderPlot, 250);
    document.getElementById('thresh-pol1-mid').addEventListener('input', debouncedRender);
    document.getElementById('thresh-pol2-mid').addEventListener('input', debouncedRender);
    document.getElementById('thresh-mono-osc').addEventListener('input', debouncedRender);
    document.getElementById('thresh-bi-osc').addEventListener('input', debouncedRender);
    document.getElementById('link-pairs-checkbox').addEventListener('change', renderPlot);
    document.getElementById('color-range-min').addEventListener('input', debouncedRender);
    document.getElementById('color-range-max').addEventListener('input', debouncedRender);

    var hideCheck = document.getElementById('hide-mistracked-checkbox');
    if (hideCheck) hideCheck.addEventListener('change', renderPlot);

    var showGoodCheck = document.getElementById('show-only-good-checkbox');
    if (showGoodCheck) showGoodCheck.addEventListener('change', renderPlot);

    // Keyboard Navigation: Left/Right keys jump through connected parts of the same global cell
    document.addEventListener('keydown', function(e) {
        var activeTag = document.activeElement ? document.activeElement.tagName : '';
        if (activeTag === 'INPUT' || activeTag === 'TEXTAREA' || activeTag === 'SELECT') {
            return;
        }

        var linkPairs = document.getElementById('link-pairs-checkbox').checked;
        if (!linkPairs) return;
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;

        if (!selectedGID || !trajData[selectedGID]) return;

        var currentCell = trajData[selectedGID];
        var gcid = currentCell.gcid || currentCell.global_cell_id;
        if (!gcid) return;

        // Collect all local parts belonging to this global cell
        var groupGIDs = [];
        for (var gid in trajData) {
            var cell = trajData[gid];
            if (cell && (cell.gcid === gcid || cell.global_cell_id === gcid)) {
                groupGIDs.push(gid);
            }
        }

        if (groupGIDs.length <= 1) return;

        // Sort groupGIDs by FL number (FL1..FL6) or relative time
        groupGIDs.sort(function(a, b) {
            var ca = trajData[a], cb = trajData[b];
            var flA = ca && (ca.gfp_film || ca.sequence || a) ? (ca.gfp_film || ca.sequence || a).match(/FL(\d+)/i) : null;
            var flB = cb && (cb.gfp_film || cb.sequence || b) ? (cb.gfp_film || cb.sequence || b).match(/FL(\d+)/i) : null;
            if (flA && flB) {
                var numA = parseInt(flA[1], 10);
                var numB = parseInt(flB[1], 10);
                if (numA !== numB) return numA - numB;
            }
            var ta = (ca && ca.f && ca.f[4] !== null) ? ca.f[4] : 0;
            var tb = (cb && cb.f && cb.f[4] !== null) ? cb.f[4] : 0;
            return ta - tb;
        });

        var currIdx = groupGIDs.indexOf(selectedGID);
        if (currIdx === -1) return;

        var nextIdx = currIdx;
        if (e.key === 'ArrowRight') {
            nextIdx = (currIdx + 1) % groupGIDs.length;
        } else if (e.key === 'ArrowLeft') {
            nextIdx = (currIdx - 1 + groupGIDs.length) % groupGIDs.length;
        }

        if (nextIdx !== currIdx) {
            e.preventDefault();
            var nextGID = groupGIDs[nextIdx];
            showDetail(nextGID);
        }
    });

    // Initial render & mobile QC sync
    window.addEventListener('load', function() {
        renderPlot();
        syncMobileQC();
    });

})();
