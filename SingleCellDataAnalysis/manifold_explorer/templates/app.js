(function() {
    "use strict";

    var dimMode = "3D";
    var is3D = true;
    var lastKey = "";
    var plotDiv = document.getElementById('plot-div');
    
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


    function renderPlot() {
        var colorKey = document.getElementById('color-select').value;
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
        
        // Show pre-div checkbox only if we have linkages
        var hasLinkages = activePlotData.data.some(function(trace) {
            return trace.customdata && trace.customdata.some(function(d) { return d[2] && d[2] !== ""; });
        });
        document.getElementById('link-pairs-container').style.display = hasLinkages ? "flex" : "none";

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

        var traceColors = [];
        var pointsMap = {}; // observation_id -> {x, y, z, val}

        for (var tIdx = 0; tIdx < activePlotData.data.length; tIdx++) {
            var srcTrace = activePlotData.data[tIdx];
            if (srcTrace.type !== "scatter3d" && srcTrace.type !== "scatter") {
                continue;
            }

            var gids = srcTrace.customdata.map(function(d) { return d[0]; });
            var mappedColors = [];
            
            for (var pIdx = 0; pIdx < gids.length; pIdx++) {
                var gid = gids[pIdx];
                var cell = trajData[gid];
                var val = null;

                if (cell) {
                    if (colorKey === "Dynamic modes") {
                        // Return integer category 0-4
                        val = getCategory(cell.raw_feats.pol1_mid, cell.raw_feats.pol2_mid, cell.raw_feats.periodicity);
                    } else {
                        val = colorValues[cell.idx];
                    }
                }
                mappedColors.push(val === null ? NaN : val);

                // Save point coords for pair linking
                pointsMap[gid] = {
                    x: srcTrace.x[pIdx],
                    y: srcTrace.y[pIdx],
                    z: is3D ? srcTrace.z[pIdx] : 0,
                    val: val
                };
            }

            var cscale = (colorKey === "Dynamic modes") ? dynamicColorscale : customColorscale;
            var cmin_val = (colorKey === "Dynamic modes") ? -0.5 : chosen_cmin;
            var cmax_val = (colorKey === "Dynamic modes") ? 4.5 : chosen_cmax;

            var newTrace = Object.assign({}, srcTrace, {
                marker: Object.assign({}, srcTrace.marker, {
                    color: mappedColors,
                    cmin: cmin_val,
                    cmax: cmax_val,
                    colorscale: cscale,
                    showscale: (tIdx === 0),
                    line: {
                        color: '#475569',
                        width: 0.5
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
            // Reapply saved visibility so legend toggles survive re-renders
            if (srcTrace.name && savedVisibility[srcTrace.name] !== undefined) {
                newTrace.visible = savedVisibility[srcTrace.name];
            }
            currentPlot.data.push(newTrace);
        }

        // Draw pair links if enabled — 10-segment fading grey lines matching original format
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
            for (var tIdx2 = 0; tIdx2 < currentPlot.data.length; tIdx2++) {
                var t2 = currentPlot.data[tIdx2];
                if (!t2.customdata || !t2.x || !t2.y) continue;
                // Respect legend visibility (legendonly or false = hidden)
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

            // Build N=10 gradient segments (fading from translucent to opaque grey)
            var N = 10;
            var segX = [], segY = [], segZ = [];
            for (var si = 0; si < N; si++) { segX.push([]); segY.push([]); if (is3D) segZ.push([]); }
            var hasData = false;

            for (var gcid in gcidGroups) {
                var gidsInGroup = gcidGroups[gcid];
                if (gidsInGroup.length === 2) {
                    var gid1 = gidsInGroup[0], gid2b = gidsInGroup[1];
                    var cell1 = trajData[gid1], cell2b = trajData[gid2b];
                    var t1 = cell1.f[4], t2b = cell2b.f[4];
                    if (t1 !== null && t2b !== null) {
                        var c1 = coordsMap[gid1], c2b = coordsMap[gid2b];
                        if (c1 && c2b) {
                            hasData = true;
                            // GFP1 is more negative time (earlier), GFP2 is less negative
                            var start = t1 < t2b ? c1 : c2b;
                            var end   = t1 < t2b ? c2b : c1;
                            for (var si2 = 0; si2 < N; si2++) {
                                var fStart = si2 / N, fEnd = (si2 + 1) / N;
                                segX[si2].push(start.x + fStart * (end.x - start.x), start.x + fEnd * (end.x - start.x), null);
                                segY[si2].push(start.y + fStart * (end.y - start.y), start.y + fEnd * (end.y - start.y), null);
                                if (is3D) segZ[si2].push(start.z + fStart * (end.z - start.z), start.z + fEnd * (end.z - start.z), null);
                            }
                        }
                    }
                }
            }

            if (hasData) {
                for (var si3 = 0; si3 < N; si3++) {
                    var opacity = 0.3 + 0.7 * (si3 / (N - 1));
                    var segTrace = {
                        type: is3D ? 'scatter3d' : 'scatter',
                        mode: 'lines',
                        x: segX[si3],
                        y: segY[si3],
                        line: { color: 'rgba(100,100,100,' + opacity.toFixed(3) + ')', width: 2.0 },
                        hoverinfo: 'none',
                        showlegend: false
                    };
                    if (is3D) { segTrace.z = segZ[si3]; segTrace.scene = 'scene'; }
                    currentPlot.data.push(segTrace);
                }
            }
        }

        Plotly.react(plotDiv, currentPlot.data, currentPlot.layout, { responsive: true, displayModeBar: true });
        
        // Re-bind click handlers
        bindClick();
    }

    function bindClick() {
        plotDiv.removeAllListeners("plotly_click");
        plotDiv.removeAllListeners("plotly_legendclick");
        plotDiv.removeAllListeners("plotly_legenddoubleclick");
        plotDiv.on('plotly_legendclick', function() {
            // Let Plotly perform its own toggle, then re-render to update link lines
            setTimeout(renderPlot, 50);
            // Do NOT return false — that would cancel the toggle
        });
        plotDiv.on('plotly_click', function(eventData) {
            if (!eventData || !eventData.points || eventData.points.length === 0) return;
            var pt = eventData.points[0];
            if (!pt.customdata) return;
            var gid = pt.customdata[0];
            updateSidebar(gid);
        });
    }

    function updateSidebar(gid) {
        if (lastKey === gid) return;
        lastKey = gid;

        var cell = trajData[gid];
        var sidebar = document.getElementById('sidebar');
        sidebar.innerHTML = ""; // Clear

        if (!cell) {
            sidebar.appendChild(el('div', { className: 'placeholder' }, "Cell data not found."));
            return;
        }

        var header = el('h2', null, [
            "Cell: " + gid,
            el('span', { 
                className: 'cycle-badge', 
                style: { background: cell.gfp_film.indexOf('TP1') !== -1 ? '#0284c7' : '#10b981' } 
            }, cell.gfp_film)
        ]);
        sidebar.appendChild(header);

        // Metadata Card
        var mode = getCategory(cell.raw_feats.pol1_mid, cell.raw_feats.pol2_mid, cell.raw_feats.periodicity);
        var modeLabel = categoryLabel(mode);
        var metadataCard = el('div', { className: 'card' }, [
            el('h2', null, "Cell Metadata"),
            el('div', { className: 'stat' }, ["Global Cell ID", el('span', { className: 'val' }, cell.global_cell_id)]),
            el('div', { className: 'stat' }, ["Local GFP ID", el('span', { className: 'val' }, cell.local_gfp_id)]),
            el('div', { className: 'stat' }, ["Estimated Cycle Stage", el('span', { className: 'val' }, cell.f[3] !== null ? cell.f[3] : "N/A")]),
            el('div', { className: 'stat' }, ["Time relative to div (min)", el('span', { className: 'val' }, cell.f[4] !== null ? cell.f[4] + " min" : "N/A")]),
            el('div', { className: 'stat' }, ["Time Alignment Method", el('span', { className: 'val' }, cell.f[5] !== null ? cell.f[5] : "N/A")]),
            el('div', { className: 'stat' }, ["Dynamic Mode", el('span', { className: 'val' }, [
                el('span', { className: 'cycle-badge', style: { background: modeLabel.col, marginLeft: '0' } }, modeLabel.txt)
            ])])
        ]);
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

        // 4. Autocorrelation Card (Collapsible)
        var acorHeader = el('h2', { style: 'cursor: pointer; display: flex; justify-content: space-between; align-items: center;' }, [
            "Autocorrelation Functions (Detrended)",
            el('span', { id: 'acor-toggle-indicator' }, "▼")
        ]);
        var formulaDiv = el('div', {
            id: 'acor-formula',
            style: { marginTop: '12px', paddingTop: '8px', borderTop: '1px dashed #cbd5e1', fontSize: '0.72rem', lineHeight: '1.45', color: '#1e293b' }
        });
        var acorContent = el('div', { id: 'acor-content-wrapper' }, [
            el('div', { id: 'acor-div' }),
            el('div', { className: 'legend' }, "Solid: Pol1 | Dashed: Pol2"),
            formulaDiv
        ]);
        var acorCard = el('div', { className: 'card' }, [
            acorHeader,
            acorContent
        ]);
        acorHeader.addEventListener('click', function() {
            if (acorContent.style.display === 'none') {
                acorContent.style.display = 'block';
                document.getElementById('acor-toggle-indicator').textContent = '▼';
                Plotly.Plots.resize('acor-div');
            } else {
                acorContent.style.display = 'none';
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

    // Initial render
    window.addEventListener('load', function() {
        renderPlot();
    });

})();
