async function loadExperiments() {
    try {
        const res = await fetch('/api/list_experiments');
        const data = await res.json();
        state.experiments = data.experiments || [];
        
        const expSelect = document.getElementById('expSelect');
        if (expSelect) {
            expSelect.innerHTML = state.experiments.map(e => 
                `<option value="${e.id}">${e.display_name || e.id}</option>`
            ).join('');
            
            if (state.experiments.length > 0) {
                // Default to M160 if present, else first
                const m160 = state.experiments.find(e => e.id.includes('M160'));
                state.selectedExp = m160 ? m160.id : state.experiments[0].id;
                expSelect.value = state.selectedExp;
                await loadFilmsAndSequences(state.selectedExp);
            }
        }
    } catch (err) {
        console.error("Failed to load experiments:", err);
    }
}

async function loadFilmsAndSequences(exp) {
    try {
        const res = await fetch(`/api/list_films_and_sequences?experiment=${exp}`);
        const data = await res.json();
        state.sequences = data.sequences || [];
        state.films = data.films || [];
        
        const seqSelect = document.getElementById('sequenceSelect');
        if (seqSelect) {
            if (state.sequences.length > 0) {
                seqSelect.innerHTML = state.sequences.map(s => `<option value="${s}">${s}</option>`).join('');
                seqSelect.value = state.sequences[0];
                state.selectedSequence = state.sequences[0];
            } else if (state.films.length > 0) {
                seqSelect.innerHTML = state.films.map(f => `<option value="${f}">${f}</option>`).join('');
                seqSelect.value = state.films[0];
                state.selectedSequence = state.films[0];
            } else {
                seqSelect.innerHTML = '<option value="">No datasets available</option>';
                state.selectedSequence = null;
            }
        }
        
        if (state.selectedSequence) {
            state.selectedCell = null;
            await loadCells(exp, state.selectedSequence);
        }
    } catch (err) {
        console.error("Failed to load films/sequences:", err);
    }
}
