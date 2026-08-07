async function loadExperiments() {
    const res = await fetch('/api/list_experiments');
    const data = await res.json();
    state.experiments = data.experiments || [];
    
    const expSelect = document.getElementById('experimentSelect');
    if (expSelect) {
        expSelect.innerHTML = state.experiments.map(e => `<option value="${e}">${e}</option>`).join('');
    }
    
    if (state.experiments.length > 0) {
        state.selectedExp = state.experiments[0];
        await loadFilmsAndSequences(state.selectedExp);
    }
}

async function loadFilmsAndSequences(exp) {
    const res = await fetch(`/api/list_films_and_sequences?experiment=${exp}`);
    const data = await res.json();
    state.sequences = data.sequences || [];
    state.films = data.films || [];
    
    const seqSelect = document.getElementById('sequenceSelect');
    if (seqSelect) {
        if (state.sequences.length > 0) {
            seqSelect.innerHTML = state.sequences.map(s => `<option value="${s}">${s}</option>`).join('');
            seqSelect.value = state.sequences[0];
        } else {
            seqSelect.innerHTML = '<option value="">No datasets available</option>';
        }
    }
    
    if (state.sequences.length > 0) {
        state.selectedSequence = state.sequences[0];
        state.selectedCell = null;
        await loadCells(exp, state.selectedSequence);
    }
}
