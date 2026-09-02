# Staged handoff prompts

How to hand a multi-step change to another agent — Antigravity today, a scripted
`codex exec` verifier later — as a sequence of individually verifiable stages.
This is the repository's standard shape for that work. Worked examples live at
repository root: `antigravity_prompts_septum_board.md`,
`antigravity_prompts_m156_strip_bug_and_id_drift.md`,
`antigravity_prompts_m156_umap.md`.

Referenced by [PROJECT_POLICY.md](PROJECT_POLICY.md) P7.

## When to use it

- A change big enough that an agent starting cold would spend most of its effort
  re-deriving context this repository already knows (and getting some of it
  wrong).
- Any change whose last step is irreversible-ish (moving files, overwriting a
  canonical output, rebuilding a shipped dashboard) — the staging is what keeps
  that step from running against an unverified fix (P5).
- A change with a load-bearing assumption that should be proven before anything
  is built on it ("does the new tool actually read/write the same dataset?",
  "is the rotation fix actually correct?").

For a small, self-contained change, just make it. This is for the ones where the
cost of a wrong turn three steps in is high.

## Anatomy of a handoff file

```
# Prompts for <agent>: <one-line goal>

<Feed instruction: one at a time, in order. Which stage must be verified
 before the next. Which stage is the risky/irreversible one and why it is
 isolated to the end.>

Shared facts <agent> should not have to re-derive (confirmed against the
current code and data, dated YYYY-MM-DD):

- <fact>
- <fact>

---

## Stage 1 — <de-risk the core assumption>

​```
<verbatim prompt>
​```

## Stage 2 — <build on the verified foundation>

​```
<verbatim prompt>
​```

## Stage 3 — <irreversible step, gated on stages 1-2>

​```
<verbatim prompt>
​```
```

## The shared-facts preamble

The point is to spend your context, once, writing down what you have already
verified, so the downstream agent does not rediscover it unreliably.

- **Date it and source it.** "confirmed against the current code and data, dated
  2026-08-24". A reader then knows how much to trust it and when to re-check.
- **Use precise identifiers, not descriptions.** File paths, function names,
  line ranges (`pipeline.py` around line 499-521), exact cell / film / id values
  (`global_cell_id 3_F0_cell_79`, film `3_FL3_F0`, local id 91), exact column
  names (`new_cell_id` vs `local_fl_id`). A described location gets guessed at; a
  named one does not.
- **State the data model.** What a "global cell" is, which coordinate space the
  frontend is allowed to send (see [COORDINATE_SYSTEMS.md](COORDINATE_SYSTEMS.md)),
  which CSV rows to use and which to ignore, where a file is written and under
  what naming.
- **Give the root cause, not just the symptom.** "skimage `orientation` is
  measured from the row axis, so `-angle_deg` over-rotates by ~90°" — with the
  evidence that established it.
- **List the explicit "do NOT"s.** Do not register `masks_bp`. Do not resurrect
  the legacy-name fallback. Do not average in the `_1`/`_2` rows. Do not fork the
  experiment list into a second YAML. These are where a reasonable agent goes
  wrong.
- **Say what to do if a premise looks false.** "If that fix isn't present, stop
  and report it rather than reapplying blindly." The downstream agent must not
  proceed on, or silently repair, a broken assumption.

## Stage rules

1. **One stage = one verifiable outcome.** Stage 1 produces a trustworthy
   feature table, or a working read-only view, or a confirmed correctness fix —
   not "table plus a bit of UI".
2. **Every stage ends with `Verify by <concrete check>`.** Pixel-identical strip
   responses between two tools; a flag-summary script over all 732 cells; grep
   the shipped HTML for the pre-rename key. The check is named in the prompt, not
   left to the agent's judgement. This is the P2 ladder applied per stage.
3. **Stage 1 de-risks; it does not polish.** No UI work, no cleanup, no
   optimization until the load-bearing assumption is proven.
4. **The irreversible stage is last and explicitly gated.** "This stage moves
   files and should only run after stage 1's regeneration and stage 2's lookup
   fix are both verified." Move, do not delete. Print before/after counts so the
   action is auditable. This is P5, delivered as a prompt.
5. **Each stage prompt is a fenced block** so it can be pasted verbatim with no
   editing. Write it as an instruction to the agent, in the imperative.
6. **Later stages may restate the one or two facts they hinge on** rather than
   assuming the preamble is still in the agent's context.

## Template

Copy this to `handoff_<topic>_<date>.md` (repository root, next to the existing
examples) and fill it in.

```
# Prompts for <agent>: <goal>

Feed these to <agent> one at a time, in order. Stage <N> must be verified before
stage <N+1>. Stage <last> is <the irreversible step> — do not run it against an
unverified earlier stage.

Shared facts <agent> should not have to re-derive (confirmed against the current
code and data, dated <YYYY-MM-DD>):

- <architecture / where things live>
- <data model definitions>
- <API or file-format contract>
- <known root cause / prior bug, with the evidence>
- <explicit do-NOTs>
- <what to do if a stated premise looks false>

---

## Stage 1 — <de-risk the core assumption; no polish>

​```
<verbatim imperative prompt>

Verify by <concrete check with exact identifiers and expected result>.
​```

## Stage 2 — <build on the verified foundation>

​```
<verbatim imperative prompt>

Verify by <concrete check>.
​```

## Stage 3 — <irreversible step; gated>

​```
This stage <moves files / overwrites canonical output / rebuilds the shipped
artifact> and should only run after stages 1-2 are verified.

<verbatim imperative prompt>. Move, do not delete. Print before/after counts.

Verify by <concrete check>.
​```
```
