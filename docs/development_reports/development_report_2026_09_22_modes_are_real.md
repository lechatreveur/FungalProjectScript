# The discrete modes are real; the clustering test could not see them

**Date:** 2026-09-22
**Experiments:** 2026_09_09_M162 (primary), with 2026_08_28_M160 and
2026_09_03_M161 as cross-checks
**Figures:** `figures/modes_are_real/`
**Supersedes:** the "no discrete dynamic modes" conclusion in
[the 2026-09-21 report](development_report_2026_09_21_m160_continuous_manifold.md)
and its 2026-09-22 correction section, and section 6b of
[the 2026-09-22 pipeline report](development_report_2026_09_22_m161_m162_pipeline_and_growth.md)

---

## 1. What prompted this

The repository owner, reading the newly built M162 explorer, reported that the
modes *are* visible: monopolar and bipolar separated by a gap.

That contradicted the conclusion I had reached and committed twice — that no
dataset in this project shows discrete structure. It turned out the owner was
right and the conclusion was an artifact of the test, not a property of the
data.

## 2. Why the earlier test was wrong

P16's null calibration asks the **unconditioned** question: *are there clusters
anywhere?* It compares a silhouette over k-means against a shuffled-dimension
null. That test is weak by construction, and weaker still for the geometry this
data actually has.

**The structure is not two blobs.** `pol2_mid` is a **point mass near zero plus
a long right tail**: roughly half the cells have no second pole at all, and the
rest have one, broadly distributed. That is a genuine discrete distinction —
a second pole is present or it is not — but it looks nothing like the
two-Gaussian geometry k-means and silhouette are built to detect. A spike
beside a continuum scores as one elongated cloud.

So the earlier conclusion should have been stated as *"the silhouette test does
not detect structure"*, which is a fact about the instrument. Instead it was
stated as *"there are no discrete modes"*, which is a claim about the biology,
and that claim was false.

## 3. The evidence

### 3.1 The feature is multimodal on its own (the decisive test)

No autoencoder, no UMAP, no threshold. Gaussian-mixture BIC for 1 vs 2 vs 3
components on `pol2_mid`, and Hartigan's dip against a Gaussian null:

| dataset | n | ΔBIC (2 vs 1) | best k | dip | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| M162 FL1 | 760 | **−815** | 3 | 0.502 | <0.001 |
| M161 FL1 | 363 | **−105** | 2 | 0.379 | <0.001 |
| **M160 all films** | 6,244 | **−2,627** | 3 | 0.480 | <0.001 |
| M160 FL1 | 861 | −13 | 2 | 0.338 | 0.010 |

**Every dataset is multimodal, including M160.** `Periodicity` is multimodal
too in M162 (ΔBIC −186, p < 0.001). `pol1_mid` is unimodal, as expected — in
M162 essentially every cell clears the polarity threshold.

### 3.2 Mode labels occupy distinct latent regions — with a caveat

k-nearest-neighbour label purity in the 6D latent space, against a
label-permutation null holding the geometry fixed (k=15, 200 permutations):

| | observed | null | z |
| --- | ---: | ---: | ---: |
| all five Mode labels | 0.776 | 0.384 ± 0.006 | **+71.0** |
| monopolar vs bipolar only (n=752) | 0.862 | 0.503 ± 0.006 | **+58.7** |

**This test is partly circular and must not be quoted as independent
evidence.** The Mode labels are produced by thresholding `pol2_mid`, and
`pol2_mid` is one of the eleven features fed to the autoencoder. Labels cut
from an input feature will separate in the latent space almost by construction.
What this shows is that the autoencoder *preserves* the split, which is worth
knowing but is not why we believe the split exists. Section 3.1 is why.

## 4. What this changes, and what it does not

**Withdrawn.** "No dataset in this project shows discrete structure." False.
All of them do, in `pol2_mid`.

**Withdrawn.** "A continuous, structureless polarity manifold is what an
arrested population should produce." Doubly wrong: M160 is not structureless,
and neither is the healthy arm.

**Retained.** The growth-rate result is untouched — M160 doubles in 57.7 h
against M162's 3.93 h, measured on brightfield areas with no reference to the
manifold or to any learned representation.

**Retained.** M160's low polarity contrast (pole/cytoplasm excess 3.9% against
M162's 12.0%) and its 4–10× lower `d` and `pol1_mid`. Arrest still explains
these: non-growing cells do not extend tips, so the signal is weak.

**Retained, and now better supported.** P16's warning that a silhouette on a
UMAP embedding is a poor instrument. The failure mode documented here is worse
than the one P16 describes: P16 warned the metric can *overstate* structure in
noise; this shows it can also *miss* structure that is plainly present in a raw
feature.

**Sharpened.** The honest description of the biology is **presence or absence
of a second pole**, not "two clusters". The bipolar side is itself a continuum.

## 5. Consequences for how this project tests for structure

1. **Test the features before testing the manifold.** A multimodality test on a
   raw feature is cheaper, needs no representation learning, and is not subject
   to the geometry assumptions that defeated the silhouette test. It should
   come first, always.
2. **Never report an unconditioned clustering result as evidence of absence.**
   "The silhouette test does not detect structure" and "there is no structure"
   are different statements, and only the first is supported by that test.
3. **A conditioned test built on a thresholded input feature is circular.** It
   can confirm that a representation preserves a known split; it cannot
   establish that the split is real.
4. The Mode thresholds in the explorer (`pol1 4.04, pol2 2.0, mono 1.14,
   bi 1.60`) are inherited from M160 and are not calibrated for M162, whose
   `pol2_mid` distribution is different. The 2.0 boundary happens to sit at the
   trough between the spike and the tail, which is why it works — but that is
   luck, not calibration, and it should be re-derived per experiment.

## 6. Next

FL2–FL4 are being quantified so the M162 map can carry all four fluorescence
films (2,270 cells). That will also make the explorer's same-cell navigation
useful: with FL1 alone, no global cell has two datapoints.

Then the open question worth asking is no longer "are there modes" but **do
cells switch modes over time** — which needs the multi-film map and is exactly
what the arrow-key navigation between a cell's datapoints was built for.
