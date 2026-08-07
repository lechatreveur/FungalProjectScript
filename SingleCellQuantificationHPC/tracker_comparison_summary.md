# Cell Tracker Benchmark Summary: SAM2 vs. Custom AI Tracker

Comparison evaluated on 20 cell tracks from M93 and 2025_09_17 datasets.

## Aggregate Performance Metrics

| Metric | Meta SAM2 (Tiny) | Custom AI Tracker (ResNet/HMM) | Improvement / Difference |
| :--- | :--- | :--- | :--- |
| **Average Mean IoU** | 0.898 | 0.863 | +0.034 |
| **Average Survival Rate** (IoU &ge; 0.5) | 98.7% | 86.4% | +12.3% |
| **Average Final Frame IoU** | 0.818 | 0.699 | +0.119 |
| **Propagation Speed** (FPS) | 1.1 | 3.5 | -2.3 |

## Performance by Experiment Group

| Experiment | Tracker | Mean IoU | Survival Rate | Final IoU |
| :--- | :--- | :--- | :--- | :--- |
| 2025_09_17 | Meta SAM2 | 0.900 | 99.6% | 0.806 |
| 2025_09_17 | Custom AI | 0.888 | 92.2% | 0.705 |
| 2026_01_08_M93 | Meta SAM2 | 0.895 | 97.8% | 0.830 |
| 2026_01_08_M93 | Custom AI | 0.838 | 80.6% | 0.694 |

## Individual Cell Tracking Detailed Breakdown

| Film | Cell ID | Status | Length (Frames) | SAM2 Mean IoU | Custom AI Mean IoU | SAM2 Final IoU | Custom AI Final IoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A14_BF_2_F1 | 25 | good | 36 | 0.815 | 0.971 | 0.436 | 1.000 |
| A14_BF_2b_F2 | 1 | corrected | 26 | 0.949 | 1.000 | 0.944 | 1.000 |
| A14_BF_2_F1 | 40 | corrected | 36 | 0.883 | 0.929 | 0.807 | 0.360 |
| A14_BF_1_F1 | 23 | corrected | 60 | 0.913 | 0.412 | 0.902 | 0.326 |
| A14_BF_2b_F1 | 33 | good | 26 | 0.902 | 0.997 | 0.861 | 1.000 |
| A14_BF_1_F1 | 20 | good | 60 | 0.857 | 0.431 | 0.832 | 0.380 |
| A14_BF_2b_F2 | 16 | corrected | 26 | 0.945 | 1.000 | 0.936 | 1.000 |
| A14_BF_2b_F1 | 34 | corrected | 26 | 0.937 | 0.994 | 0.928 | 1.000 |
| A14_BF_2_F1 | 9 | corrected | 36 | 0.849 | 0.966 | 0.784 | 0.322 |
| A14_BF_1_F0 | 16 | corrected | 60 | 0.901 | 0.684 | 0.871 | 0.547 |
| A14_1TP1_BF_F1 | 42 | corrected | 120 | 0.929 | 1.000 | 0.924 | 1.000 |
| A14_1TP1_BF_F1 | 9 | corrected | 120 | 0.891 | 0.728 | 0.652 | 0.417 |
| A14_1TP1_BF_F1 | 36 | corrected | 120 | 0.907 | 0.998 | 0.862 | 1.000 |
| A14_1TP2_BF_F1 | 53 | corrected | 120 | 0.866 | 0.798 | 0.407 | 0.084 |
| A14_1TP1_BF_F1 | 32 | good | 120 | 0.895 | 0.998 | 0.849 | 1.000 |
| A14_1TP1_BF_F1 | 41 | good | 120 | 0.860 | 1.000 | 0.817 | 1.000 |
| A14_1TP1_BF_F1 | 27 | corrected | 120 | 0.929 | 0.576 | 0.934 | 0.443 |
| A14_1TP1_BF_F1 | 69 | corrected | 120 | 0.900 | 0.940 | 0.867 | 0.422 |
| A14_1TP1_BF_F1 | 18 | corrected | 120 | 0.928 | 0.913 | 0.894 | 0.683 |
| A14_1TP1_BF_F1 | 26 | corrected | 120 | 0.894 | 0.928 | 0.858 | 1.000 |
