# Does the global model help each hospital?

**The question:** is a hospital better off using the shared FedAvg model, or a model trained only on its own data?

**Method (`scripts/global_vs_local.py`):**
- Every hospital trains two models for the same number of rounds (30; X-ray 20). Each round is 2 epochs, batch 32, learning rate 0.003.
  - **Own model:** trained only on that hospital's training split.
  - **Global model:** that hospital's share of FedAvg across all four hospitals.
- A **pooled** model is also trained on all hospitals' data together. Real hospitals can't do this, so it serves as the upper reference.
- Every model is scored on the **same hospital's held-out patients** using macro-F1. Those patients were never used for training.
- There are 3 seeds per modality (X-ray only 1, because it is slow on CPU).
- **Small-hospital test:** the same setup, but each hospital keeps only 60 training records (diabetes: 40).

**Data:** ECG, EHR and X-ray use the **SIMULATED** hospitals. Diabetes uses the **real PIMA** split.

## Results: mean macro-F1 over all hospitals' test patients

| Modality | Own | Global | Pooled |
|---|---|---|---|
| ECG | 0.960 | **0.997** | 0.997 |
| EHR | 0.637 | **0.673** | 0.665 |
| Diabetes (real) | 0.718 | 0.719 | 0.724 |
| X-ray (1 seed) | **0.766** | 0.712 | 1.000 |
| ECG, 60 per hospital | 0.747 | **0.936** | 0.930 |
| EHR, 60 per hospital | 0.541 | 0.540 | 0.631 |
| Diabetes, 40 per hospital | 0.654 | **0.671** | 0.690 |

Per-hospital bars are in `results/1_global_vs_own_normal.png` and `results/2_global_vs_own_small.png`.

## What it shows (and what it doesn't)

- **ECG:** the global model beats or matches every hospital's own model. It also equals the pooled model without any data being shared. The gain is largest when hospitals are small (+0.19 F1).
- **EHR:** there is a small gain at normal size (+0.04). With only 60 records per hospital, FedAvg does not reach the pooled score (0.54 vs 0.63).
- **Diabetes:** this is effectively a tie. PIMA is a single cohort split at random, so each hospital already sees the same kind of patients. The global model helps a little when hospitals are small (+0.02). The test sets are small (24–74 patients), so individual hospital numbers are noisy.
- **X-ray:** FedAvg has **not** converged in 20 rounds. It is better at hospitals A and C and worse at B and D. The pooled model reaches 1.0, so the task itself is learnable.
  - The X-ray federated result is a limitation, not a success.
  - Next steps would be more rounds, a lower learning rate, or FedProx.
- The global model also usually does better than a hospital's own model on **other** hospitals' patients (fields `own_on_others` / `glob_on_others`). That matters because an own model is tuned to one site.
- **Run-to-run variation:** a repeat EHR run with the same seeds gave different per-hospital numbers. Read single-hospital differences under about 0.03 as noise.
