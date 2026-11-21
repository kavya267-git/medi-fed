# Diabetes model and prediction for new patients

Three additions answer "what is the trained model for?" and "what if our columns are named differently?":

1. **Diabetes** is a fourth kind of data, trained federatedly like the others. The real records are the PIMA
   Indians Diabetes Database, split across hospitals A–D.
2. **Predict** in the MediFed Hospital Agent uses the global model that the hospital kept from its last round on
   a new patient, and explains the result. This runs on the hospital's own computer and sends nothing.
3. **Import data** in the agent matches the hospital's own export to the model's inputs, translates text and
   units, and writes the data folder.

Both are research prototypes, not medical devices. Every prediction carries that statement.

## Diabetes data

| Column | Meaning | Unit | Accepted range |
| --- | --- | --- | --- |
| `pregnancies` | Number of times pregnant | count | 0–25 |
| `glucose` | Plasma glucose 2 h into an oral glucose tolerance test | mg/dL | 30–600 |
| `diastolic_bp` | Diastolic blood pressure | mmHg | 20–160 |
| `skin_thickness` | Triceps skin fold | mm | 2–120 |
| `insulin` | 2-hour serum insulin | µU/mL | 2–1200 |
| `bmi` | Body mass index | kg/m² | 10–90 |
| `pedigree` | Diabetes pedigree function (family history score) | score | 0–3 |
| `age` | Age | years | 18–120 |
| `label` | 1 = diabetes, 0 = no diabetes | | 0 or 1 |

- **Layout.** The folder is `<data folder>/diabetes/train.csv` and `val.csv` (75% / 25%), the same layout as EHR.
- **Empty cells.** These are allowed. Each node fills them by KNN imputation fitted on its own training records.
- **Missing-value limit.** Up to 65% of a column may be empty. The limit is higher than EHR's 50% because 2-hour
  insulin is missing for 49% of the real PIMA records.
- **Ranges.** The ranges are plausibility checks, not clinical reference ranges. They catch unit mistakes, such as
  glucose entered in mmol/L.
- **Standardisation.** Inputs are scaled with fixed reference values: the PIMA mean and SD of each measurement.
  These are constants in `ml/diabetes/features.py`, so no hospital shares its own statistics.

### Real records for the demo hospitals

```bash
python scripts/prepare_real_data.py diabetes --pima
```

This command:
- downloads the PIMA file once to `data/raw/pima/diabetes.arff`, which is git-ignored, and checks it against the
  published MD5;
- turns the PIMA "0 = not measured" convention into gaps;
- splits the 768 records across hospitals A–D with the same Dirichlet split as experiment 1g (alpha 1.0, seed 0).

The resulting counts:

| Hospital | Training | Held-out |
| --- | --- | --- |
| A | 164 | 54 |
| B | 72 | 24 |
| C | 119 | 40 |
| D | 221 | 74 |

Run it on the coordinator laptop for the in-process demo nodes. When the nodes run on separate laptops, run it
once and copy each `data/hospitals/hospital_x/diabetes/` folder to that hospital's laptop.

A hospital's own file goes in with `diabetes --csv patients.csv`, optionally with:
- `--mapping '{"Glucose": "glucose", ...}'`
- `--label-column Outcome`
- `--zero-means-missing`

Without any of these, the simulated hospitals get **SIMULATED** diabetes records (`ml/simulation/diabetes.py`), as
they do for the other kinds of data.

### Model

The model is a small neural network: 8 inputs → 16 ReLU units → 2 classes (`ml/diabetes/mlp.py`). It trains with
the same rounds, validation, screening, FedAvg, federated evaluation and DP options as the other models. It uses
class-balanced loss, like EHR.

Why not TabNet? On the real PIMA split, 30 rounds over 3 seeds gave these pooled held-out ROC-AUC scores:

| Model | ROC-AUC |
| --- | --- |
| Small network | 0.80–0.81 |
| TabNet | ≈ 0.77 |
| Logistic regression | ≈ 0.81 |

The comparison was made after seeing those held-out scores. It is a design choice, not a pre-registered result.
For pre-registered PIMA numbers, see experiment 1g in `experiments/RESULTS.md`.

## Predict (MediFed Hospital Agent)

**Predict…** opens a window with the hospital's saved global models for EHR and diabetes. The model is the copy
the agent keeps after each completed round: `global_latest.npz` and `.json` in the agent's folder.

For one patient:
1. Type the measurements.
2. Leave a box empty if it was not measured.
3. Press **Predict**.

The window shows:
- **Result.** "Diabetes" or "No diabetes", with the probability and the 50% threshold.
- **Reasons.** The measurements that raised the risk most and those that lowered it most. These are Shapley
  values, computed against 32 of the hospital's own training records.
- **Estimates.** Any value the agent estimated, by KNN imputation from the hospital's training records, and the
  estimated value.
- **Model.** The model version and the round it came from.
- **Disclaimer.** "Research prototype, not a medical device."

**Score a CSV file…** scores every row of a file that has the same column names. It writes
`<file>_predictions.csv` next to the file, with these columns:
- `probability`
- `prediction`
- `estimated_inputs`
- `main_reasons`
- `problem`, for a row that could not be scored, for example a value outside its range.

Before predicting, the agent checks:
- the saved file's SHA-256 against its record;
- that the model's architecture matches the agent's.

Values outside the plausibility range are refused with "check the units".

Code: `ml/prediction/local.py`, which is the logic and is tested in `ml/tests/test_prediction.py`, and
`hospital_agent/medifed_agent/predict_window.py`, which is the window.

## Import data (MediFed Hospital Agent)

Hospitals name their columns differently: "Age", "AGE_YRS" or "pt_age"; "Pulse", "HR" or "heart_rate_bpm". The
agent does not guess silently. **Import data…** lets the hospital match its own CSV export to the model's inputs
on its own computer:

1. **Choose the export.** Pick the kind of data (EHR or diabetes) and the CSV file.
2. **Check each match.** Every model input has a dropdown of the file's own headings, with a guess
   pre-selected and the reason shown next to it, most certain first:
   - **by name:** common names, also inside headings with units or punctuation ("HR_BPM", "Heart Rate (bpm)",
     "BP Systolic (mmHg)");
   - **by LOINC code** in the heading, for example `8867-4` heart rate or `2160-0` creatinine;
   - **similar name:** a known word inside a longer heading ("S. Creatinine", "HGB g/dL");
   - **spelling variant:** "Creatnine";
   - **from the values:** for coded headings such as "obs_01", when the values are distinctive (a column of
     36-40 for temperature, 60-120 for heart rate). Only inputs with distinctive values are guessed this way.

   Weaker matches are marked "please check". The person checks every row, or picks **not recorded**.
3. **Pick the outcome and patient columns.** Choose the outcome column. Optionally choose a patient-ID column,
   which keeps one patient's rows on the same side of the 75% / 25% split.
4. **Translate text values.** The agent lists the text it found ("Died" / "Discharged", "M" / "F",
   "tested_positive"). The person ticks which values mean yes, or male.
5. **Confirm unit conversions.** When a column's typical value suggests other units, the agent offers to
   convert. Each conversion can be unticked:
   - °F → °C
   - glucose mmol/L → mg/dL
   - creatinine µmol/L → mg/dL
   - haemoglobin g/L → g/dL
   - SpO2 as a fraction → %

   For diabetes, the person can also mark "0 means not measured", as in the PIMA file.
6. **Import.** The agent writes `train.csv` and `val.csv` into the data folder, runs the local data check
   (pass or fail with the problems listed) and saves the matching. The next export with the same headings
   opens pre-filled.

**Hospitals with different measurements.** A hospital does not need to record every input:
- **Inputs it does not record.** These are filled with the input's reference value. A column that is more than
  half empty (65% for diabetes) is treated the same way, because KNN cannot fill it reliably. The shared model
  learns those inputs at other hospitals, and at this hospital it predicts from the inputs that are there.
- **Extra columns.** Columns the model does not use (cholesterol, ward) are ignored.
- **Coverage.** The data check and the dashboard's hospital page show "12 of 14 inputs recorded (not recorded:
  lactate, haemoglobin)". Only input names and counts are shared.

How much this costs was measured on the real PIMA split (30 rounds, 3 seeds; held-out ROC-AUC):

| Situation | ROC-AUC |
| --- | --- |
| Every hospital records everything | 0.81 |
| One hospital does not record insulin or skin fold | 0.81 |
| A patient's glucose is unknown | 0.77 |
| A patient's BMI and age are unknown | 0.78 |

Training with randomly hidden inputs ("input masking" at 15% or 30%) was also tried. It changed ROC-AUC by
0.005 or less, so it is not used.

**Scoring a CSV.** **Score a CSV file…** in the Predict window uses the same saved matching, so a file of new
patients with the hospital's own headings scores directly. A measurement missing from that file is estimated
from the hospital's own records.

Code: `ml/common/column_mapping.py`, tested in `ml/tests/test_column_mapping.py`, and
`hospital_agent/medifed_agent/import_window.py`.

## What it is not

- **No server-side prediction.** The website never sees patient values, so there is no prediction page on it.
- **No tuned threshold.** The 50% threshold is not tuned, and the probabilities are not calibrated for any real
  population.
- **No clinical validation.** PIMA is one small public dataset of adult women of Pima heritage. A model trained
  on it says nothing about other populations.


## ECG heartbeats

**Predict** in the agent also labels heartbeats with the global ECG model (CNN-BiLSTM+RR) that the hospital kept
from its last ECG round. Choose **ECG heartbeat class** in the model list, then **Open heartbeat file…**.

- **Input.** One heartbeat per row, 187 samples at 125 Hz (the MIT-BIH heartbeat format), optionally followed by
  the beat's pre-RR, post-RR and local average RR, each divided by the patient's average RR (190 columns).
  Without them a regular rhythm (1.0) is assumed, and early S and V beats are missed more often.
  - A 188th column with a reference label (0 = N, 1 = S, 2 = V, 3 = F, 4 = Q) is allowed, as in the Kaggle
    `mitbih_test.csv` export. The agent then also reports how often the model agreed with it. Q beats are outside
    the model's four classes and are not counted.
  - CSV, NPY, or the platform's own NPZ (`signals`, `rr`, `labels`) are accepted, so `data/hospitals/hospital_a/ecg/val.npz`
    works as a quick demo file. Those beats are SIMULATED.
- **Output.** Every beat is labelled N (normal), S (supraventricular ectopic), V (ventricular ectopic) or F (fusion),
  with the probability of each class. The results go to `<file>_ecg_predictions.csv` next to the input.
- **Key region.** For each beat (up to 300 per file), the agent flattens 20-sample windows one at a time and finds
  the window whose removal lowers the predicted class's probability the most (occlusion sensitivity). The window is
  shaded on the beat plot and reported in milliseconds.
- **Privacy.** The beats, labels and explanations stay on the hospital's computer. Nothing is sent.
- **Rhythm effect.** For each explained beat the agent also reports how much the predicted class's probability
  drops when the RR inputs are replaced by a regular rhythm: a large drop means the timing, not the shape, drove
  the label.
- **Limits.** On real MIT-BIH data with unseen patients, this model (FedAvg with RR inputs, experiment 1f) reached
  macro-F1 0.477, against 0.409 for beat shape alone; S and F beats are still often missed. Treat every label as a
  research output, not a diagnosis. Code: `ml/prediction/ecg.py`; tests: `ml/tests/test_ecg_prediction.py`.

## RR intervals in the ECG model (agent 1.3.0)

The platform's ECG model is now **CNN-BiLSTM+RR**: the beat-shape network of experiments 1a–1e, with each beat's
pre-RR, post-RR and local average RR (relative to the patient's mean RR) joining the pooled features before the
classifier. Experiment 1f measured the gain on real MIT-BIH records with unseen patients: macro-F1 0.409 → 0.477.

- **Data files.** `ecg/train.npz` and `val.npz` may hold an `rr` array ([n, 3]) next to `signals` and `labels`.
  - `scripts/prepare_real_data.py ecg --mitdb <folder>` builds them from the PhysioNet MIT-BIH Arrhythmia Database
    records (44 non-paced records, one patient each), with RR intervals. Each patient stays in one hospital and one split.
  - The Kaggle heartbeat export (`--csv`) has no RR intervals; its beats are read as a regular rhythm, and the
    dataset description says so.
  - Simulated hospitals get SIMULATED RR intervals: S and V beats come early, and V beats are followed by a
    compensatory pause.
- **Compatibility.** The new model has a different architecture signature, so the coordinator starts a new ECG
  model line and never averages it with old CNN-BiLSTM weights. Hospitals need agent 1.3.0 or later to train it.
- **Code.** `ml/ecg/preprocess.py` (`ecg_input`, `prepare_ecg_input`), `ml/ecg/cnn_bilstm.py`
  (`CNNBiLSTMRRAdapter`), `ml/simulation/ecg.py` (`simulate_rr`), `ml/common/real_data.py`
  (`prepare_mitdb_records`).
