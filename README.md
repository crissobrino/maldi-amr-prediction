# Antimicrobial Resistance Prediction from MALDI-TOF Mass Spectrometry

Coursework project for a private Kaggle competition run at Universidad Carlos III de Madrid
(Oct 2025 – Jan 2026), by a team of three: **Nuria Balbás, Ana Pascual, and Cristina Sobrino**.

## The problem

Antimicrobial resistance testing normally means growing a bacterial culture and exposing it to
each antibiotic — a process that takes **days**. MALDI-TOF mass spectrometry already runs
routinely in clinical microbiology labs to identify *which* bacterial species is present, and it
does so in **minutes**. If the same spectrum can also predict *resistance*, labs get an
actionable signal long before culture-based susceptibility results come back.

This project predicts resistance (0/1) to 8 antibiotics — Ampicillin, Levofloxacin,
Ciprofloxacin, Imipenem, Amoxicillin/Clavulanic acid, Ertapenem, Cefotaxime, and Cefuroxime —
directly from MALDI-TOF spectra of bacterial isolates, across 4 species (*E. coli*,
*K. pneumoniae*, *P. mirabilis*, *P. aeruginosa*). It's a multi-label, semi-supervised, partially
missing-data problem: 3,360 labelled training isolates (some antibiotics missing labels for some
rows), 1,000 test isolates, evaluated as the average AUC across the 8 antibiotics (public
leaderboard: a random 40% of test; private: the other 60%).

Two things make this hard:

- **Incomplete labels.** Some antibiotics — Amoxicillin/Clavulanic acid especially, with ~43%
  of rows missing — have far fewer labelled examples than others.
- **Species heterogeneity.** Resistance mechanisms differ across the 4 bacterial species, so
  species isn't just metadata — it's an informative feature that changes which model structure
  works best per antibiotic.

**Final result: 0.81567 AUC (private leaderboard), 0.84394 AUC (public leaderboard)** — against
a course benchmark of 0.80.

## Repo structure

```
amr-maldi-tof/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_random_forest_baseline.ipynb
    ├── 02_cnn_baseline.ipynb
    ├── 03_xgboost_baseline.ipynb
    ├── 04_catboost_per_antibiotic.ipynb
    ├── 05_feature_engineering_pca.ipynb
    ├── 06_svm_specialist_nystrom.ipynb
    ├── 07_pseudo_labelling.ipynb
    └── 08_final_ensemble.ipynb
```

Notebooks are numbered in pipeline/story order, not necessarily the order they were run on
Kaggle. Outputs (printed AUCs, plots) are left intact — GitHub renders `.ipynb` files inline, so
the numbers and charts are visible without re-running anything.

## The notebooks

1. **Random Forest baseline** — one Random Forest per antibiotic, plus exploratory PCA/clustering
   on the spectra and an early pseudo-labelling pass. Used as the simplest baseline (5-fold CV
   macro AUC ≈ 0.80).
2. **CNN baseline** — a PyTorch model built specifically for spectra: a `LocallyConnected1D`
   layer (deliberately *not* translation-invariant, since MALDI peaks sit at fixed m/z
   positions), a species embedding, a shared encoder, and one head per antibiotic. Trains and
   produces predictions but this notebook doesn't compute a validation AUC.
3. **XGBoost** — one model per antibiotic, with species-specific models for Amoxicillin/Clavulanic
   acid, Levofloxacin and Ciprofloxacin. Strong on carbapenems and cephalosporins (mean OOF AUC
   0.898); weakest on Amoxicillin/Clavulanic acid, the label-sparse target.
4. **CatBoost** — same per-antibiotic structure as the XGBoost baseline. Outperformed the
   individual XGBoost models (Macro AUC 0.898) and gave the best global ranking performance.
5. **Feature engineering — PCA** — tested whether a 50-component PCA latent representation of the
   spectra (fit leak-free per fold) would help CatBoost. It didn't: AUC dropped for every target
   tested (Ampicillin −0.007, Ciprofloxacin −0.058, Levofloxacin −0.038, Amox/Clav −0.055) —
   these antibiotics are feature-limited, not just noise-limited.
6. **Feature engineering + SVM specialist** — Nyström kernel approximation into a linear SVM,
   motivated by local spectral shifts and geometry-aware modelling. Only improved things for
   Ampicillin (OOF AUC 0.929), so it was kept as a specialist inside the final ensemble rather
   than a general-purpose model.
7. **Semi-supervised pseudo-labelling** — fold-safe teacher/student pseudo-labelling with CatBoost,
   tried per-target. Marginal for Cefotaxime (+0.004 AUC); for Amoxicillin/Clavulanic acid — the
   target this was really aimed at, given its ~43% missing labels — the run in this notebook
   errors out on a CUDA/GPU issue before finishing, so the comparison isn't captured here. Kept
   anyway: a pseudo-labelling attempt that mostly didn't pay off is a more honest record of what
   was tried than a cleaned-up success story.
8. **Final ensemble** — the most complete pipeline: XGBoost + CatBoost blended with an optimal
   per-antibiotic weight, a correlation-aware logistic-regression meta-stack for the weaker
   targets (using OOF predictions from correlated antibiotics), and another pass at species-aware
   pseudo-labelling. Highest macro OOF AUC found across all experiments (0.9039, bootstrap 95% CI
   [0.8999, 0.9082]).

**Headline findings:** CatBoost beat individual XGBoost models; combining models beat complex
stacking alone; and performance depended heavily on the antibiotic — carbapenems
(Imipenem, Ertapenem) were near-ceiling (AUC ≈ 0.99) while the heterogeneous beta-lactams
(Amoxicillin/Clavulanic acid especially) needed specialist models and ensembling to move the
needle.

| Antibiotic | CatBoost OOF AUC | XGBoost OOF AUC | Ensemble OOF AUC |
|---|---|---|---|
| Imipenem | 0.990 | 0.989 | 0.990 |
| Ertapenem | 0.988 | 0.988 | 0.989 |
| Cefuroxime | 0.946 | 0.943 | 0.947 |
| Cefotaxime | 0.932 | 0.931 | 0.934 |
| Ampicillin | 0.933 | 0.924 | 0.930 |
| Ciprofloxacin | 0.853 | 0.856 | 0.859 |
| Levofloxacin | 0.851 | 0.854 | 0.857 |
| Amoxicillin/Clavulanic acid | 0.736 | 0.705 | 0.731 |

*(from the project poster — see note below)*

## What's not here

- **LightGBM.** The original work used a per-antibiotic LightGBM model as a higher-variance
  diversity source for ensembling. No LightGBM training notebook was recovered — one experiment
  in this pull referenced a `submission_lgbm.csv` from a private "previous submissions" Kaggle
  dataset, confirming it existed, but the notebook that produced it wasn't found.
- A poster (exported from the team's presentation) is referenced for the results table above and
  for cross-checking the narrative in this README, but isn't included in this repo.

## Running these notebooks

These are Kaggle notebooks and won't run as-is outside Kaggle:

- All notebooks read from `/kaggle/input/antimicrobial-resistance-prediction-from-maldi-tof/...`,
  which only resolves inside a Kaggle kernel attached to the competition dataset.
- `03_xgboost_baseline.ipynb` has leftover (unused) Google Colab cells referencing
  `/content/drive/MyDrive/ML/Kaggle`.
- `02_cnn_baseline.ipynb` lists a private `/kaggle/input/prev-submissions/` dataset among its
  attached inputs, but never actually reads from it — safe to ignore.

Install dependencies with:

```
pip install -r requirements.txt
```
