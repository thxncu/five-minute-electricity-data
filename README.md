# Reproducibility package

**Assessing the fitness for use of public electricity data across consumer policies and observation windows**

**Supplementary Material 2**  
**Author and corresponding author:** Chankook Park  
**Affiliation:** Division of Climate Change, Hankuk University of Foreign Studies, Global Campus  
**Address:** 81 Oedae-ro, Mohyeon-eup, Cheoin-gu, Yongin-si, Gyeonggi-do 17035, Republic of Korea  
**Email:** ctrend@hufs.ac.kr  

The package accompanies the article and Supplementary Material 1. It supplies source records, executable analyses and numerical reference results for the quality audit, consumer policies, observation windows, retrospective demand forecasting, temporal models, change screening and external clock-schema demonstration. Detailed tables and equations are in Supplementary Material 1.

## Environment and execution

Use Python 3.12. The tested interpreter is Python 3.12.14. The requirements fix NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, Matplotlib 3.10.8 and tzdata 2026.4. From the extracted package directory:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The supplied Conda specification provides an alternative:

```bash
conda env create -f environment.yml
conda activate electricity-data-analysis
python run_all.py
```

After dependency installation, all analyses operate offline on the supplied files. Paths are relative to the scripts or package directory. The runner uses two numerical-computation threads, a noninteractive plotting backend and the pinned time-zone database. Simulations and resampling use fixed seeds recorded in the scripts and scientific summaries.

Each full run recreates `derived_data/` and `reproduced_results/`. Those directories contain generated outputs only; do not place source files or independent work there. The runner leaves sources and fixed references unchanged. It executes panel construction, core results, figures, consumer-policy evaluation, forecasting, temporal analyses, the external-schema check and two reference-comparison scripts, in that order. The full analysis completed in about two minutes in the tested environment; forecasting and simulations account for most computation. Hardware, numerical libraries and available memory affect runtime.

## Package contents

| Location | Contents |
|---|---|
| `data_raw/` | Eight immutable quarterly Korea Power Exchange (KPX) source files, July 2024 through June 2026. |
| `data_raw_external/caiso/` | Three California Independent System Operator (CAISO) demand-chart files and a manifest containing source URLs, retrieval times and checksums. |
| `scripts/01_build_audited_panel.py` through `04_evaluate_decision_usability.py` | Source parsing, field predicates, consumer-policy gates, window statistics, placement references and core figures. |
| `scripts/06_downstream_forecast_benchmark.py` | Chronological forecasting benchmark, all saved test predictions and paired error comparisons. |
| `scripts/temporal/` | Exact-decimal residual analysis, publication-precision checks, run-length and Markov analyses, calibrated multichannel change screening and their figures. |
| `scripts/external_schema_check.py` | CAISO field, boundary-clock and daylight-saving checks. |
| `scripts/05_validate_reproduction.py` and `07_validate_extended_reproduction.py` | Numerical reference comparisons, source-checksum verification and cross-analysis consistency checks. |
| `expected_results/` | Fixed scientific tables, summaries, predictions and figure exports. |
| `validation/` | Two fixed key-result JSON references for the source audit and consumer-policy analysis. |

Execution creates the audited panel in `derived_data/`, core tables and figures in `reproduced_results/tables/` and `reproduced_results/figures/`, and additional results in `reproduced_results/downstream/`, `reproduced_results/temporal/` and `reproduced_results/external_schema/`. Comparison reports are written only to `reproduced_results/validation/` during execution.

## Results map

All table numbers below refer to the article and Supplementary Material 1. File paths are relative to `expected_results/`; a full run produces the corresponding paths under `reproduced_results/`.

| Article or supplementary item | Scientific result files |
|---|---|
| Main audit and window tables; Tables S1A–S17 | `tables/`, plus `key_results.json` and `mission_key_results.json` produced at runtime and compared with the fixed references in `validation/`. Table S14 uses exact decimal residuals and the supplied tolerance-sensitivity results. |
| Main Table VII | `downstream/main_table.csv`. Error columns prefixed `common_` use the identical common test cohort; scorable coverage uses the same evaluable-target denominator for all four policies. |
| Table S18 | `downstream/split_sample_counts.csv`, `validation_model_selection.csv` and `run_summary.json`. |
| Table S19 | `downstream/test_policy_scores.csv`. The scored cohort is each policy's own issued subset. |
| Table S20 | `downstream/marginal_test_scores.csv`. |
| Table S21 | `downstream/paired_block_bootstrap.csv`. |
| Table S22 | `downstream/monthly_test_scores.csv`. |
| Forecast values and retained quality evidence | `downstream/test_predictions.csv.gz`, with every scheduled test origin, target, prediction, issuance flag, quality-history count and reconstruction age. |
| Tables S23–S24 | `temporal/markov_transition_parameters.csv`, `markov_and_exact_window_comparison.csv`, `markov_chronological_holdout.csv`, `markov_run_length_diagnostics.csv`, `markov_quarterly_parameters.csv`, `markov_simulations_*.csv` and `markov_summary.json`. |
| Tables S25–S26 | `temporal/multichannel_change_threshold_sensitivity.csv`, `multichannel_transition_policy_sensitivity.csv`, the flagged-record, bracket and June-context tables, and `transition_summary.json`. |
| Table S27 | Domain hypotheses supported by the cited standards and historical operational report in the article. No synthetic component or adjudication data are supplied. |
| Table S28 | `external_schema/caiso_schema_check.csv` and `caiso_schema_check.json`. Main Table IX also maps PJM Interconnection (PJM) and European Network of Transmission System Operators for Electricity (ENTSO-E) documentation; those mappings do not represent downloaded observations. |
| Main Figures 1–5 and Figures S1–S5 | `figures/`. |
| Figure S6 | `temporal/figures/Temporal_Markov_Diagnostics.png`, with PDF and EPS exports. |
| Figure S7 | `temporal/figures/Temporal_Change_Screen_June30.png`, with PDF and EPS exports. |
| Publication precision and threshold basis | `temporal/publication_precision_record_diagnostics.csv`, `publication_precision_distribution.csv`, `supply_identity_residual_exact_distribution.csv`, `supply_identity_tolerance_exact_decimal.csv` and `rounding_summary.json`. |

## Source clocks and quality rules

The KPX files contain 209,323 released records on an expected grid of 210,240 five-minute slots. Source text and numerical tokens remain unchanged. Their timestamps are interpreted as Korea Standard Time (KST), nine hours ahead of Coordinated Universal Time (UTC). The source manifest in `expected_results/tables/source_file_manifest_sha256.csv` records each file's 256-bit Secure Hash Algorithm (SHA-256) checksum and coverage.

The supply-reserve residual is evaluated from original decimal tokens. The 1 MW exclusion threshold is a declared publication-precision tolerance, not an optimized physical-error threshold or a governor-deadband criterion. Exactly six records exceed it. The precision analysis distinguishes decimal representation from physical accuracy. Reserve-nesting flags describe a relation between released aggregates; the archive cannot identify their component definitions or acquisition timing.

The full-service policies retain the specified 11-point context interval on 30 June 2026. The separately calibrated change screen and its three-point June bracket are evaluated as policy sensitivities. A coincident change does not identify which released values are erroneous or establish a recovery time.

## Retrospective forecasting protocol

The target is released national demand 60 minutes after a five-minute origin. The feature vector contains seven source fields at the origin and preceding eleven grid slots, plus six target-calendar sine and cosine terms. There are 90 features. The model predicts demand change relative to the current demand input; the same input is added back to form the forecast. A persistence forecast retains that input.

Initial training uses July 2024 through June 2025. Validation uses July through December 2025. Earlier training origins whose targets enter the next period are excluded. The common validation cohort selects one iteration count from 100, 200 and 300 by mean root mean squared error (RMSE) across all four policies; 300 is selected. Final models are refitted on the 18 months before 2026 and evaluated once on January through June 2026. HistGradientBoostingRegressor uses squared-error loss, learning rate 0.05, at most 31 leaves, at least 40 samples per leaf, quadratic regularization 1.0, 255 bins and seed 20261011. Automatic early stopping is disabled. Features, hyperparameters, calendar periods and seed are shared across policies; eligible training samples depend on the policy.

Raw forecasts require a complete observed history for all seven fields and retain flagged values unchanged. Strict and permissive forecasts require their respective full-service history gates. Permissive acceptance retains reserve-nesting ambiguity. Conventional reconstruction masks fields according to the declared quality rule and carries only each field's last available value forward. Missing timestamps mask all seven fields; zero forecasts mask the forecast field; failed supply identities mask all four related fields; operating-reserve zero, conflict and transition flags mask operating reserve and its rate. Transition reconstruction does not alter capacity or demand. Nesting ambiguity remains flagged and unchanged. Reconstruction has no age limit; the maximum carried-forward age is 375 minutes. Flags and ages remain in saved outputs but are not model features. Targets are never reconstructed.

All policies use the same target screen: released demand must be observed and positive, and its record must have an absolute supply-identity residual no greater than 1 MW. The screen avoids selecting a faulty operand in an inconsistent relation; it does not independently verify demand. No test-period record fails the identity criterion, so 298 missing target observations account for all test exclusions.

There are 52,116 scheduled test origins, 51,818 evaluable target origins and 47,663 common scored origins. Scheduled coverage uses all scheduled origins. Scorable coverage uses the same 51,818 evaluable-target denominator for each policy. Main Table VII scores error only on the 47,663 common origins. Policy-specific and marginal-cohort scores remain separate because those cohorts may differ in difficulty. Scores include RMSE, mean absolute error and mean absolute percentage error. Paired intervals use 2,000 moving-block draws of seven consecutive calendar days with fixed fitted models. The intervals describe conditional error differences, not model-fitting uncertainty or causal effects of a policy.

## Temporal analysis and external schema

Exact complete-window counts are derived from usable-run lengths. The two-state Markov reference estimates usable-to-unavailable and unavailable-to-usable probabilities from adjacent eligibility states. Pooled results describe July 2024 through June 2026. A separate chronological check fits only the first 18 months and compares stationary window-loss predictions against the next six months. The test contains 52,128 window endings, including trailing pre-test history. The forecasting denominator is twelve origins smaller because forecasting also requires a target one hour later inside the archive.

Temporal intervals use 1,000 moving-block draws of seven consecutive calendar days. Daily sufficient statistics retain original adjacent transitions and do not invent transitions between sampled blocks. A separate set of 1,000 chain simulations examines run lengths and interruptions. Agreement on pooled mean loss does not validate stationary extrapolation or the observed episode-length distribution. Multichannel change screening uses calibration medians, scaled median absolute deviations and a fixed calibration quantile. Test states do not estimate those parameters. Flagged changes describe review burden; verified fault labels are unavailable for false-alarm or detection-rate estimation.

The external demonstration contains three selected CAISO demand-chart snapshots: an ordinary day, a daylight-saving fall transition and a spring transition. The adapter interprets the provider's Pacific Time calendar as `America/Los_Angeles`, retains boundary rows, tests both ambiguous-time folds and identifies nonexistent local times. Files without offset or fold metadata do not yield a unique elapsed-time grid. Optional blank demand-response fields remain distinct from missing current demand. The analysis reports schema and clock evidence and defers window availability. Three selected snapshots do not establish CAISO-wide reliability or transfer of forecast accuracy. The manifest records exact source URLs, retrieval times, hashes and selection rationale.

## Reference comparisons and interpretation limits

The first comparator checks every core reference table, key value and supplied figure export. The extended comparator checks all downstream, temporal and external-schema tables and scientific JSON values, including every saved test prediction, plus all eleven source-file checksums. Integer values and flags must match exactly. Extended floating comparisons use absolute tolerance 1e-7 and relative tolerance 1e-11; the core-table comparator uses absolute tolerance 1e-8 and relative tolerance 1e-10. Interpreter metadata and elapsed-time fields do not enter scientific comparisons. Figures must be regenerated in every supplied format; their numerical sources are compared as tables. Missing reference outputs fail the run. The final checks also verify the common forecast cohort, one-hour horizon, policy-set inclusion, two test denominators and six identity exclusions across analyses.

The quarterly archive does not record issue-time availability, release revisions or independent physical demand. Timestamp-respecting features and past-only reconstruction prevent future archive observations from entering forecasts, but do not establish real-time deployment performance. Policies select different training samples, so common-test comparisons still reflect training composition. A filled input is not evidence of a fresh measurement. Quality flags do not establish physical faults, and the temporal model does not estimate power-system reliability. The package supplies no artificial alarm labels or unobserved resource components.

Distribution and reuse of source data remain subject to the providers' terms. The package accompanies the study for editorial assessment and research reproducibility.
