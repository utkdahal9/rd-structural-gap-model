# Methodology

## Research question
Did COVID-19 cause a lasting structural shift in the supply of available
R&D-oriented industrial/flex real estate across U.S. metro areas — beyond
what pre-pandemic trends would have predicted?

## Model specification
A pooled LightGBM regression with one-year-lagged features, trained only on
pre-pandemic data (2006–2019) and used to generate a **counterfactual**
prediction for 2020–2023 — i.e. "what would available space have looked like
in each MSA if COVID had not happened."

```
log(Available_SF_Total_t) = f(RD_Econ_{t-1}, RE_{t-1}, Supply_{t-1},
                                BDS_{t-1}, VMT_{t-1}, Productivity_{t-1},
                                Patents_{t-1}, Year_FE)
```

The **structural gap** for each MSA-year is:

```
Structural_Gap = Available_SF_Total (actual) − Counterfactual_Space_SF (predicted)
```

A persistent negative gap (less available space than the counterfactual
predicts) is read as a structural deficit; a persistent positive gap as a
structural surplus.

## Data sources
| Source | Content | Access |
|---|---|---|
| CoStar Group | Real estate: available SF, rents, inventory | Proprietary — institutional license required |
| JobsEQ | Advanced-industry employment, location quotients | Proprietary — institutional license required |
| NSF HERD | R&D expenditure by institution/region | Public |
| BEA Regional Accounts | GDP by industry, MSA-level | Public |
| Census ACS / BDS | Demographics, firm-age/size dynamics | Public |
| FHWA | Vehicle miles traveled (VMT) | Public |

Panel: 100 U.S. MSAs (2 of the original 102 — Hattiesburg, MS and
Gulfport-Biloxi, MS — are excluded across the entire pipeline due to
insufficient native CoStar coverage requiring full KNN imputation), annual,
2005–2023.

## Validation
- **MSA-level leave-one-out cross-validation (LOOCV)**, R² ≈ 0.82 for the
  available-space model.
- **Model comparison**: LightGBM benchmarked against Random Forest, XGBoost,
  Ridge, OLS, a naive AR(1) persistence baseline, and a two-way
  fixed-effects panel regression, on an identical feature set and
  evaluation protocol.
- **Bias correction**: a size-conditional correction is fit and applied
  because raw residuals correlated with MSA size in earlier iterations.
- **Feature leakage controls**: near-tautological features
  (`log_CoStar_Inventory_SF`, `Shadow_Space_SF`) are excluded; an AR(1) term
  was deliberately excluded from the main feature set after it was found to
  dominate SHAP importance and suppress economic signal (quantified
  separately via the standalone AR(1) baseline comparison).
- **Robustness checks explored during development**: synthetic control,
  synthetic difference-in-differences, and hierarchical Bayesian panel AR
  specifications, prior to settling on the LightGBM dynamic panel approach.

## Interpretability
- **Global**: SHAP mean-|value| feature importance, cross-checked against
  standardized OLS coefficients on the same feature set.
- **Local**: per-observation SHAP decompositions for individual MSA-years,
  including all case-study markets' 2023 predictions specifically (the
  fitted TreeExplainer applied out-of-sample to the predict-period rows).

## R&D-weighting
A secondary, R&D-weighted version of the structural gap rescales each MSA's
raw square-footage gap by `LQ_AdvInd_Emp / (1 + LQ_AdvInd_Emp)` — a bounded
(0,1) weight anchored to the MSA's 2015–2018 average advanced-industry
employment location quotient. This down-weights large gaps in MSAs with
little R&D-adjacent economic activity, and up-weights gaps in MSAs where
that activity is concentrated. The location-quotient denominator uses a
true national advanced-industry employment share (external, year-indexed),
not an in-panel average, so it isn't sensitive to which MSAs happen to be
in the training panel.

## Forecast
A Monte Carlo mean-reversion model projects each MSA's structural gap back
toward its own 2015–2018 equilibrium anchor (λ = 30%/year), aggregated
nationally on an R&D-weighted basis to produce a point estimate (and
uncertainty band) for when the market is projected to structurally
normalize, out to 2045.

## Statistical testing
- Kruskal-Wallis test for regional differences in structural gap
  (H = 21.63, p = 0.0001), followed by Bonferroni-corrected pairwise
  Mann-Whitney U tests (Northeast vs. South/West significant).
- LISA spatial clustering with Benjamini-Hochberg FDR correction (no raw-
  significant clusters survived correction in the current run).
- A LOOCV-corrected event study confirming pre-COVID gaps were jointly
  statistically indistinguishable from zero, while post-COVID gaps were
  significant in all four years.

## Known limitations
- The Kruskal-Wallis / Mann-Whitney tests treat each MSA-year as an
  independent observation, which understates true variance given repeated
  observations of the same MSA over time.
- The "primary state" region assignment for multi-state MSA names is a
  simplification (first state token), not an official Census rule.
- Two MSAs are excluded from the panel for data-quality reasons (see above);
  results generalize to the other 100 major U.S. metros, not to small or
  data-sparse metros generally.
