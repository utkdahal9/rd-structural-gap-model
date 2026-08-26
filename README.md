# R&D Space Market Structural Gap — COVID-19 Counterfactual Analysis

Did COVID-19 permanently reshape the market for R&D-oriented industrial real
estate in the U.S.? This project builds a dynamic-panel LightGBM
counterfactual model across 100 U.S. metro areas (2005–2023) to measure
whether the post-pandemic supply of available R&D space diverged
structurally from what pre-pandemic trends would predict — and if so,
where, by how much, and when (if ever) it's projected to normalize.

**[Live dashboard →](#)** *(placeholder — add your deployed Streamlit URL here)*

## Key results
- MSA-level LOOCV R² ≈ 0.82 for the counterfactual available-space model.
- Post-COVID structural gaps are statistically significant in all four years
  (2020–2023); pre-COVID gaps are jointly indistinguishable from zero.
- Significant regional variation (Kruskal-Wallis H = 21.63, p = 0.0001),
  concentrated in a Northeast vs. South/West divide.
- Mean-reversion forecast projects when (or whether) markets structurally
  normalize by 2045, MSA by MSA.

Full write-up of the modeling choices, validation strategy, and limitations
is in [`docs/methodology.md`](docs/methodology.md).

## Repository structure
```
├── notebooks/           # Analysis, grouped by theme (run in numeric order)
│   ├── 01_main_model.ipynb          # LightGBM counterfactual + LOOCV + SHAP
│   ├── 02_model_validation.ipynb    # vs. 6 alternative model specs
│   ├── 03_regional_analysis.ipynb   # Census-region aggregation & testing
│   ├── 04_forecast.ipynb            # Mean-reversion forecast to 2045
│   ├── 05_rd_weighted_gap.ipynb     # R&D-employment-weighted gap
│   └── 06_case_studies.ipynb        # Named-metro deep dives
├── src/                  # Reusable, tested utilities
│   ├── regions.py           # Census region assignment
│   └── sanity_checks.py     # Panel-integrity guardrails for downstream notebooks
├── app/
│   └── dashboard.py       # Streamlit dashboard (see below)
├── tests/                # pytest unit tests for src/
├── docs/
│   └── methodology.md     # Full model spec, data sources, validation, limitations
├── figures/                # Exported charts
├── data/
│   ├── raw/                # Not committed — see data/raw/README.md
│   └── processed/          # Not committed — regenerate via notebooks/
├── requirements.txt
└── LICENSE
```

## Data
This project fuses proprietary real estate data (CoStar, JobsEQ) with public
sources (NSF HERD, BEA regional accounts, Census ACS/BDS, FHWA VMT) into a
100-MSA × 2005–2023 panel. **Raw data is not included in this repository**
because CoStar and JobsEQ are licensed sources that cannot be redistributed
— see `data/raw/README.md` for what's needed to reproduce the pipeline from
scratch, or reach out for access to a de-identified processed panel.

## Setup
```bash
git clone https://github.com/<your-username>/rd-covid-counterfactual.git
cd rd-covid-counterfactual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in order (each depends on CSV exports from the ones
before it — see the note at the top of each notebook):
```bash
jupyter lab notebooks/
```

Run the test suite:
```bash
pytest tests/
```

Launch the dashboard (once notebooks have generated the required CSVs):
```bash
streamlit run app/dashboard.py
```

## Background
Developed as part of graduate research at the University of Southern
Mississippi, supporting a NARSC (North American Regional Science Council)
conference paper, *"Measuring the Innovation Divide: The Metropolitan
Geography of Structural Gaps in U.S. R&D Space Markets."*

## License
MIT — see [LICENSE](LICENSE).
