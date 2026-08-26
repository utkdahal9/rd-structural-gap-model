"""
R&D Space Market Structural Gap — Interactive Dashboard
=========================================================
Streamlit app for exploring the COVID-19 structural-gap counterfactual
model's results: national trends, by-MSA detail, regional comparisons,
SHAP drivers, and the 2045 mean-reversion forecast.

Run with:  streamlit run app/dashboard.py

Reads CSV exports produced by the notebooks in notebooks/ (run 01-06
first). Each section degrades gracefully with a clear message if its
source file isn't present yet, rather than crashing the whole app.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.regions import add_region_column, REGIONS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

st.set_page_config(
    page_title="R&D Space Market Structural Gap",
    layout="wide",
)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def missing_data_notice(filename: str):
    st.info(
        f"`{filename}` not found in `data/processed/`. Run the relevant "
        f"notebook first, or point `DATA_DIR` at your export location."
    )


st.title("R&D Space Market Structural Gap")
st.caption(
    "Counterfactual analysis of COVID-19's structural impact on R&D "
    "industrial real estate across 100 U.S. metro areas, 2005–2023."
)

tab_overview, tab_msa, tab_regional, tab_forecast = st.tabs(
    ["National Overview", "By MSA", "Regional", "2045 Forecast"]
)

# ── National overview ────────────────────────────────────────────────
with tab_overview:
    results = load_csv("AvailSFTotal_Counterfactual_Results.csv")
    if results is None:
        missing_data_notice("AvailSFTotal_Counterfactual_Results.csv")
    else:
        national = (
            results.groupby("Year")[["Available_SF_Total", "Counterfactual_Space_SF"]]
            .mean()
            .reset_index()
        )
        fig = px.line(
            national,
            x="Year",
            y=["Available_SF_Total", "Counterfactual_Space_SF"],
            labels={"value": "Available SF (mean across MSAs)", "variable": ""},
            title="National Actual vs. Counterfactual Available Space",
        )
        fig.add_vrect(x0=2020, x1=2023, fillcolor="gray", opacity=0.1, line_width=0,
                       annotation_text="COVID period")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        recent = results[results["Year"] == results["Year"].max()]
        col1.metric("MSAs in panel", results["MSA_Name"].nunique())
        if "Structural_Gap" in recent:
            gap_pct = (np.exp(recent["Structural_Gap"]) - 1).mean() * 100
            col2.metric("Mean structural gap (latest year)", f"{gap_pct:+.1f}%")
        else:
            col2.metric("Mean structural gap (latest year)", "—")
        col3.metric("Years covered", f"{results['Year'].min()}–{results['Year'].max()}")

# ── By-MSA detail ─────────────────────────────────────────────────────
with tab_msa:
    results = load_csv("AvailSFTotal_Counterfactual_Results.csv")
    if results is None:
        missing_data_notice("AvailSFTotal_Counterfactual_Results.csv")
    else:
        msa_choice = st.selectbox("Select an MSA", sorted(results["MSA_Name"].unique()))
        msa_df = results[results["MSA_Name"] == msa_choice].sort_values("Year")
        fig = px.line(
            msa_df,
            x="Year",
            y=["Available_SF_Total", "Counterfactual_Space_SF"],
            title=f"{msa_choice} — Actual vs. Counterfactual",
        )
        st.plotly_chart(fig, use_container_width=True)
        if "Structural_Gap" in msa_df:
            st.caption(
                "Structural_Gap is in log-space (the model predicts "
                "log available space): positive = actual space exceeds "
                "the counterfactual (surplus), negative = deficit."
            )
            st.dataframe(
                msa_df[["Year", "Structural_Gap", "Market_Category"]]
                if "Market_Category" in msa_df
                else msa_df[["Year", "Structural_Gap"]],
                use_container_width=True,
                hide_index=True,
            )

# ── Regional comparison ────────────────────────────────────────────────
with tab_regional:
    results = load_csv("AvailSFTotal_Counterfactual_Results.csv")
    if results is None:
        missing_data_notice("AvailSFTotal_Counterfactual_Results.csv")
    else:
        regioned = add_region_column(results)
        yearly = (
            regioned.dropna(subset=["Region"])
            .groupby(["Year", "Region"])["Structural_Gap"]
            .mean()
            .reset_index()
        ) if "Structural_Gap" in regioned else None
        if yearly is not None:
            fig = px.line(
                yearly,
                x="Year",
                y="Structural_Gap",
                color="Region",
                category_orders={"Region": REGIONS},
                title="Mean Structural Gap by Census Region (log-space)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("`Structural_Gap` column not found in results export.")

# ── Forecast ───────────────────────────────────────────────────────────
with tab_forecast:
    forecast = load_csv("MeanReversion_RDWeighted_ByMSA_v14b.csv")
    if forecast is None:
        missing_data_notice("MeanReversion_RDWeighted_ByMSA_v14b.csv")
    else:
        st.dataframe(forecast, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Model details, validation, and limitations: see docs/methodology.md in the repo. "
    "Data: CoStar, JobsEQ, NSF HERD, BEA, Census ACS/BDS, FHWA."
)
