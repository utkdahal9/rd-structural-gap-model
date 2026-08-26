"""
R&D Space Market Structural Gap — Interactive Dashboard
=========================================================
Streamlit app for exploring the COVID-19 structural-gap counterfactual
model's results: national trends, by-MSA detail, regional comparisons,
and the 2045 mean-reversion forecast.

Run with:  streamlit run app/dashboard.py

Reads CSV exports produced by the notebooks in notebooks/ (run 01-06
first). Each section degrades gracefully with a clear message if its
source file isn't present yet, rather than crashing the whole app.

UNITS NOTE: the underlying model predicts log(Available_SF_Total), so
Structural_Gap in the source CSVs is a log-space value. Everywhere this
app displays that number to a person, it converts it to a percentage
via (exp(x) - 1) * 100 for readability.

NARRATIVE NOTE: every chart is followed by a plain-English "takeaway"
sentence computed live from the actual data, not hardcoded, so a
non-technical reader (e.g. a hiring manager) gets the "so what" without
needing to interpret the numbers themselves.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def to_pct(series: pd.Series) -> pd.Series:
    """Convert the model's log-space Structural_Gap to a plain percentage."""
    return (np.exp(series) - 1) * 100


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column matching one of the candidate names, case-insensitively,
    falling back to a substring match. Returns None if nothing matches."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        if any(cand.lower() in c.lower() for cand in candidates):
            return c
    return None


def describe_gap(pct: float, subject: str = "metros") -> str:
    """Plain-English description of a structural gap percentage."""
    if pct > 5:
        return (
            f"That's a **surplus**: on average, {subject} currently have "
            f"{pct:+.1f}% more available R&D space than the pre-COVID "
            f"trend would have predicted."
        )
    elif pct < -5:
        return (
            f"That's a **deficit**: on average, {subject} currently have "
            f"{pct:+.1f}% less available R&D space than the pre-COVID "
            f"trend would have predicted — i.e. tighter supply than expected."
        )
    return (
        f"That's roughly **balanced** — the {pct:+.1f}% average gap is "
        f"close to what the pre-COVID trend predicted for {subject}."
    )


# ── Header / intro ──────────────────────────────────────────────────
st.title("R&D Space Market Structural Gap")
st.markdown(
    "**The question this project answers:** did COVID-19 permanently "
    "change how much R&D-oriented industrial space (labs, flex space, "
    "advanced-manufacturing buildings) is available across major U.S. "
    "metro areas — or did markets simply return to their pre-pandemic "
    "trajectory?\n\n"
    "**How:** a machine-learning model was trained only on data from "
    "before COVID (2006–2019) and used to predict what 2020–2023 "
    "*should* have looked like if the pandemic never happened. The gap "
    "between that prediction and what actually happened is the "
    "**structural gap** — the pandemic's lasting footprint on this "
    "real estate market, isolated from the normal ups and downs."
)
st.caption(
    "Counterfactual analysis across 100 U.S. metro areas, 2005–2023. "
    "Full methodology: `docs/methodology.md` in the repo."
)

tab_overview, tab_msa, tab_regional, tab_forecast = st.tabs(
    ["National Overview", "By MSA", "Regional", "2045 Forecast"]
)

GAP_HELP = (
    "How much more (or less) available R&D space an MSA has, on "
    "average, than the model predicts it would have had if COVID had "
    "never happened. +40% means 40% more available space than the "
    "pre-COVID trend predicted — a surplus. Negative means a deficit."
)

# ── National overview ────────────────────────────────────────────────
with tab_overview:
    st.markdown(
        "This tab shows the **national picture**: how much available "
        "space actually existed each year (blue/red line below) versus "
        "what the model expected without COVID (the counterfactual line)."
    )
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
        latest_year = int(results["Year"].max())
        col1.metric("MSAs in panel", results["MSA_Name"].nunique())
        gap_pct = None
        if "Structural_Gap" in recent:
            gap_pct = to_pct(recent["Structural_Gap"]).mean()
            col2.metric(
                "Mean structural gap (latest year)",
                f"{gap_pct:+.1f}%",
                help=GAP_HELP,
            )
        else:
            col2.metric("Mean structural gap (latest year)", "—")
        col3.metric("Years covered", f"{results['Year'].min()}–{results['Year'].max()}")

        if gap_pct is not None:
            st.markdown(
                f"**Takeaway:** as of {latest_year}, the pre-pandemic trend "
                f"before the shading above should never have crossed — but it "
                f"did. {describe_gap(gap_pct)} The lines above diverging inside "
                f"the shaded COVID period, and staying apart afterward, is the "
                f"visual signature of a *structural* (lasting) change rather "
                f"than a temporary blip."
            )

# ── By-MSA detail ─────────────────────────────────────────────────────
with tab_msa:
    st.markdown(
        "Pick any individual metro to see its own actual-vs-counterfactual "
        "path and year-by-year gap — useful for spot-checking whether the "
        "national pattern above holds locally, or whether a specific metro "
        "is an outlier."
    )
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
            msa_df = msa_df.copy()
            msa_df["Structural_Gap_%"] = to_pct(msa_df["Structural_Gap"]).round(1)
            latest_row = msa_df.iloc[-1]
            st.markdown(
                f"**Takeaway for {msa_choice}:** as of {int(latest_row['Year'])}, "
                f"{describe_gap(latest_row['Structural_Gap_%'], subject='this metro')}"
            )
            cols = ["Year", "Structural_Gap_%", "Market_Category"] if "Market_Category" in msa_df else ["Year", "Structural_Gap_%"]
            st.dataframe(msa_df[cols], use_container_width=True, hide_index=True)

# ── Regional comparison ────────────────────────────────────────────────
with tab_regional:
    st.markdown(
        "Same structural gap measure, grouped by U.S. Census region — "
        "shows whether the pandemic's impact on R&D real estate landed "
        "evenly across the country, or concentrated in specific areas."
    )
    results = load_csv("AvailSFTotal_Counterfactual_Results.csv")
    if results is None:
        missing_data_notice("AvailSFTotal_Counterfactual_Results.csv")
    else:
        regioned = add_region_column(results)
        if "Structural_Gap" in regioned:
            regioned = regioned.copy()
            regioned["Structural_Gap_%"] = to_pct(regioned["Structural_Gap"])
            yearly = (
                regioned.dropna(subset=["Region"])
                .groupby(["Year", "Region"])["Structural_Gap_%"]
                .mean()
                .reset_index()
            )
            fig = px.line(
                yearly,
                x="Year",
                y="Structural_Gap_%",
                color="Region",
                category_orders={"Region": REGIONS},
                labels={"Structural_Gap_%": "Mean structural gap (%)"},
                title="Mean Structural Gap by Census Region",
            )
            st.plotly_chart(fig, use_container_width=True)

            latest_year = yearly["Year"].max()
            latest_by_region = yearly[yearly["Year"] == latest_year].set_index("Region")["Structural_Gap_%"]
            if not latest_by_region.empty:
                highest_region = latest_by_region.idxmax()
                lowest_region = latest_by_region.idxmin()
                st.markdown(
                    f"**Takeaway:** as of {int(latest_year)}, **{highest_region}** "
                    f"shows the largest gap ({latest_by_region[highest_region]:+.1f}%), "
                    f"while **{lowest_region}** shows the smallest "
                    f"({latest_by_region[lowest_region]:+.1f}%) — a "
                    f"{latest_by_region[highest_region] - latest_by_region[lowest_region]:.1f} "
                    f"point spread suggests the pandemic's structural impact was "
                    f"regional, not uniform nationwide."
                )
        else:
            st.warning("`Structural_Gap` column not found in results export.")

# ── Forecast ───────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown(
        "This isn't a prediction of *how many square feet will exist in "
        "2045*. It models **how fast each MSA's current deviation from "
        "its own pre-COVID equilibrium closes** — the trajectory back "
        "toward normal, not a single future snapshot. Think of it like "
        "a thermostat: the market overshot its old equilibrium, and this "
        "traces how quickly it's expected to settle back."
    )

    national_fc = load_csv("MeanReversion_RDWeighted_National_2015_2045_v14b.csv")
    if national_fc is None:
        missing_data_notice("MeanReversion_RDWeighted_National_2015_2045_v14b.csv")
    else:
        year_col = find_col(national_fc, ["Year"])
        gap_col = find_col(
            national_fc,
            ["Deviation_Median", "Median_Deviation", "Deviation", "Gap",
             "Structural_Gap", "RDWeighted_Deviation", "Projected_Deviation"],
        )
        lower_col = find_col(national_fc, ["Lower", "CI_Lower", "P10", "Low"])
        upper_col = find_col(national_fc, ["Upper", "CI_Upper", "P90", "High"])

        if year_col is None or gap_col is None:
            st.warning(
                "Couldn't automatically identify the year/deviation columns "
                "in this file — showing the raw table instead."
            )
            st.dataframe(national_fc, use_container_width=True, hide_index=True)
        else:
            plot_df = national_fc[[year_col, gap_col]].copy()
            is_log_space = plot_df[gap_col].abs().max() < 5
            y_label = "National mean-reversion deviation (%)"
            if is_log_space:
                plot_df[gap_col] = to_pct(plot_df[gap_col])
                if lower_col:
                    plot_df["_lower"] = to_pct(national_fc[lower_col])
                if upper_col:
                    plot_df["_upper"] = to_pct(national_fc[upper_col])
            else:
                if lower_col:
                    plot_df["_lower"] = national_fc[lower_col]
                if upper_col:
                    plot_df["_upper"] = national_fc[upper_col]

            fig = go.Figure()
            if "_lower" in plot_df and "_upper" in plot_df:
                fig.add_trace(go.Scatter(
                    x=pd.concat([plot_df[year_col], plot_df[year_col][::-1]]),
                    y=pd.concat([plot_df["_upper"], plot_df["_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(31,119,180,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Uncertainty band", showlegend=True,
                ))
            fig.add_trace(go.Scatter(
                x=plot_df[year_col], y=plot_df[gap_col],
                mode="lines", name="Projected deviation",
                line=dict(color="#1f77b4", width=2.5),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray",
                          annotation_text="Equilibrium (fully normalized)")
            fig.update_layout(
                title="National R&D-Weighted Structural Gap — Projected Path to 2045",
                yaxis_title=y_label,
                xaxis_title="Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            near_zero = plot_df[plot_df[gap_col].abs() <= 5]
            start_val = plot_df[gap_col].iloc[0]
            end_val = plot_df[gap_col].iloc[-1]
            direction = "shrinking" if abs(end_val) < abs(start_val) else "widening"
            takeaway = (
                f"**Takeaway:** the model projects the national gap "
                f"{direction} from {start_val:+.1f}% toward "
                f"{end_val:+.1f}% by {int(plot_df[year_col].max())}."
            )
            if not near_zero.empty:
                closure_year = int(near_zero[year_col].min())
                takeaway += (
                    f" It's projected to settle within ±5% of equilibrium "
                    f"(effectively \"normalized\") by **{closure_year}**."
                )
            else:
                takeaway += (
                    " It is not projected to fully close within this "
                    "forecast window."
                )
            st.markdown(takeaway)

    st.divider()
    msa_fc = load_csv("MeanReversion_RDWeighted_ByMSA_v14b.csv")
    if msa_fc is not None:
        st.subheader("Per-MSA forecast")
        st.markdown("Same trajectory concept, broken out by individual metro.")
        msa_name_col = find_col(msa_fc, ["MSA_Name", "MSA"])
        if msa_name_col:
            msa_pick = st.selectbox(
                "Select an MSA", sorted(msa_fc[msa_name_col].unique()), key="forecast_msa"
            )
            st.dataframe(
                msa_fc[msa_fc[msa_name_col] == msa_pick],
                use_container_width=True, hide_index=True,
            )
        else:
            st.dataframe(msa_fc, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "**About this project:** developed as part of NSF-funded graduate "
    "research at the University of Southern Mississippi. All structural "
    "gap figures above are shown as percentage deviation from the "
    "pre-COVID counterfactual trend — see `docs/methodology.md` in the "
    "repo for the full model specification, validation approach "
    "(MSA-level LOOCV, R² ≈ 0.82), and known limitations. "
    "Data sources: CoStar, JobsEQ, NSF HERD, BEA, Census ACS/BDS, FHWA."
)
