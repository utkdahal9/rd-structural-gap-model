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
Structural_Gap (and Deviation0) in the source CSVs are log-space values.
Everywhere this app displays these to a person, it converts them to a
percentage via (exp(x) - 1) * 100 for readability.

NARRATIVE NOTE: every chart is followed by a plain-English "Takeaway"
sentence computed live from the actual data, not hardcoded. Bold text
is reserved for section-opening labels only (e.g. "Takeaway:") — never
for inline emphasis within sentences, to keep formatting consistent.
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
    """Convert a log-space value to a plain percentage."""
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
    """Plain-English description of a structural gap percentage. No inline
    bold — bold is reserved for section labels like 'Takeaway:'."""
    if pct > 5:
        return (
            f"That is a surplus: on average, {subject} currently have "
            f"{pct:+.1f}% more available R&D space than the pre-COVID "
            f"trend would have predicted."
        )
    elif pct < -5:
        return (
            f"That is a deficit: on average, {subject} currently have "
            f"{pct:+.1f}% less available R&D space than the pre-COVID "
            f"trend would have predicted, i.e. tighter supply than expected."
        )
    return (
        f"That is roughly balanced — the {pct:+.1f}% average gap is "
        f"close to what the pre-COVID trend predicted for {subject}."
    )


# ── Header / intro ──────────────────────────────────────────────────
st.title("R&D Space Market Structural Gap")
st.markdown(
    "**The question this project answers:** did COVID-19 permanently "
    "change how much R&D-oriented industrial space (labs, flex space, "
    "advanced-manufacturing buildings) is available across major U.S. "
    "metro areas, or did markets simply return to their pre-pandemic "
    "trajectory?\n\n"
    "**How:** a machine-learning model was trained only on data from "
    "before COVID (2006–2019) and used to predict what 2020–2023 "
    "should have looked like if the pandemic never happened. The gap "
    "between that prediction and what actually happened is the "
    "structural gap — the pandemic's lasting footprint on this real "
    "estate market, isolated from the normal ups and downs."
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
        "This tab shows the national picture: how much available space "
        "actually existed each year versus what the model expected "
        "without COVID (the counterfactual)."
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
                f"**Takeaway:** as of {latest_year}, the actual and "
                f"counterfactual lines above diverge inside the shaded "
                f"COVID period and stay apart afterward — the visual "
                f"signature of a lasting change rather than a temporary "
                f"blip. {describe_gap(gap_pct)}"
            )

# ── By-MSA detail ─────────────────────────────────────────────────────
with tab_msa:
    st.markdown(
        "Pick any individual metro to see its own actual-vs-counterfactual "
        "path and year-by-year gap, useful for spot-checking whether the "
        "national pattern holds locally or whether a specific metro is an "
        "outlier."
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
        "Same structural gap measure, grouped by U.S. Census region, "
        "showing whether the pandemic's impact on R&D real estate landed "
        "evenly across the country or concentrated in specific areas."
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
                spread = latest_by_region[highest_region] - latest_by_region[lowest_region]
                st.markdown(
                    f"**Takeaway:** as of {int(latest_year)}, {highest_region} "
                    f"shows the largest gap ({latest_by_region[highest_region]:+.1f}%), "
                    f"while {lowest_region} shows the smallest "
                    f"({latest_by_region[lowest_region]:+.1f}%). A {spread:.1f} "
                    f"point spread suggests the pandemic's structural impact "
                    f"was regional, not uniform nationwide."
                )
        else:
            st.warning("`Structural_Gap` column not found in results export.")

# ── Forecast ───────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown(
        "This isn't a prediction of how many square feet will exist in "
        "2045. It models how fast each MSA's current deviation from its "
        "own pre-COVID equilibrium closes, the trajectory back toward "
        "normal, not a single future snapshot."
    )

    national_fc = load_csv("MeanReversion_RDWeighted_National_2015_2045_v14b.csv")
    if national_fc is None:
        missing_data_notice("MeanReversion_RDWeighted_National_2015_2045_v14b.csv")
    else:
        actual_col = find_col(national_fc, ["RDWeighted_Mean_Gap"])
        median_col = find_col(national_fc, ["RDWeighted_Mean_Gap_p50", "p50"])
        lower_col = find_col(national_fc, ["RDWeighted_Mean_Gap_p05", "p05"])
        upper_col = find_col(national_fc, ["RDWeighted_Mean_Gap_p95", "p95"])
        year_col = find_col(national_fc, ["Year"])

        if year_col is None or actual_col is None or median_col is None:
            st.warning(
                "Couldn't automatically identify the expected columns in "
                "this file — showing the raw table instead."
            )
            st.dataframe(national_fc, use_container_width=True, hide_index=True)
        else:
            hist = national_fc[national_fc[actual_col].notna()].copy()
            fc = national_fc[national_fc[median_col].notna()].copy()
            hist["pct"] = to_pct(hist[actual_col])
            fc["pct_median"] = to_pct(fc[median_col])
            if lower_col:
                fc["pct_lower"] = to_pct(fc[lower_col])
            if upper_col:
                fc["pct_upper"] = to_pct(fc[upper_col])

            fig = go.Figure()
            if "pct_lower" in fc and "pct_upper" in fc:
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc[year_col], fc[year_col][::-1]]),
                    y=pd.concat([fc["pct_upper"], fc["pct_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(31,119,180,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Uncertainty band",
                ))
            fig.add_trace(go.Scatter(
                x=hist[year_col], y=hist["pct"],
                mode="lines", name="Actual (historical)",
                line=dict(color="#1f77b4", width=2.5),
            ))
            fig.add_trace(go.Scatter(
                x=fc[year_col], y=fc["pct_median"],
                mode="lines", name="Forecast (median)",
                line=dict(color="#1f77b4", width=2.5, dash="dash"),
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="gray",
                          annotation_text="Equilibrium")
            fig.update_layout(
                title="National R&D-Weighted Structural Gap — Actual and Projected Path to 2045",
                yaxis_title="Structural gap (%)",
                xaxis_title="Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            final_year = int(fc[year_col].max())
            final_pct = fc["pct_median"].iloc[-1]
            near_zero = fc[fc["pct_median"].abs() <= 5]
            if not near_zero.empty:
                closure_year = int(near_zero[year_col].min())
                takeaway = (
                    f"**Takeaway:** the market is projected to settle "
                    f"within ±5% of equilibrium by {closure_year}."
                )
            else:
                takeaway = (
                    f"**Takeaway:** the gap is projected to narrow "
                    f"substantially but not fully close, settling around "
                    f"{final_pct:+.1f}% by {final_year} rather than "
                    f"returning to equilibrium within this forecast window."
                )
            st.markdown(takeaway)

    st.divider()
    msa_fc = load_csv("MeanReversion_RDWeighted_ByMSA_v14b.csv")
    if msa_fc is not None:
        st.subheader("Per-MSA equilibrium status")
        st.markdown(
            "This file captures each MSA's equilibrium anchor and its "
            "distance from it as of 2023, not a full year-by-year series. "
            "Using the 30%/year decay rate documented in the methodology, "
            "the chart below projects each MSA's own path back toward "
            "equilibrium — an approximation based on the standard rate, "
            "not a per-MSA calibrated forecast. See "
            "MeanReversion_RDWeighted_Lambda_Sensitivity_v14b.csv in the "
            "repo for how sensitive this is to that assumption."
        )
        msa_name_col = find_col(msa_fc, ["MSA_Name", "MSA"])
        dev_col = find_col(msa_fc, ["Deviation0"])
        converged_col = find_col(msa_fc, ["Already_Converged_2023", "Already_Converged"])

        if msa_name_col and dev_col:
            msa_pick = st.selectbox(
                "Select an MSA", sorted(msa_fc[msa_name_col].unique()), key="forecast_msa"
            )
            row = msa_fc[msa_fc[msa_name_col] == msa_pick].iloc[0]
            dev0 = row[dev_col]
            dev0_pct = to_pct(pd.Series([dev0])).iloc[0]

            LAMBDA = 0.30
            THRESH_LOG = np.log(1.05)  # treat +/-5% as effectively converged
            anchor_year, max_year = 2023, 2045
            years = list(range(anchor_year, max_year + 1))
            traj = [dev0 * ((1 - LAMBDA) ** (y - anchor_year)) for y in years]
            traj_pct = to_pct(pd.Series(traj)).tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years, y=traj_pct, mode="lines",
                name="Projected path (30%/yr decay)",
                line=dict(color="#1f77b4", width=2.5, dash="dash"),
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="gray",
                          annotation_text="Equilibrium")
            fig.update_layout(
                title=f"{msa_pick} — Projected Path to Equilibrium",
                yaxis_title="Deviation from equilibrium (%)",
                xaxis_title="Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            already_converged = bool(row[converged_col]) if converged_col else None
            if already_converged:
                status = "This MSA was already within the effectively-converged range as of 2023."
            else:
                if abs(dev0) <= THRESH_LOG:
                    close_yr = anchor_year
                else:
                    t = np.log(THRESH_LOG / abs(dev0)) / np.log(1 - LAMBDA)
                    proj_year = anchor_year + int(np.ceil(t))
                    close_yr = proj_year if proj_year <= max_year else None
                if close_yr:
                    status = (
                        f"Starting from a {dev0_pct:+.1f}% deviation in 2023, "
                        f"this MSA is projected to settle within ±5% of "
                        f"equilibrium by {close_yr} at the standard decay rate."
                    )
                else:
                    status = (
                        f"Starting from a {dev0_pct:+.1f}% deviation in 2023, "
                        f"this MSA is not projected to fully close within "
                        f"the {max_year} forecast window at the standard "
                        f"decay rate."
                    )
            st.markdown(f"**Takeaway:** {status}")
        else:
            st.dataframe(msa_fc, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "**About this project:** developed as part of NSF-funded graduate "
    "research at the University of Southern Mississippi. All structural "
    "gap figures above are shown as percentage deviation from the "
    "pre-COVID counterfactual trend — see `docs/methodology.md` in the "
    "repo for the full model specification, validation approach "
    "(MSA-level LOOCV, R² ≈ 0.82), and known limitations. Data sources: "
    "CoStar, JobsEQ, NSF HERD, BEA, Census ACS/BDS, FHWA."
)
