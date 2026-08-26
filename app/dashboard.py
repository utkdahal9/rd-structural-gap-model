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

EQUILIBRIUM NOTE: the mean-reversion forecast does not project the gap
back to 0%. Each MSA reverts toward its OWN long-run equilibrium target
(Residual_Equilibrium + Bias_2023 in the source data), which is usually
NOT zero. The national number is the R&D-weighted average of all 100
of those individual targets. This dashboard computes and displays the
real target explicitly, rather than assuming it's zero.

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
        "2045. Each MSA's forecast reverts toward its OWN long-run "
        "equilibrium level, not toward zero, so the national gap "
        "settling above 0% doesn't mean the market never normalizes — "
        "it means the model's estimate of \"normal\" for this metric "
        "isn't zero to begin with."
    )

    msa_fc = load_csv("MeanReversion_RDWeighted_ByMSA_v14b.csv")
    lam_fc = load_csv("MeanReversion_RDWeighted_Lambda_Sensitivity_v14b.csv")
    results_hist = load_csv("AvailSFTotal_Counterfactual_Results.csv")

    national_target_pct = None
    if msa_fc is not None:
        eq_col = find_col(msa_fc, ["Residual_Equilibrium"])
        bias_col = find_col(msa_fc, ["Bias_2023"])
        weight_col = find_col(msa_fc, ["RD_Weight_2023"])
        if eq_col and bias_col and weight_col:
            eq_forecast_all = msa_fc[eq_col] + msa_fc[bias_col]
            national_target_log = np.average(eq_forecast_all, weights=msa_fc[weight_col])
            national_target_pct = to_pct(pd.Series([national_target_log])).iloc[0]

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
            if national_target_pct is not None:
                fig.add_hline(
                    y=national_target_pct, line_dash="dot", line_color="gray",
                    annotation_text=f"R&D-weighted long-run target ({national_target_pct:+.1f}%)",
                )
            fig.update_layout(
                title="National R&D-Weighted Structural Gap — Actual and Projected Path to 2045",
                yaxis_title="Structural gap (%)",
                xaxis_title="Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            canonical_year = None
            if lam_fc is not None:
                is_mean_col = find_col(lam_fc, ["Is_Assumed_Mean"])
                lam_year_col = find_col(lam_fc, ["Year"])
                if is_mean_col and lam_year_col:
                    mean_row = lam_fc[lam_fc[is_mean_col] == True]
                    if not mean_row.empty:
                        canonical_year = int(mean_row.iloc[0][lam_year_col])

            if national_target_pct is not None and canonical_year:
                takeaway = (
                    f"**Takeaway:** the market isn't projected to return to "
                    f"0% — it's projected to settle near its R&D-weighted "
                    f"long-run target of {national_target_pct:+.1f}%, which "
                    f"reflects a lasting (not fully reversing) shift. At "
                    f"the model's assumed reversion speed, that settling "
                    f"is essentially complete by **{canonical_year}**."
                )
            else:
                final_pct = fc["pct_median"].iloc[-1]
                final_year = int(fc[year_col].max())
                takeaway = (
                    f"**Takeaway:** the gap is projected to narrow toward "
                    f"{final_pct:+.1f}% by {final_year}, its apparent "
                    f"long-run level, rather than returning to 0%."
                )
            st.markdown(takeaway)

    st.divider()
    if msa_fc is not None:
        st.subheader("Per-MSA equilibrium trajectory")

        canonical_year = None
        if lam_fc is not None:
            is_mean_col = find_col(lam_fc, ["Is_Assumed_Mean"])
            lam_year_col = find_col(lam_fc, ["Year"])
            if is_mean_col and lam_year_col:
                mean_row = lam_fc[lam_fc[is_mean_col] == True]
                if not mean_row.empty:
                    canonical_year = int(mean_row.iloc[0][lam_year_col])

        if canonical_year:
            st.markdown(
                f"At the model's assumed reversion speed (λ=30%/year), "
                f"every MSA closes 95% of its own remaining deviation by "
                f"**{canonical_year}** — this timing doesn't depend on how "
                f"large a given MSA's current gap is, since the same share "
                f"closes each year. What differs by MSA is where it lands: "
                f"each metro reverts toward its own historical equilibrium, "
                f"which is rarely exactly zero."
            )

        msa_name_col = find_col(msa_fc, ["MSA_Name", "MSA"])
        dev_col = find_col(msa_fc, ["Deviation0"])
        eq_col = find_col(msa_fc, ["Residual_Equilibrium"])
        bias_col = find_col(msa_fc, ["Bias_2023"])
        converged_col = find_col(msa_fc, ["Already_Converged_2023"])

        if msa_name_col and dev_col and eq_col and bias_col:
            msa_pick = st.selectbox(
                "Select an MSA", sorted(msa_fc[msa_name_col].unique()), key="forecast_msa"
            )
            row = msa_fc[msa_fc[msa_name_col] == msa_pick].iloc[0]
            dev0 = row[dev_col]
            eq_fc = row[eq_col] + row[bias_col]
            eq_fc_pct = to_pct(pd.Series([eq_fc])).iloc[0]

            years = list(range(2023, 2046))

            def traj_pct_for(lam):
                vals = [eq_fc + dev0 * ((1 - lam) ** (y - 2023)) for y in years]
                return to_pct(pd.Series(vals)).tolist()

            median_traj = traj_pct_for(0.30)
            slow_traj = traj_pct_for(0.05)
            fast_traj = traj_pct_for(0.60)
            band_lo = [min(a, b) for a, b in zip(slow_traj, fast_traj)]
            band_hi = [max(a, b) for a, b in zip(slow_traj, fast_traj)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years + years[::-1], y=band_hi + band_lo[::-1],
                fill="toself", fillcolor="rgba(31,119,180,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Range across plausible reversion speeds",
            ))

            if results_hist is not None and "Structural_Gap" in results_hist:
                hmsa = results_hist[results_hist["MSA_Name"] == msa_pick].sort_values("Year")
                hmsa = hmsa[hmsa["Year"] <= 2023]
                if not hmsa.empty:
                    fig.add_trace(go.Scatter(
                        x=hmsa["Year"], y=to_pct(hmsa["Structural_Gap"]),
                        mode="lines", name="Actual (historical)",
                        line=dict(color="#1f77b4", width=2.5),
                    ))

            fig.add_trace(go.Scatter(
                x=years, y=median_traj, mode="lines",
                name="Forecast (median reversion speed)",
                line=dict(color="#1f77b4", width=2.5, dash="dash"),
            ))
            fig.add_hline(
                y=eq_fc_pct, line_dash="dot", line_color="gray",
                annotation_text=f"This MSA's own long-run target ({eq_fc_pct:+.1f}%)",
            )
            fig.update_layout(
                title=f"{msa_pick} — Structural Gap, Actual and Projected Path",
                yaxis_title="Structural gap (%)",
                xaxis_title="Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            already_converged = bool(row[converged_col]) if converged_col else None
            dev0_pct = to_pct(pd.Series([dev0])).iloc[0]
            if already_converged:
                status = "this MSA was already close to its own equilibrium as of 2023"
            else:
                status = (
                    f"as of 2023 this MSA sat {dev0_pct:+.1f} percentage "
                    f"points from its own long-run target"
                )
            st.markdown(
                f"**Takeaway:** {status}; that target is {eq_fc_pct:+.1f}%, "
                f"not 0%, reflecting a level specific to this metro rather "
                f"than full reversion to the no-COVID counterfactual."
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
    "(MSA-level LOOCV, R² ≈ 0.82), and known limitations. Data sources: "
    "CoStar, JobsEQ, NSF HERD, BEA, Census ACS/BDS, FHWA."
)
