"""
R&D Space Market Structural Gap — Interactive Dashboard
=========================================================
Streamlit app for exploring the COVID-19 structural-gap counterfactual
model's results: national trends, by-MSA detail, regional comparisons,
SHAP feature importance, named case studies, and the 2045 mean-reversion
forecast.

Run with:  streamlit run app/dashboard.py

Reads CSV exports produced by the notebooks in notebooks/ (run 01-06
first). Each section degrades gracefully with a clear message if its
source file isn't present yet, rather than crashing the whole app.

UNITS NOTE: the underlying model predicts log(Available_SF_Total), so
Structural_Gap (and Deviation0) in the source CSVs are log-space values,
converted to a percentage via (exp(x) - 1) * 100 for readability. The
R&D-weighted gap figures used for regional ranking and case-study
summaries (Mean_Gap_RDWeighted_*, R&D_Weighted_Gap_SF) are a DIFFERENT
metric — raw square footage, not log-space — never run through that
conversion. They're ABSOLUTE figures that scale with market size, so a
large metro can rank higher on this measure than a smaller metro with
a bigger PERCENTAGE gap. Both are shown, clearly labeled, rather than
picking one. The "Highest/Lowest in Region" ranking language is used
deliberately instead of "Surplus/Deficit" for this metric, since that
wording is reserved for the model's own Market_Category classification
and the two must not be conflated.

MARKET_CATEGORY NOTE: "Structurally Balanced" / "Moderate Surplus" /
etc. are NOT based on a fixed percentage threshold. They come from the
model's own classification: a z-score of the gap against LOOCV_SIGMA
(the model's typical prediction noise, dynamically loaded from
SizeBias_Correction_Verification.csv). That means "Structurally
Balanced" spans roughly -16% to +19%, not some narrow band around
zero. This dashboard always uses the model's own category label rather
than recomputing a different one, and explains the real thresholds
explicitly wherever the category is shown.

NARRATIVE NOTE: every chart is followed by a plain-English "Takeaway"
sentence computed live from the actual data. Bold text is reserved for
section-opening labels only (e.g. "Takeaway:") — never for inline
emphasis within sentences.
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
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
IMAGE_MAX_WIDTH = 800  # cap static figure width so they don't stretch/blur full-page

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


def fmt_sf(x: float) -> str:
    """Format a raw square-footage figure with a sign and thousands separators."""
    return f"{x:+,.0f} SF"


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


def latest_pct_lookup(results_df: pd.DataFrame) -> pd.Series | None:
    """Series of each MSA's latest-year Structural_Gap_%, indexed by MSA_Name."""
    if results_df is None or "Structural_Gap" not in results_df:
        return None
    recent = results_df[results_df["Year"] == results_df["Year"].max()]
    return to_pct(recent.set_index("MSA_Name")["Structural_Gap"])


@st.cache_data
def category_thresholds_pct():
    """Real Market_Category boundaries in percentage terms, computed from
    the model's own LOOCV_SIGMA. Returns None if the file isn't available."""
    sb = load_csv("SizeBias_Correction_Verification.csv")
    if sb is None:
        return None
    metric_col = find_col(sb, ["Metric"])
    value_col = find_col(sb, ["Value"])
    if not metric_col or not value_col:
        return None
    lookup = sb.set_index(metric_col)[value_col]
    sigma = lookup.get("LOOCV_SIGMA_Corrected")
    if sigma is None:
        return None
    sig_hi = (np.exp(1.5 * sigma) - 1) * 100
    sig_lo = (np.exp(-1.5 * sigma) - 1) * 100
    mod_hi = (np.exp(0.5 * sigma) - 1) * 100
    mod_lo = (np.exp(-0.5 * sigma) - 1) * 100
    return sig_hi, mod_hi, mod_lo, sig_lo


def category_explainer_text() -> str:
    bounds = category_thresholds_pct()
    if bounds is None:
        return (
            "Market categories (Significant/Moderate Surplus or Deficit, "
            "Structurally Balanced) are based on a z-score of the gap "
            "against the model's own prediction noise (LOOCV_SIGMA), not "
            "a fixed percentage band — so \"Structurally Balanced\" can "
            "span a wider range than ±5%."
        )
    sig_hi, mod_hi, mod_lo, sig_lo = bounds
    return (
        f"Market categories aren't based on a fixed percentage band — "
        f"they're a z-score of the gap against the model's own typical "
        f"prediction noise (LOOCV_SIGMA). In percentage terms, that "
        f"currently works out to roughly: **Significant Surplus** above "
        f"{sig_hi:+.0f}%, **Moderate Surplus** {mod_hi:+.0f}% to "
        f"{sig_hi:+.0f}%, **Structurally Balanced** {mod_lo:+.0f}% to "
        f"{mod_hi:+.0f}%, **Moderate Deficit** {sig_lo:+.0f}% to "
        f"{mod_lo:+.0f}%, and **Significant Deficit** below {sig_lo:+.0f}%. "
        f"That's why a double-digit negative number can still read as "
        f"\"balanced\" — the bands are wider than intuition suggests. "
        f"This is a separate system from the \"Highest/Lowest in Region\" "
        f"ranking shown elsewhere, which is based on absolute square "
        f"footage, not this statistical classification."
    )


CASE_STUDY_IMAGES = {
    "Seattle-Tacoma-Bellevue, WA": "casestudy_dashboard_Seattle_Tacoma_Bellevue_WA.png",
    "Houston-Pasadena-The Woodlands, TX": "casestudy_dashboard_Houston_Pasadena_The_Woodlands_TX.png",
    "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD": "casestudy_dashboard_Philadelphia_Camden_Wilmington_PA_NJ_DE_MD.png",
    "Dallas-Fort Worth-Arlington, TX": "casestudy_dashboard_Dallas_Fort_Worth_Arlington_TX.png",
    "Boston-Cambridge-Newton, MA-NH": "casestudy_dashboard_Boston_Cambridge_Newton_MA_NH.png",
    "New York-Newark-Jersey City, NY-NJ": "casestudy_dashboard_New_York_Newark_Jersey_City_NY_NJ.png",
    "San Francisco-Oakland-Fremont, CA": "casestudy_dashboard_San_Francisco_Oakland_Fremont_CA.png",
    "San Jose-Sunnyvale-Santa Clara, CA": "casestudy_dashboard_San_Jose_Sunnyvale_Santa_Clara_CA.png",
}

RANK_VS_PCT_NOTE = (
    "Rank is based on absolute R&D-weighted square footage, which "
    "scales with market size — a large metro can rank higher (or "
    "lower) than a smaller one even with a smaller percentage gap, "
    "because the same percentage move represents far more square feet "
    "in a bigger market. This ranking and the percentage figure answer "
    "different questions and can point in different directions; "
    "neither overrides the other."
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

(
    tab_overview, tab_msa, tab_regional, tab_shap,
    tab_casestudies, tab_forecast,
) = st.tabs(
    ["National Overview", "By MSA", "Regional", "SHAP Feature Importance",
     "Case Studies", "2045 Forecast"]
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
                f"blip. On average, that's a "
                f"{'surplus' if gap_pct > 0 else 'deficit'} of "
                f"{gap_pct:+.1f}% relative to the pre-COVID trend "
                f"(this is a simple average across 100 MSAs, not a "
                f"single market category — see the By MSA tab for how "
                f"individual metros are classified)."
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
            has_category = "Market_Category" in msa_df
            if has_category:
                st.markdown(
                    f"**Takeaway for {msa_choice}:** as of "
                    f"{int(latest_row['Year'])}, this metro's gap is "
                    f"classified **{latest_row['Market_Category']}** "
                    f"({latest_row['Structural_Gap_%']:+.1f}% relative to "
                    f"the counterfactual)."
                )
                with st.expander("What do these categories mean?"):
                    st.markdown(category_explainer_text())
            cols = ["Year", "Structural_Gap_%", "Market_Category"] if has_category else ["Year", "Structural_Gap_%"]
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
                    f"shows the largest average gap "
                    f"({latest_by_region[highest_region]:+.1f}%), while "
                    f"{lowest_region} shows the smallest "
                    f"({latest_by_region[lowest_region]:+.1f}%). A {spread:.1f} "
                    f"point spread suggests the pandemic's structural impact "
                    f"was regional, not uniform nationwide."
                )
        else:
            st.warning("`Structural_Gap` column not found in results export.")

    st.divider()
    st.subheader("Highest and lowest R&D-weighted markets by region")
    st.markdown(
        "The five highest and five lowest R&D-weighted gaps, in absolute "
        "square feet, within each Census region. This is a pure ranking "
        "within the region, not a statistical classification — a metro "
        "can land in the \"Lowest\" group even with a positive percentage "
        "gap, if its absolute weighted square footage is still smaller "
        "than its regional peers'. New York, for example, shows a mildly "
        "positive percentage gap but still ranks among the Northeast's "
        "lowest absolute R&D-weighted values, because that measure "
        "reflects market size as well as direction."
    )
    st.markdown(RANK_VS_PCT_NOTE)
    top5 = load_csv("Regional_Top5_RDWeighted_v14b.csv")
    if top5 is None:
        missing_data_notice("Regional_Top5_RDWeighted_v14b.csv")
    else:
        pct_lookup = latest_pct_lookup(results)
        rank_label_map = {
            "Top 5 Surplus": "Highest in Region (R&D-weighted SF)",
            "Top 5 Deficit": "Lowest in Region (R&D-weighted SF)",
        }
        region_pick = st.selectbox("Select a region", REGIONS, key="top5_region")
        region_df = top5[top5["Region"] == region_pick].copy()
        rank_type_col = find_col(region_df, ["Rank_Type"])
        if rank_type_col:
            region_df["Rank"] = region_df[rank_type_col].map(rank_label_map).fillna(region_df[rank_type_col])
        if pct_lookup is not None:
            region_df["Structural_Gap_%"] = region_df["MSA_Name"].map(pct_lookup).round(1)
            show_cols = [c for c in ["MSA_Name", "Structural_Gap_%", "Rank"] if c in region_df.columns]
            st.dataframe(region_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(region_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Small-base, fast-growing R&D markets")
    st.markdown(
        "These 20 metros have the smallest R&D real estate base "
        "(2015–2018 anchor period) in the panel — markets that may be "
        "emerging R&D hubs rather than established ones."
    )
    smallbase = load_csv("Top20_SmallBase_Growth_LQTrend_v14b.csv")
    if smallbase is None:
        missing_data_notice("Top20_SmallBase_Growth_LQTrend_v14b.csv")
    else:
        cagr_col = find_col(smallbase, ["CAGR"])
        slope_col = find_col(smallbase, ["LQ_Slope"])
        rising_col = find_col(smallbase, ["Rising_Concentration"])
        display = smallbase.copy()
        if cagr_col:
            display["CAGR (%)"] = (display[cagr_col] * 100).round(1)
        if slope_col and rising_col:
            display["LQ Trend"] = display.apply(
                lambda r: f"{'Rising' if r[rising_col] else 'Declining'} ({r[slope_col]:+.4f}/yr)",
                axis=1,
            )
        show_cols = [c for c in ["MSA_Name", "CAGR (%)", "LQ Trend"] if c in display.columns]
        st.dataframe(display[show_cols], use_container_width=True, hide_index=True)
        st.markdown(
            "**CAGR** is the compound annual growth rate of *available* "
            "(i.e. unleased) R&D space specifically, from a 2015–2018 "
            "anchor to a 2021–2023 recent period — not overall real "
            "estate stock or demand. Negative values mean the pool of "
            "available space has been shrinking, which is consistent "
            "with strong absorption (space getting leased up faster "
            "than it's added) even in a market whose overall industry "
            "is growing. \"Fast-growing\" in this tab's title refers to "
            "growth relative to the rest of the 100-MSA panel, not to "
            "positive CAGR in absolute terms — most of the panel's CAGR "
            "is negative too. **LQ Trend** tracks the metro's "
            "advanced-industry employment location quotient: rising "
            "means R&D-relevant employment concentration is increasing "
            "year over year, declining means it's easing."
        )

# ── SHAP feature importance ─────────────────────────────────────────────
with tab_shap:
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values quantify how much "
        "each feature actually pushes the model's individual predictions "
        "up or down, rather than just measuring correlation with the "
        "outcome."
    )

    st.subheader("Global importance")
    shap_df = load_csv("AvailSFTotal_SHAP_importance.csv")
    if shap_df is None:
        missing_data_notice("AvailSFTotal_SHAP_importance.csv")
    else:
        feat_col = find_col(shap_df, ["Feature"])
        shap_col = find_col(shap_df, ["Mean_SHAP"])
        if feat_col and shap_col:
            top_shap = shap_df.nlargest(15, shap_col).sort_values(shap_col)
            fig = px.bar(
                top_shap, x=shap_col, y=feat_col, orientation="h",
                title="Top 15 Features by SHAP Importance",
                labels={shap_col: "Mean |SHAP value|", feat_col: ""},
            )
            st.plotly_chart(fig, use_container_width=True)
            top_feature = shap_df.nlargest(1, shap_col).iloc[0][feat_col]
            st.markdown(
                f"This ranks features by their average impact across all "
                f"predictions — {top_feature} moves the model's output "
                f"more, on average, than any other feature. A longer bar "
                f"means that feature matters more to the model overall, "
                f"but says nothing about direction (whether it pushes "
                f"predictions up or down) or about any single prediction "
                f"specifically — that's what the beeswarm plot below adds."
            )

    st.divider()
    st.subheader("Per-observation detail")
    beeswarm_path = FIGURES_DIR / "lag_counterfactual_shap_beeswarm_v14b.png"
    if beeswarm_path.exists():
        st.image(str(beeswarm_path), width=IMAGE_MAX_WIDTH)
        st.markdown(
            "Each dot is one MSA-year observation. A feature's row shows "
            "every observation's individual SHAP value, not just the "
            "average — so you can see both how much a feature matters "
            "and in which direction, and whether that effect is "
            "consistent across observations or varies a lot (e.g. "
            "helping some metros' predictions while hurting others')."
        )
    else:
        st.info(
            "`lag_counterfactual_shap_beeswarm_v14b.png` not found in "
            "`figures/`."
        )

# ── Case studies ─────────────────────────────────────────────────────
with tab_casestudies:
    st.markdown(
        "Eight named metros were profiled in depth. Pick one to see its "
        "category, ranking, R&D-employment concentration trend, and "
        "equilibrium status side by side."
    )
    st.markdown(RANK_VS_PCT_NOTE)
    profiles = load_csv("Market_CaseStudies_Combined_Profile.csv")
    results_hist = load_csv("AvailSFTotal_Counterfactual_Results.csv")
    if profiles is None:
        missing_data_notice("Market_CaseStudies_Combined_Profile.csv")
    else:
        msa_col = find_col(profiles, ["MSA_Name"])
        msa_pick = st.selectbox(
            "Select a case-study MSA", sorted(profiles[msa_col].unique()), key="casestudy_msa"
        )
        row = profiles[profiles[msa_col] == msa_pick].iloc[0]

        cat_col = find_col(profiles, ["Typical_Market_Category"])
        rank_col = find_col(profiles, ["Rank_RDWeighted"])
        gap_sf_col = find_col(profiles, ["Mean_Gap_RDWeighted_2020_2023"])
        conv_col = find_col(profiles, ["Already_Converged_2023"])

        pct_lookup = latest_pct_lookup(results_hist)

        c1, c2, c3 = st.columns(3)
        if cat_col:
            c1.metric("Market category", row[cat_col])
        if pct_lookup is not None and msa_pick in pct_lookup.index:
            c2.metric("Structural gap (latest year)", f"{pct_lookup[msa_pick]:+.1f}%")
        if rank_col:
            c3.metric("Rank (R&D-weighted, absolute SF)", f"#{int(row[rank_col])} of 100")

        if cat_col:
            with st.expander("What do these categories mean?"):
                st.markdown(category_explainer_text())

        if gap_sf_col:
            st.caption(
                f"R&D-weighted mean gap, 2020–2023 (raw square footage, "
                f"drives the rank above — a different metric from the "
                f"percentage shown): {fmt_sf(row[gap_sf_col])}"
            )

        if msa_pick in CASE_STUDY_IMAGES:
            img_path = FIGURES_DIR / CASE_STUDY_IMAGES[msa_pick]
            if img_path.exists():
                st.image(str(img_path), width=IMAGE_MAX_WIDTH)
            else:
                st.info(f"Figure not found at `figures/{CASE_STUDY_IMAGES[msa_pick]}`.")
        else:
            st.caption(
                "No pre-built multi-panel figure exists for this MSA — "
                "profile data is shown below regardless."
            )

        if conv_col is not None:
            status = "already at" if bool(row[conv_col]) else "still converging toward"
            st.markdown(
                f"**Takeaway:** {msa_pick} is {status} its own long-run "
                f"equilibrium as of 2023."
            )

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
