import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoChoice India — Data Intelligence",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{font-size:1.9rem;font-weight:700;color:#1a5276;
  padding-bottom:.3rem;border-bottom:3px solid #1a5276;margin-bottom:1rem}
.sub-header{font-size:1.05rem;font-weight:600;color:#2e86c1;margin:1rem 0 .4rem}
.insight-box{background:#eafaf1;border-left:4px solid #1e8449;
  padding:.75rem 1rem;border-radius:6px;margin:.5rem 0;font-size:.9rem}
.warning-box{background:#fef9e7;border-left:4px solid #d4ac0d;
  padding:.75rem 1rem;border-radius:6px;margin:.5rem 0;font-size:.9rem}
.action-card{background:#f4f6f7;border-radius:10px;padding:.9rem;
  border:1px solid #aed6f1;margin:.35rem 0;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PIPE_COLS = [
    "Q13_products_interested","Q14_clothing_types","Q15_cookware_types",
    "Q16_colour_preference","Q17_purchase_occasions",
    "Q19_top3_purchase_factors","Q22_social_media_channels"
]
LIKERT_COLS = [
    "Q10_eco_awareness_1to5","Q21_bundle_deal_appeal_1to5","Q23_eco_label_influence_1to5"
]
ORDINAL_MAPS = {
    "Q1_age_group":{"Below 18":0,"18-25":1,"26-35":2,"36-50":3,"Above 50":4},
    "Q4_city_tier":{"Metro":3,"Tier-2 city":2,"Tier-3 town":1,"Rural / Village":0},
    "Q5_education":{"Up to 10th":0,"12th / Diploma":1,"Undergraduate":2,"Postgraduate or above":3},
    "Q7_income_band":{"Below 15000":0,"15000-30000":1,"30001-60000":2,"60001-100000":3,"Above 100000":4},
    "Q8_monthly_spend_band":{"Below 500":0,"500-2000":1,"2001-5000":2,"5001-10000":3,"Above 10000":4},
    "Q9_willingness_to_pay_premium":{
        "Only if cheaper":0,"Same price":1,"Up to 10% more":2,"Up to 25% more":3,"More than 25% premium":4},
    "Q11_reduce_plastic_freq":{"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always / It's a habit":4},
    "Q12_prior_eco_purchase":{
        "No and not interested":0,"No but interested":1,"Yes once or twice":2,"Yes regularly":3},
    "Q20_shopping_frequency":{
        "Rarely / Only when needed":0,"Once every 3 months":1,"Once a month":2,
        "2-3 times a month":3,"Weekly":4},
    "Q_waste_segregation":{"No":0,"Sometimes":1,"Yes":2},
    "Q_hh_type":{"Nuclear":0,"Joint family":1},
}
NOMINAL_COLS = [
    "Q2_gender","Q3_state","Q6_occupation","Q18_primary_channel",
    "Q24_purchase_barrier","Q_purchase_decision_maker","persona_label"
]

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    for col in LIKERT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    for col in ["Q_children_under15","Q7_income_midpoint_INR","Q8_spend_midpoint_INR","Q_hh_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    for col in PIPE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col+"_enc"] = df[col].map(mapping).fillna(0).astype(int)

    for col in NOMINAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            df = pd.concat([df, dummies], axis=1)

    for col in PIPE_COLS:
        if col not in df.columns:
            continue
        all_items: set = set()
        df[col].dropna().apply(
            lambda x: all_items.update([i.strip() for i in str(x).split("|")])
        )
        all_items -= {"Unknown","None",""}
        for item in all_items:
            safe = (item.replace(" ","_").replace("/","_").replace("(","")
                       .replace(")","").replace(".","").replace(",","")
                       .replace("'","").replace("-","_").replace("&","and"))
            cname = col[:15]+"_"+safe[:25]
            df[cname] = df[col].apply(
                lambda x: 1 if isinstance(x, str) and item in x else 0
            )
    return df


@st.cache_data(show_spinner=False)
def get_feature_matrix(df: pd.DataFrame, target_col: str = "Q25_purchase_binary"):
    exclude = {
        "respondent_id","Q25_purchase_likelihood","Q25_purchase_binary",
        "Q7_income_midpoint_INR","Q8_spend_midpoint_INR","cluster"
    }
    exclude |= set(PIPE_COLS) | set(NOMINAL_COLS) | set(ORDINAL_MAPS.keys())

    feat_cols = [
        c for c in df.columns
        if c not in exclude
        and c != target_col
        and df[c].dtype in [np.int64, np.float64, int, float]
    ]
    X = df[feat_cols].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    y = df[target_col].fillna(0).astype(int) if target_col in df.columns else None
    return X_imp, y, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ♻ EcoChoice India")
    st.markdown("### Data Intelligence Suite")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload survey CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_raw):,} rows")
    else:
        try:
            df_raw = pd.read_csv("EcoChoice_India_Survey_2000.csv")
            st.info("Using default dataset (2,000 respondents)")
        except FileNotFoundError:
            st.error("Upload the survey CSV to begin.")
            st.stop()

    st.markdown("---")
    st.markdown("### Global Filters")
    city_opts = ["All"] + sorted(df_raw["Q4_city_tier"].dropna().unique().tolist())
    sel_city  = st.selectbox("City Tier", city_opts)
    all_income = df_raw["Q7_income_band"].dropna().unique().tolist()
    sel_income = st.multiselect("Income Band", all_income, default=all_income)
    all_persona = df_raw["persona_label"].dropna().unique().tolist()
    sel_persona = st.multiselect("Persona", all_persona, default=all_persona)
    st.markdown("---")
    st.caption("Founder's Data Intelligence Dashboard")

# Apply filters
df_f = df_raw.copy()
if sel_city != "All":
    df_f = df_f[df_f["Q4_city_tier"] == sel_city]
if sel_income:
    df_f = df_f[df_f["Q7_income_band"].isin(sel_income)]
if sel_persona:
    df_f = df_f[df_f["persona_label"].isin(sel_persona)]

if len(df_f) < 50:
    st.warning("Too few rows after filtering. Broaden your filters.")
    st.stop()

df_p = load_and_preprocess(df_f)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive",
    "🔍 Diagnostic",
    "🤖 Classification",
    "🧩 Clustering",
    "📐 Regression & Association"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="main-header">📊 Descriptive Overview</div>', unsafe_allow_html=True)
    st.caption(f"Showing **{len(df_f):,}** respondents after filters")

    n    = len(df_f)
    pos  = (df_f["Q25_purchase_binary"]==1).mean()*100
    awar = df_f["Q10_eco_awareness_1to5"].mean()
    spnd = df_f["Q8_spend_midpoint_INR"].mean()
    eco  = df_f["Q12_prior_eco_purchase"].isin(["Yes regularly","Yes once or twice"]).mean()*100

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Respondents", f"{n:,}")
    k2.metric("Likely to Buy", f"{pos:.1f}%")
    k3.metric("Avg Eco Awareness", f"{awar:.2f}/5")
    k4.metric("Avg Monthly Spend", f"₹{spnd:,.0f}")
    k5.metric("Prior Eco Buyers", f"{eco:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sub-header">Age Group Distribution</div>', unsafe_allow_html=True)
        ao = ["Below 18","18-25","26-35","36-50","Above 50"]
        ac = df_f["Q1_age_group"].value_counts().reindex(ao).dropna()
        fig = px.bar(x=ac.index, y=ac.values, text=ac.values,
                     color=ac.values, color_continuous_scale="Blues",
                     labels={"x":"Age Group","y":"Count"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,coloraxis_showscale=False,height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sub-header">City Tier Split</div>', unsafe_allow_html=True)
        cc = df_f["Q4_city_tier"].value_counts()
        fig = px.pie(values=cc.values, names=cc.index,
                     color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.42)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sub-header">Income Band</div>', unsafe_allow_html=True)
        io = ["Below 15000","15000-30000","30001-60000","60001-100000","Above 100000"]
        ic = df_f["Q7_income_band"].value_counts().reindex(io).dropna()
        fig = px.bar(x=ic.index, y=ic.values, text=ic.values,
                     color=ic.values, color_continuous_scale="Teal",
                     labels={"x":"Income (₹/month)","y":"Count"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,coloraxis_showscale=False,height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="sub-header">Persona Distribution</div>', unsafe_allow_html=True)
        pc = df_f["persona_label"].value_counts()
        fig = px.bar(y=pc.index, x=pc.values, orientation="h", text=pc.values,
                     color=pc.values, color_continuous_scale="Purples",
                     labels={"x":"Count","y":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,coloraxis_showscale=False,height=300)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="sub-header">Purchase Likelihood (Q25)</div>', unsafe_allow_html=True)
        lo = ["Very unlikely","Unlikely","Neutral","Likely","Very likely"]
        lc = df_f["Q25_purchase_likelihood"].value_counts().reindex(lo).dropna()
        cmap = {"Very unlikely":"#e74c3c","Unlikely":"#f39c12","Neutral":"#95a5a6",
                "Likely":"#2ecc71","Very likely":"#1a5276"}
        fig = px.bar(x=lc.index, y=lc.values, text=lc.values,
                     color=lc.index, color_discrete_map=cmap,
                     labels={"x":"Likelihood","y":"Count"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown('<div class="sub-header">Top 10 States</div>', unsafe_allow_html=True)
        sc = df_f["Q3_state"].value_counts().head(10)
        fig = px.bar(y=sc.index, x=sc.values, orientation="h", text=sc.values,
                     color=sc.values, color_continuous_scale="Greens",
                     labels={"x":"Count","y":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,coloraxis_showscale=False,height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Product Interest (% respondents — Q13)</div>', unsafe_allow_html=True)
    prods_map = {
        "rPET Bottles/Containers":"rPET Bottles",
        "Outdoor Garden Furniture":"Garden Furniture",
        "Storage Organisers":"Storage/Organisers",
        "Agri Crates/Tools":"Agri Crates",
        "Stationery/School Supplies":"Stationery"
    }
    pcounts = {lbl: df_f["Q13_products_interested"].str.contains(prod, na=False).sum()
               for prod, lbl in prods_map.items()}
    pct_df = pd.DataFrame({"Product": list(pcounts.keys()),
                            "Pct": [v/n*100 for v in pcounts.values()]})
    fig = px.bar(pct_df, x="Product", y="Pct", text=pct_df["Pct"].round(1),
                 color="Pct", color_continuous_scale="Blues",
                 labels={"Pct":"% Interested"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=330)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Purchase Likelihood by City Tier</div>', unsafe_allow_html=True)
    cross = df_f.groupby(["Q4_city_tier","Q25_purchase_likelihood"]).size().reset_index(name="Count")
    fig = px.bar(cross, x="Q4_city_tier", y="Count", color="Q25_purchase_likelihood",
                 barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
                 labels={"Q4_city_tier":"City Tier","Q25_purchase_likelihood":"Likelihood"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Waste Segregation vs Avg Eco Awareness</div>', unsafe_allow_html=True)
    ws = df_f.groupby("Q_waste_segregation")["Q10_eco_awareness_1to5"].mean().reset_index()
    ws.columns = ["Waste Segregation","Avg Awareness"]
    fig = px.bar(ws, x="Waste Segregation", y="Avg Awareness",
                 text=ws["Avg Awareness"].round(2),
                 color="Avg Awareness", color_continuous_scale="Greens",
                 labels={"Avg Awareness":"Avg Eco Awareness (1-5)"})
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Occupation Distribution</div>', unsafe_allow_html=True)
    oc = df_f["Q6_occupation"].value_counts().reset_index()
    oc.columns = ["Occupation","Count"]
    fig = px.bar(oc, x="Occupation", y="Count", text="Count",
                 color="Count", color_continuous_scale="Oranges")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="main-header">🔍 Diagnostic Deep Dive</div>', unsafe_allow_html=True)
    st.caption("Why are customers buying — or not?")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sub-header">Eco Awareness by Persona</div>', unsafe_allow_html=True)
        ap = df_f.groupby("persona_label")["Q10_eco_awareness_1to5"].mean().sort_values().reset_index()
        fig = px.bar(ap, x="Q10_eco_awareness_1to5", y="persona_label", orientation="h",
                     text=ap["Q10_eco_awareness_1to5"].round(2),
                     color="Q10_eco_awareness_1to5", color_continuous_scale="Blues",
                     labels={"Q10_eco_awareness_1to5":"Avg Awareness","persona_label":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=310)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sub-header">Purchase Barriers (%)</div>', unsafe_allow_html=True)
        bc = df_f["Q24_purchase_barrier"].value_counts().reset_index()
        bc.columns = ["Barrier","Count"]
        bc["Pct"] = (bc["Count"]/n*100).round(1)
        fig = px.bar(bc, x="Pct", y="Barrier", orientation="h",
                     text="Pct", color="Pct", color_continuous_scale="Reds_r",
                     labels={"Pct":"% Respondents","Barrier":""})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=310)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sub-header">WTP Premium → Buy Rate</div>', unsafe_allow_html=True)
        wo = ["Only if cheaper","Same price","Up to 10% more","Up to 25% more","More than 25% premium"]
        wdf = df_f.groupby("Q9_willingness_to_pay_premium")["Q25_purchase_binary"].mean().reindex(wo).dropna().reset_index()
        wdf.columns = ["WTP","BuyRate"]
        wdf["BuyRate%"] = (wdf["BuyRate"]*100).round(1)
        fig = px.line(wdf, x="WTP", y="BuyRate%", markers=True, text="BuyRate%",
                      labels={"WTP":"Willingness to Pay","BuyRate%":"% Likely to Buy"})
        fig.update_traces(textposition="top center", line_color="#1a5276", marker_size=9)
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="sub-header">Prior Eco Purchase → Buy Rate</div>', unsafe_allow_html=True)
        eo = ["No and not interested","No but interested","Yes once or twice","Yes regularly"]
        edf = df_f.groupby("Q12_prior_eco_purchase")["Q25_purchase_binary"].mean().reindex(eo).dropna().reset_index()
        edf.columns = ["PriorPurchase","BuyRate"]
        edf["BuyRate%"] = (edf["BuyRate"]*100).round(1)
        fig = px.bar(edf, x="PriorPurchase", y="BuyRate%", text="BuyRate%",
                     color="BuyRate%", color_continuous_scale="Greens",
                     labels={"PriorPurchase":"","BuyRate%":"% Likely to Buy"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=310)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Correlation Heatmap — Key Variables</div>', unsafe_allow_html=True)
    hcols = [c for c in [
        "Q10_eco_awareness_1to5","Q21_bundle_deal_appeal_1to5","Q23_eco_label_influence_1to5",
        "Q7_income_midpoint_INR","Q8_spend_midpoint_INR","Q_hh_size",
        "Q_children_under15","Q25_purchase_binary"
    ] if c in df_f.columns]
    hdf = df_f[hcols].apply(pd.to_numeric, errors="coerce").dropna()
    fig = px.imshow(hdf.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="sub-header">Channel vs Purchase Likelihood</div>', unsafe_allow_html=True)
        chl = df_f.groupby(["Q18_primary_channel","Q25_purchase_likelihood"]).size().reset_index(name="Count")
        fig = px.bar(chl, x="Q18_primary_channel", y="Count",
                     color="Q25_purchase_likelihood", barmode="stack",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={"Q18_primary_channel":"Channel","Q25_purchase_likelihood":"Likelihood"})
        fig.update_layout(height=360, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown('<div class="sub-header">Decision Maker → Buy Rate</div>', unsafe_allow_html=True)
        dm = df_f.groupby("Q_purchase_decision_maker")["Q25_purchase_binary"].mean().reset_index()
        dm.columns = ["DecisionMaker","BuyRate"]
        dm["BuyRate%"] = (dm["BuyRate"]*100).round(1)
        dm = dm.sort_values("BuyRate%")
        fig = px.bar(dm, x="BuyRate%", y="DecisionMaker", orientation="h",
                     text="BuyRate%", color="BuyRate%", color_continuous_scale="Blues",
                     labels={"DecisionMaker":""})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=310)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Eco Awareness Score Distribution by Purchase Outcome</div>', unsafe_allow_html=True)
    fig = px.histogram(df_f, x="Q10_eco_awareness_1to5",
                       color="Q25_purchase_likelihood",
                       barmode="overlay", opacity=0.65,
                       color_discrete_sequence=px.colors.qualitative.Set1,
                       labels={"Q10_eco_awareness_1to5":"Eco Awareness (1-5)"})
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Income vs Spend — Scatter by City Tier</div>', unsafe_allow_html=True)
    fig = px.scatter(df_f, x="Q7_income_midpoint_INR", y="Q8_spend_midpoint_INR",
                     color="Q4_city_tier", opacity=0.5,
                     labels={"Q7_income_midpoint_INR":"Monthly Income (₹)",
                             "Q8_spend_midpoint_INR":"Monthly Spend (₹)",
                             "Q4_city_tier":"City Tier"},
                     trendline="ols")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">🟢 <b>Insight:</b> WTP Premium and Prior Eco Purchase are the two strongest behavioural predictors of conversion. Qualify leads on these two signals first.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">🟡 <b>Watch:</b> "Doubt about quality" dominates barriers. Lead marketing with certification, guarantee, and testimonials — not price cuts.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="main-header">🤖 Classification — Predict Purchase Likelihood</div>', unsafe_allow_html=True)
    st.caption("Random Forest & Logistic Regression on Q25 (binary). Test set = 30%.")

    X, y, feat_cols = get_feature_matrix(df_p)

    if y is None or len(y) < 80:
        st.error("Not enough data for classification. Broaden filters.")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a: n_est   = st.slider("RF — trees",        50, 300, 150, 50)
        with col_b: max_d   = st.slider("RF — max depth",    3,  20,  8)
        with col_c: C_val   = st.select_slider("LR — C (regularisation)",
                                                options=[0.01,0.1,1.0,10.0], value=1.0)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        with st.spinner("Training classifiers..."):
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                         random_state=42, class_weight="balanced")
            rf.fit(X_tr, y_tr)
            yp_rf   = rf.predict(X_te)
            ypr_rf  = rf.predict_proba(X_te)[:,1]

            lr = LogisticRegression(C=C_val, max_iter=1000,
                                     random_state=42, class_weight="balanced")
            lr.fit(X_tr_sc, y_tr)
            yp_lr   = lr.predict(X_te_sc)
            ypr_lr  = lr.predict_proba(X_te_sc)[:,1]

        def metrics_row(yt, yp, ypr, name):
            return {"Model":name,
                    "Accuracy": round(accuracy_score(yt,yp),4),
                    "Precision":round(precision_score(yt,yp,zero_division=0),4),
                    "Recall":   round(recall_score(yt,yp,zero_division=0),4),
                    "F1 Score": round(f1_score(yt,yp,zero_division=0),4),
                    "ROC-AUC":  round(roc_auc_score(yt,ypr),4)}

        mdf = pd.DataFrame([metrics_row(y_te,yp_rf,ypr_rf,"Random Forest"),
                             metrics_row(y_te,yp_lr,ypr_lr,"Logistic Regression")]).set_index("Model")

        st.markdown('<div class="sub-header">Performance Metrics (Test Set)</div>', unsafe_allow_html=True)
        st.dataframe(mdf.style.format("{:.4f}").highlight_max(axis=0,color="#d5f5e3"),
                     use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sub-header">ROC Curve</div>', unsafe_allow_html=True)
            fpr_rf, tpr_rf, _ = roc_curve(y_te, ypr_rf)
            fpr_lr, tpr_lr, _ = roc_curve(y_te, ypr_lr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines",
                name=f"Random Forest (AUC={mdf.loc['Random Forest','ROC-AUC']:.3f})",
                line=dict(color="#1a5276",width=2.5)))
            fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode="lines",
                name=f"Logistic Reg (AUC={mdf.loc['Logistic Regression','ROC-AUC']:.3f})",
                line=dict(color="#e74c3c",width=2.5,dash="dash")))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                name="Baseline",line=dict(color="gray",dash="dot")))
            fig.update_layout(xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              height=380, legend=dict(x=0.38,y=0.08))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="sub-header">Confusion Matrix — Random Forest</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_te, yp_rf)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Pred: No Buy","Pred: Buy"],
                            y=["Actual: No Buy","Actual: Buy"])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Feature Importance — Random Forest (Top 20)</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({"Feature":feat_cols,"Importance":rf.feature_importances_})
        fi = fi.sort_values("Importance",ascending=False).head(20)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     text=fi["Importance"].round(4),
                     color="Importance", color_continuous_scale="Blues",
                     labels={"Feature":"","Importance":"Importance Score"})
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False,height=520,yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">5-Fold Cross-Validation F1 — Random Forest</div>', unsafe_allow_html=True)
        with st.spinner("Running cross-validation..."):
            cv = cross_val_score(
                RandomForestClassifier(n_estimators=100,max_depth=max_d,
                                        random_state=42,class_weight="balanced"),
                X, y, cv=5, scoring="f1"
            )
        cv_df = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(5)],"F1":cv})
        fig = px.bar(cv_df, x="Fold", y="F1", text=cv_df["F1"].round(3),
                     color="F1", color_continuous_scale="Greens",
                     title=f"Mean F1 = {cv.mean():.3f}  ±  {cv.std():.3f}")
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False,height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Prediction Probability Distribution</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Buy Probability": ypr_rf,
                                 "Actual": y_te.map({0:"No Buy",1:"Buy"})})
        fig = px.histogram(prob_df, x="Buy Probability", color="Actual",
                           nbins=30, barmode="overlay", opacity=0.7,
                           color_discrete_map={"Buy":"#1e8449","No Buy":"#e74c3c"})
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">🟢 <b>Prescriptive:</b> Use RF probability scores >0.5 as your lead scoring threshold. Customers with score >0.7 = premium outreach; 0.4–0.7 = bundle-deal campaign; <0.4 = awareness-only content.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="main-header">🧩 Clustering — Customer Persona Segmentation</div>', unsafe_allow_html=True)
    st.caption("K-Means on behavioural features. Each cluster gets a prescriptive marketing strategy.")

    cl_feats = [c for c in [
        "Q10_eco_awareness_1to5","Q21_bundle_deal_appeal_1to5",
        "Q23_eco_label_influence_1to5","Q7_income_midpoint_INR",
        "Q8_spend_midpoint_INR"
    ] + [c+"_enc" for c in ORDINAL_MAPS] if c in df_p.columns]

    Xcl = df_p[cl_feats].apply(pd.to_numeric,errors="coerce")
    imp_cl = SimpleImputer(strategy="median")
    Xcl_imp = pd.DataFrame(imp_cl.fit_transform(Xcl), columns=Xcl.columns)
    sc_cl = StandardScaler()
    Xcl_sc = sc_cl.fit_transform(Xcl_imp)

    k_val = st.slider("Number of Clusters (K)", 2, 8, 5)

    with st.spinner("Computing elbow + silhouette..."):
        inertias, sils = [], []
        for k in range(2,9):
            km_tmp = KMeans(n_clusters=k,random_state=42,n_init=10)
            labs = km_tmp.fit_predict(Xcl_sc)
            inertias.append(km_tmp.inertia_)
            sils.append(silhouette_score(Xcl_sc,labs))
        km = KMeans(n_clusters=k_val,random_state=42,n_init=10)
        df_p["cluster"]  = km.fit_predict(Xcl_sc)
        df_f = df_f.copy()
        df_f["cluster"]  = df_p["cluster"].values

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sub-header">Elbow Curve</div>', unsafe_allow_html=True)
        fig = px.line(x=list(range(2,9)), y=inertias, markers=True,
                      labels={"x":"K","y":"Inertia (WCSS)"})
        fig.update_traces(line_color="#1a5276",marker_size=8)
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sub-header">Silhouette Score by K</div>', unsafe_allow_html=True)
        fig = px.bar(x=list(range(2,9)), y=sils,
                     text=[round(s,3) for s in sils],
                     color=sils, color_continuous_scale="Greens",
                     labels={"x":"K","y":"Silhouette Score"})
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False,height=280)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Cluster Size Distribution</div>', unsafe_allow_html=True)
    cl_cnt = df_f["cluster"].value_counts().sort_index().reset_index()
    cl_cnt.columns = ["Cluster","Count"]
    cl_cnt["Cluster"] = "Cluster "+cl_cnt["Cluster"].astype(str)
    fig = px.bar(cl_cnt, x="Cluster", y="Count", text="Count",
                 color="Count", color_continuous_scale="Purples")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False,height=280)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Cluster Profile — Mean Values</div>', unsafe_allow_html=True)
    profile_cols = [c for c in [
        "Q10_eco_awareness_1to5","Q21_bundle_deal_appeal_1to5",
        "Q23_eco_label_influence_1to5","Q7_income_midpoint_INR",
        "Q8_spend_midpoint_INR","Q25_purchase_binary"
    ] if c in df_f.columns]
    prof = df_f.groupby("cluster")[profile_cols].mean().round(2)
    prof.index = ["Cluster "+str(i) for i in prof.index]
    st.dataframe(prof.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.markdown('<div class="sub-header">PCA 2D Cluster Visualisation</div>', unsafe_allow_html=True)
    pca = PCA(n_components=2,random_state=42)
    Xpca = pca.fit_transform(Xcl_sc)
    pca_df = pd.DataFrame(Xpca,columns=["PC1","PC2"])
    pca_df["Cluster"] = df_p["cluster"].astype(str)
    pca_df["Persona"] = df_p["persona_label"].values
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                     hover_data=["Persona"],
                     color_discrete_sequence=px.colors.qualitative.Set1,
                     labels={"PC1":f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                             "PC2":f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"},
                     title=f"PCA Projection — {k_val} Clusters")
    fig.update_traces(marker=dict(size=5,opacity=0.65))
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Cluster Buy Rate vs Avg Spend</div>', unsafe_allow_html=True)
    cl_stat = df_f.groupby("cluster").agg(
        Buy_Rate=("Q25_purchase_binary", lambda x: round(x.mean()*100,1)),
        Avg_Spend=("Q8_spend_midpoint_INR","mean"),
        Count=("Q25_purchase_binary","count")
    ).reset_index()
    cl_stat["CL"] = "C"+cl_stat["cluster"].astype(str)
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=cl_stat["CL"],y=cl_stat["Buy_Rate"],
                          name="Buy Rate (%)",marker_color="#1a5276"), secondary_y=False)
    fig.add_trace(go.Scatter(x=cl_stat["CL"],y=cl_stat["Avg_Spend"],
                              mode="lines+markers",name="Avg Spend (₹)",
                              line=dict(color="#e74c3c",width=2),marker_size=9), secondary_y=True)
    fig.update_layout(height=340,legend=dict(x=0.65,y=0.95))
    fig.update_yaxes(title_text="Buy Rate (%)",secondary_y=False)
    fig.update_yaxes(title_text="Avg Spend (₹)",secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">📋 Prescriptive Strategy Per Cluster</div>', unsafe_allow_html=True)
    for _, row in cl_stat.iterrows():
        br = row["Buy_Rate"]; sp = row["Avg_Spend"]; cid = row["CL"]
        if   br >= 40: s = "🟢 High-priority · Loyalty programme + premium upsell. Minimal discount needed."
        elif br >= 25: s = "🟡 Mid-priority · Bundle deals (15-20% off). Eco-label content. Target via Instagram/YouTube."
        elif br >= 12: s = "🟠 Low-priority · Free sample + awareness content. WhatsApp/Facebook campaigns."
        else:          s = "🔴 Dormant · Long-term nurture only. Low spend. Focus on awareness drip campaigns."
        st.markdown(
            f'<div class="action-card"><b>{cid}</b> &nbsp;|&nbsp; Buy Rate: <b>{br:.1f}%</b> &nbsp;|&nbsp; Avg Spend: <b>₹{sp:,.0f}</b><br>{s}</div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — REGRESSION + ASSOCIATION + NEW CUSTOMER UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="main-header">📐 Regression, Association Rules & New Customer Intelligence</div>', unsafe_allow_html=True)
    rtab, atab, utab = st.tabs(["📈 Regression","🔗 Association Rules","🆕 New Customer Score"])

    # ── REGRESSION ────────────────────────────────────────────────────────
    with rtab:
        st.caption("Predict monthly spend (₹) using Gradient Boosting & Linear Regression.")

        r_feats = list(dict.fromkeys([c for c in [
            "Q7_income_midpoint_INR","Q10_eco_awareness_1to5",
            "Q21_bundle_deal_appeal_1to5","Q23_eco_label_influence_1to5","Q_hh_size"
        ] + [c+"_enc" for c in ORDINAL_MAPS if c!="Q8_monthly_spend_band"]
          if c in df_p.columns and c!="Q8_spend_midpoint_INR"]))

        yr = df_p["Q8_spend_midpoint_INR"].fillna(df_p["Q8_spend_midpoint_INR"].median())
        Xr = df_p[r_feats].apply(pd.to_numeric,errors="coerce")
        imp_r = SimpleImputer(strategy="median")
        Xr_imp = pd.DataFrame(imp_r.fit_transform(Xr),columns=Xr.columns)

        Xr_tr,Xr_te,yr_tr,yr_te = train_test_split(Xr_imp,yr,test_size=0.3,random_state=42)

        with st.spinner("Training regression models..."):
            gbr = GradientBoostingRegressor(n_estimators=150,max_depth=4,
                                             learning_rate=0.1,random_state=42)
            gbr.fit(Xr_tr,yr_tr); yp_gbr = gbr.predict(Xr_te)
            lin = LinearRegression()
            lin.fit(Xr_tr,yr_tr); yp_lin = lin.predict(Xr_te)

        def rmets(yt,yp,name):
            return {"Model":name,
                    "MAE (₹)":round(mean_absolute_error(yt,yp),0),
                    "RMSE (₹)":round(np.sqrt(mean_squared_error(yt,yp)),0),
                    "R² Score":round(r2_score(yt,yp),4)}

        rmdf = pd.DataFrame([rmets(yr_te,yp_gbr,"Gradient Boosting"),
                              rmets(yr_te,yp_lin,"Linear Regression")]).set_index("Model")
        st.dataframe(rmdf.style.format({"MAE (₹)":"{:,.0f}","RMSE (₹)":"{:,.0f}","R² Score":"{:.4f}"}),
                     use_container_width=True)

        r1,r2 = st.columns(2)
        with r1:
            st.markdown('<div class="sub-header">Actual vs Predicted — GBR</div>', unsafe_allow_html=True)
            fig = px.scatter(x=yr_te.values,y=yp_gbr,opacity=0.5,
                             labels={"x":"Actual Spend (₹)","y":"Predicted Spend (₹)"},
                             color_discrete_sequence=["#1a5276"])
            fig.add_shape(type="line",x0=yr_te.min(),y0=yr_te.min(),
                          x1=yr_te.max(),y1=yr_te.max(),
                          line=dict(color="red",dash="dash"))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown('<div class="sub-header">Residuals — GBR</div>', unsafe_allow_html=True)
            res = yr_te.values - yp_gbr
            fig = px.histogram(x=res,nbins=40,
                               labels={"x":"Residual (₹)","y":"Count"},
                               color_discrete_sequence=["#2e86c1"])
            fig.add_vline(x=0,line_dash="dash",line_color="red")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Feature Importance — GBR</div>', unsafe_allow_html=True)
        fi_r = pd.DataFrame({"Feature":r_feats,"Importance":gbr.feature_importances_})
        fi_r = fi_r.sort_values("Importance",ascending=False)
        fig = px.bar(fi_r,x="Importance",y="Feature",orientation="h",
                     text=fi_r["Importance"].round(4),
                     color="Importance",color_continuous_scale="Oranges",
                     labels={"Feature":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False,height=420,yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Avg Predicted Spend by Persona</div>', unsafe_allow_html=True)
        Xfull = pd.DataFrame(imp_r.transform(df_p[r_feats].apply(pd.to_numeric,errors="coerce")),
                              columns=r_feats)
        df_p["pred_spend"] = gbr.predict(Xfull)
        sp_p = df_p.groupby("persona_label")["pred_spend"].mean().sort_values().reset_index()
        fig = px.bar(sp_p,x="pred_spend",y="persona_label",orientation="h",
                     text=sp_p["pred_spend"].round(0),
                     color="pred_spend",color_continuous_scale="Oranges",
                     labels={"pred_spend":"Avg Predicted Spend (₹)","persona_label":""})
        fig.update_traces(texttemplate="₹%{text:,.0f}",textposition="outside")
        fig.update_layout(coloraxis_showscale=False,height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Spend Distribution by City Tier (Actual vs Predicted)</div>', unsafe_allow_html=True)
        sp_ct = df_p.copy()
        sp_ct["city_tier"] = df_f["Q4_city_tier"].values
        sp_ct_g = sp_ct.groupby("city_tier").agg(
            Actual=("Q8_spend_midpoint_INR","mean"),
            Predicted=("pred_spend","mean")
        ).reset_index()
        fig = px.bar(sp_ct_g.melt(id_vars="city_tier",value_vars=["Actual","Predicted"]),
                     x="city_tier",y="value",color="variable",barmode="group",
                     text_auto=".0f",
                     labels={"city_tier":"City Tier","value":"Avg Spend (₹)","variable":""})
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True)

    # ── ASSOCIATION RULES ──────────────────────────────────────────────────
    with atab:
        st.caption("Apriori on product interest, cookware, clothing and occasions.")

        a1,a2,a3 = st.columns(3)
        with a1: min_sup  = st.slider("Min Support",    0.01,0.20,0.04,0.01)
        with a2: min_conf = st.slider("Min Confidence", 0.30,0.90,0.50,0.05)
        with a3: min_lift = st.slider("Min Lift",       1.0, 3.0, 1.1, 0.1)

        cat = st.selectbox("Association category",[
            "Products (Q13)","Cookware (Q15)","Clothing/Sarees (Q14)",
            "Purchase Occasions (Q17)","All combined"
        ])
        cat_map = {
            "Products (Q13)":          ["Q13_products_interested"],
            "Cookware (Q15)":          ["Q15_cookware_types"],
            "Clothing/Sarees (Q14)":   ["Q14_clothing_types"],
            "Purchase Occasions (Q17)":["Q17_purchase_occasions"],
            "All combined":            ["Q13_products_interested","Q15_cookware_types",
                                        "Q14_clothing_types","Q17_purchase_occasions"]
        }
        sel_cols = [c for c in cat_map[cat] if c in df_f.columns]

        txns = []
        for _, row in df_f.iterrows():
            basket = []
            for col in sel_cols:
                if pd.notna(row.get(col)):
                    basket.extend([i.strip() for i in str(row[col]).split("|")
                                   if i.strip() not in ["None","Unknown",""]])
            if len(basket) >= 2:
                txns.append(basket)

        if len(txns) < 10:
            st.warning("Not enough transactions. Broaden filters or try a different category.")
        else:
            te = TransactionEncoder()
            te_arr = te.fit_transform(txns)
            bdf = pd.DataFrame(te_arr, columns=te.columns_)
            try:
                freq = apriori(bdf, min_support=min_sup, use_colnames=True)
                if freq.empty:
                    st.warning("No frequent itemsets. Reduce min support.")
                else:
                    rules = association_rules(freq, metric="lift",
                                              min_threshold=min_lift,
                                              num_itemsets=len(freq))
                    rules = rules[rules["confidence"] >= min_conf].copy()
                    rules["antecedents_str"] = rules["antecedents"].apply(lambda x:", ".join(list(x)))
                    rules["consequents_str"] = rules["consequents"].apply(lambda x:", ".join(list(x)))
                    rules = rules.sort_values("lift",ascending=False).head(60)

                    st.markdown(
                        f'<div class="insight-box">✅ Found <b>{len(rules)}</b> rules &nbsp;|&nbsp; '
                        f'support≥{min_sup} &nbsp;|&nbsp; confidence≥{min_conf} &nbsp;|&nbsp; lift≥{min_lift}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown('<div class="sub-header">Rules Table (sorted by Lift)</div>', unsafe_allow_html=True)
                    disp = rules[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
                    disp.columns = ["If customer has →","→ Also likely to have","Support","Confidence","Lift"]
                    disp[["Support","Confidence","Lift"]] = disp[["Support","Confidence","Lift"]].round(3)
                    st.dataframe(disp.style.background_gradient(subset=["Lift","Confidence"],cmap="Greens"),
                                 use_container_width=True, height=400)

                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown('<div class="sub-header">Support vs Confidence (size = Lift)</div>', unsafe_allow_html=True)
                        fig = px.scatter(rules,x="support",y="confidence",size="lift",
                                         color="lift",hover_data=["antecedents_str","consequents_str"],
                                         color_continuous_scale="Greens",
                                         labels={"support":"Support","confidence":"Confidence","lift":"Lift"})
                        fig.update_layout(height=370)
                        st.plotly_chart(fig, use_container_width=True)

                    with b2:
                        st.markdown('<div class="sub-header">Top 15 Rules by Lift</div>', unsafe_allow_html=True)
                        t15 = rules.head(15).copy()
                        t15["Rule"] = t15["antecedents_str"]+" → "+t15["consequents_str"]
                        fig = px.bar(t15,x="lift",y="Rule",orientation="h",
                                     text=t15["lift"].round(2),
                                     color="confidence",color_continuous_scale="Greens",
                                     labels={"lift":"Lift","Rule":""},
                                     color_continuous_colorbar=dict(title="Confidence"))
                        fig.update_traces(textposition="outside")
                        fig.update_layout(coloraxis_showscale=True,height=420,
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<div class="sub-header">Confidence vs Lift — All Rules</div>', unsafe_allow_html=True)
                    fig = px.scatter(rules,x="confidence",y="lift",
                                     color="support",color_continuous_scale="Blues",
                                     hover_data=["antecedents_str","consequents_str"],
                                     size=[8]*len(rules),
                                     labels={"confidence":"Confidence","lift":"Lift","support":"Support"})
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<div class="sub-header">Support Distribution of Frequent Itemsets</div>', unsafe_allow_html=True)
                    fig = px.histogram(freq,x="support",nbins=30,
                                       color_discrete_sequence=["#1a5276"],
                                       labels={"support":"Support","count":"Itemsets"})
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Association rules error: {e}")

    # ── NEW CUSTOMER INTELLIGENCE ──────────────────────────────────────────
    with utab:
        st.markdown("### 🆕 Upload New Prospects — Get Instant Buy-Probability Scores")
        st.caption("Upload a CSV matching the survey schema. Each prospect is scored, clustered, and prescribed a marketing action.")

        st.markdown('<div class="warning-box">Your CSV must follow the same column structure as the survey dataset. Download the template below.</div>', unsafe_allow_html=True)

        tmpl = pd.DataFrame(columns=[
            "Q1_age_group","Q2_gender","Q3_state","Q4_city_tier","Q5_education",
            "Q6_occupation","Q_hh_size","Q_hh_type","Q_children_under15",
            "Q_waste_segregation","Q_purchase_decision_maker","Q7_income_band",
            "Q7_income_midpoint_INR","Q8_monthly_spend_band","Q8_spend_midpoint_INR",
            "Q9_willingness_to_pay_premium","Q10_eco_awareness_1to5",
            "Q11_reduce_plastic_freq","Q12_prior_eco_purchase",
            "Q13_products_interested","Q14_clothing_types","Q15_cookware_types",
            "Q16_colour_preference","Q17_purchase_occasions","Q18_primary_channel",
            "Q19_top3_purchase_factors","Q20_shopping_frequency",
            "Q21_bundle_deal_appeal_1to5","Q22_social_media_channels",
            "Q23_eco_label_influence_1to5","Q24_purchase_barrier"
        ])
        st.download_button("⬇ Download CSV Template", data=tmpl.to_csv(index=False),
                           file_name="new_customers_template.csv", mime="text/csv")

        new_file = st.file_uploader("📤 Upload new customer CSV", type=["csv"], key="nc")

        if new_file:
            new_df = pd.read_csv(new_file)
            st.success(f"Loaded {len(new_df):,} new customer records.")

            req_cols = ["Q7_income_midpoint_INR","Q10_eco_awareness_1to5"]
            missing  = [c for c in req_cols if c not in new_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}. Use the template above.")
            else:
                new_proc = load_and_preprocess(new_df)
                X_new_raw, _, _ = get_feature_matrix(new_proc)

                # Align columns with training set
                for c in X.columns:
                    if c not in X_new_raw.columns:
                        X_new_raw[c] = 0
                X_new_aligned = X_new_raw[X.columns]
                imp_new = SimpleImputer(strategy="median")
                imp_new.fit(X)
                X_new_final = pd.DataFrame(imp_new.transform(X_new_aligned), columns=X.columns)

                # Retrain RF on full dataset
                rf_full = RandomForestClassifier(n_estimators=150, max_depth=8,
                                                  random_state=42, class_weight="balanced")
                rf_full.fit(X, y)
                probs = rf_full.predict_proba(X_new_final)[:,1]
                preds = rf_full.predict(X_new_final)

                new_df["buy_probability_pct"] = (probs*100).round(1)
                new_df["predicted_label"]      = np.where(preds==1,"Likely to Buy","Unlikely to Buy")

                def prescribe(prob):
                    if   prob >= 70: return "🟢 High priority — personalised outreach, premium bundle offer"
                    elif prob >= 45: return "🟡 Medium priority — bundle deal (15% off), eco-label content"
                    elif prob >= 25: return "🟠 Low priority — awareness content, WhatsApp drip campaign"
                    else:            return "🔴 Nurture only — long-term awareness, minimal ad spend"

                new_df["recommended_action"] = new_df["buy_probability_pct"].apply(prescribe)

                id_cols = [c for c in ["Q1_age_group","Q4_city_tier","Q7_income_band","persona_label"]
                           if c in new_df.columns]
                show_cols = ["buy_probability_pct","predicted_label","recommended_action"] + id_cols

                st.markdown('<div class="sub-header">Scored Results</div>', unsafe_allow_html=True)
                st.dataframe(new_df[show_cols], use_container_width=True, height=420)

                st.markdown('<div class="sub-header">Buy Probability Distribution</div>', unsafe_allow_html=True)
                fig = px.histogram(new_df, x="buy_probability_pct", color="predicted_label",
                                   nbins=20, barmode="overlay",
                                   color_discrete_map={"Likely to Buy":"#1e8449","Unlikely to Buy":"#e74c3c"},
                                   labels={"buy_probability_pct":"Buy Probability (%)","count":"Customers"})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="sub-header">Action Tier Breakdown</div>', unsafe_allow_html=True)
                act_cnt = new_df["recommended_action"].value_counts().reset_index()
                act_cnt.columns = ["Action","Count"]
                fig = px.bar(act_cnt,x="Count",y="Action",orientation="h",
                             text="Count",color="Count",color_continuous_scale="Blues",
                             labels={"Action":""})
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False,height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.download_button("⬇ Download Scored Results CSV",
                                   data=new_df.to_csv(index=False),
                                   file_name="scored_customers.csv", mime="text/csv")
        else:
            st.markdown("""
<div class="insight-box">
<b>How this works:</b><br>
1. Fill the template CSV with your new prospect data (from leads, events, referrals)<br>
2. Upload it here — the trained Random Forest scores each person instantly<br>
3. Download the scored CSV with buy probability % + recommended marketing action per prospect<br>
4. Feed directly into your CRM / WhatsApp campaign / sales team priority list
</div>
""", unsafe_allow_html=True)
            st.markdown("#### Output columns you'll receive:")
            cols_out = {
                "buy_probability_pct":  "0–100% likelihood of purchasing",
                "predicted_label":       "Likely to Buy / Unlikely to Buy",
                "recommended_action":    "Tiered prescriptive marketing action"
            }
            for col, desc in cols_out.items():
                st.markdown(f"- **`{col}`** — {desc}")
