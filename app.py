import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard นักท่องเที่ยวบุรีรัมย์",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        border-left: 5px solid #2d6a9f;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4f8;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #2d6a9f !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# THAI MONTH MAP
# ─────────────────────────────────────────────────────────
MONTH_MAP = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3,
    "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9,
    "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "มกราคม - มีนาคม": 1, "เมษายน - มิถุนายน": 4,
    "กรกฎาคม - กันยายน": 7, "ตุลาคม - ธันวาคม": 10,
}
QUARTER_LABEL = {1: "Q1", 4: "Q2", 7: "Q3", 10: "Q4"}
MONTH_TH = {1:"ม.ค.", 2:"ก.พ.", 3:"มี.ค.", 4:"เม.ย.", 5:"พ.ค.", 6:"มิ.ย.",
            7:"ก.ค.", 8:"ส.ค.", 9:"ก.ย.", 10:"ต.ค.", 11:"พ.ย.", 12:"ธ.ค."}

# ─────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(path="dataCI02-09-03-2569.csv"):
    df_raw = pd.read_csv(path)

    # ── 1. Keep only rows that have actual Total_vis values ──
    df = df_raw[df_raw["Total_vis"].notna()].copy()

    # ── 2. Clean numeric columns (remove commas) ──
    num_cols = ["Total_vis", "Thai_vis", "Foreign_vis", "Guests_total",
                "Rev_total", "Rev_thai", "Rev_foreign"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 3. Add month_num ──
    df["month_num"] = df["Month&Quarter"].map(MONTH_MAP)

    # ── 4. Determine if monthly or quarterly ──
    quarterly_labels = {"มกราคม - มีนาคม", "เมษายน - มิถุนายน",
                        "กรกฎาคม - กันยายน", "ตุลาคม - ธันวาคม"}
    df["is_quarterly"] = df["Month&Quarter"].isin(quarterly_labels)

    # ── 5. Fill event NaN with 0 ──
    event_cols = ["MotoGP", "Covid", "Marathon", "PhanomRung_Festival"]
    df[event_cols] = df[event_cols].fillna(0)

    # ── 6. Convert Buddhist year → CE year ──
    df["Year_CE"] = df["Year"] - 543

    # ── 7. Sort ──
    df = df.sort_values(["Year", "month_num"]).reset_index(drop=True)

    return df

df = load_and_clean()

# ─────────────────────────────────────────────────────────
# PREPARE ANNUAL DATA (for ML)
# ─────────────────────────────────────────────────────────
@st.cache_data
def build_annual(df):
    # Sum quarterly to annual totals
    annual = (
        df.groupby("Year")
        .agg(
            Total_vis=("Total_vis", "sum"),
            Thai_vis=("Thai_vis", "sum"),
            Foreign_vis=("Foreign_vis", "sum"),
            Rev_total=("Rev_total", "sum"),
            MotoGP=("MotoGP", "max"),
            Covid=("Covid", "max"),
            Marathon=("Marathon", "max"),
            PhanomRung_Festival=("PhanomRung_Festival", "max"),
        )
        .reset_index()
    )
    annual["Year_CE"] = annual["Year"] - 543
    annual["prev_vis"] = annual["Total_vis"].shift(1)
    annual["prev2_vis"] = annual["Total_vis"].shift(2)
    annual = annual.dropna(subset=["prev_vis", "prev2_vis"])
    return annual

annual = build_annual(df)

# ─────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────
@st.cache_data
def train_models(annual):
    feat_cols = ["Year_CE", "prev_vis", "prev2_vis",
                 "MotoGP", "Covid", "Marathon", "PhanomRung_Festival"]
    X = annual[feat_cols].values
    y = annual["Total_vis"].values

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        # Train on all data (small dataset → use all for prediction)
        model.fit(X, y)
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        results[name] = {
            "model": model,
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "y_pred": y_pred,
        }
    return results, feat_cols

model_results, feat_cols = train_models(annual)

# Best model by R²
best_name = max(model_results, key=lambda k: model_results[k]["R2"])
best_model = model_results[best_name]["model"]

# ─────────────────────────────────────────────────────────
# PREDICT 2569
# ─────────────────────────────────────────────────────────
def predict_2569(model, annual, event_override=None):
    last = annual.iloc[-1]
    prev2 = annual.iloc[-2]["Total_vis"] if len(annual) >= 2 else last["Total_vis"]
    events = event_override or {"MotoGP": 1, "Covid": 0, "Marathon": 1, "PhanomRung_Festival": 1}
    X_new = np.array([[2026, last["Total_vis"], prev2,
                        events["MotoGP"], events["Covid"],
                        events["Marathon"], events["PhanomRung_Festival"]]])
    return float(model.predict(X_new)[0])

pred_2569 = predict_2569(best_model, annual)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Seal_Buriram.svg/200px-Seal_Buriram.svg.png", width=80)
    st.markdown("## 🏍️ Dashboard บุรีรัมย์")
    st.markdown("---")

    selected_tab = st.radio(
        "เมนูหลัก",
        ["🏠 ภาพรวม", "📈 การทำนาย", "📅 รายเดือน", "🎪 ผลกระทบเหตุการณ์", "🤖 เปรียบเทียบโมเดล"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ ตั้งค่าการทำนายปี 2569")
    motogp_2569 = st.selectbox("MotoGP 2569", [1, 0], format_func=lambda x: "มี" if x else "ไม่มี")
    marathon_2569 = st.selectbox("Marathon 2569", [1, 0], format_func=lambda x: "มี" if x else "ไม่มี")
    phanomrung_2569 = st.selectbox("Phanom Rung 2569", [1, 0], format_func=lambda x: "มี" if x else "ไม่มี")
    covid_2569 = st.selectbox("COVID 2569", [0, 1], format_func=lambda x: "มี" if x else "ไม่มี")

    event_settings = {
        "MotoGP": motogp_2569, "Covid": covid_2569,
        "Marathon": marathon_2569, "PhanomRung_Festival": phanomrung_2569
    }
    pred_2569_custom = predict_2569(best_model, annual, event_settings)

    st.markdown("---")
    st.caption("ข้อมูล: สถิตินักท่องเที่ยวบุรีรัมย์ 2556–2568")

# ─────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg,#1e3a5f,#2d6a9f);
            padding:24px 32px;border-radius:16px;color:white;margin-bottom:24px;'>
    <h1 style='margin:0;font-size:2rem;'>🏍️ Dashboard นักท่องเที่ยวจังหวัดบุรีรัมย์</h1>
    <p style='margin:6px 0 0;opacity:.85;font-size:1rem;'>
        ข้อมูลสถิติและการพยากรณ์นักท่องเที่ยว ปี 2556–2569
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ── TAB: ภาพรวม ──
# ─────────────────────────────────────────────────────────
if selected_tab == "🏠 ภาพรวม":
    # KPI cards
    latest_year = annual.iloc[-1]
    prev_year = annual.iloc[-2]

    col1, col2, col3, col4 = st.columns(4)
    def kpi(col, label, value, unit="", delta=None):
        delta_html = ""
        if delta is not None:
            color = "lightgreen" if delta >= 0 else "#ff6b6b"
            arrow = "▲" if delta >= 0 else "▼"
            delta_html = f"<div style='color:{color};font-size:.85rem;'>{arrow} {abs(delta):,.0f} จากปีก่อน</div>"
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{value:,.0f} {unit}</div>
            <div class='metric-label'>{label}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

    with col1:
        kpi(col1, f"นักท่องเที่ยวรวมปี {int(latest_year['Year'])}",
            latest_year["Total_vis"],
            delta=latest_year["Total_vis"] - prev_year["Total_vis"])
    with col2:
        kpi(col2, "นักท่องเที่ยวไทย", latest_year["Thai_vis"])
    with col3:
        kpi(col3, "นักท่องเที่ยวต่างชาติ", latest_year["Foreign_vis"])
    with col4:
        kpi(col4, "🔮 คาดการณ์ปี 2569", pred_2569_custom,
            delta=pred_2569_custom - latest_year["Total_vis"])

    st.markdown("---")

    # Annual trend
    st.markdown("<div class='section-header'>📊 แนวโน้มนักท่องเที่ยวรายปี (2556–2568)</div>", unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=annual["Year"], y=annual["Total_vis"],
        mode="lines+markers+text",
        name="จริง",
        line=dict(color="#2d6a9f", width=3),
        marker=dict(size=8),
        text=[f"{v/1e6:.2f}M" for v in annual["Total_vis"]],
        textposition="top center",
    ))
    # Predicted 2569
    fig_trend.add_trace(go.Scatter(
        x=[annual["Year"].iloc[-1], 2569],
        y=[annual["Total_vis"].iloc[-1], pred_2569_custom],
        mode="lines+markers",
        name="ทำนาย 2569",
        line=dict(color="#e74c3c", width=2, dash="dash"),
        marker=dict(size=10, color="#e74c3c"),
    ))
    fig_trend.add_annotation(
        x=2569, y=pred_2569_custom,
        text=f"<b>2569 ≈ {pred_2569_custom/1e6:.2f}M</b>",
        showarrow=True, arrowhead=2, bgcolor="#e74c3c", font=dict(color="white")
    )
    # COVID shade
    fig_trend.add_vrect(x0=2562.5, x1=2565.5, fillcolor="rgba(231,76,60,0.1)",
                        line_width=0, annotation_text="COVID-19", annotation_position="top left")
    fig_trend.update_layout(height=400, showlegend=True,
                            plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_title="ปี (พ.ศ.)", yaxis_title="จำนวนนักท่องเที่ยว (คน)")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Thai vs Foreign
    st.markdown("<div class='section-header'>👥 สัดส่วนนักท่องเที่ยวไทย vs ต่างชาติ</div>", unsafe_allow_html=True)
    annual_clean = annual.dropna(subset=["Thai_vis", "Foreign_vis"])
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=annual_clean["Year"], y=annual_clean["Thai_vis"],
                             name="ไทย", marker_color="#2d6a9f"))
    fig_bar.add_trace(go.Bar(x=annual_clean["Year"], y=annual_clean["Foreign_vis"],
                             name="ต่างชาติ", marker_color="#e67e22"))
    fig_bar.update_layout(barmode="stack", height=360,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="ปี (พ.ศ.)", yaxis_title="จำนวน (คน)")
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ── TAB: การทำนาย ──
# ─────────────────────────────────────────────────────────
elif selected_tab == "📈 การทำนาย":
    st.markdown("<div class='section-header'>🔮 การพยากรณ์ปี 2569 ด้วยโมเดล ML</div>", unsafe_allow_html=True)

    # Model comparison bar
    model_names = list(model_results.keys())
    r2_vals = [model_results[m]["R2"] for m in model_names]
    mae_vals = [model_results[m]["MAE"] for m in model_names]
    rmse_vals = [model_results[m]["RMSE"] for m in model_names]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.info(f"""
        **โมเดลที่ดีที่สุด**: {best_name}  
        **R²**: {model_results[best_name]['R2']:.4f}  
        **MAE**: {model_results[best_name]['MAE']:,.0f}  
        **RMSE**: {model_results[best_name]['RMSE']:,.0f}  
        
        **📌 ผลการทำนายปี 2569**  
        (ตามค่าที่ตั้งใน Sidebar)  
        # {pred_2569_custom:,.0f} คน
        """)

    with col_b:
        fig_r2 = go.Figure(go.Bar(
            x=r2_vals, y=model_names,
            orientation="h",
            marker_color=["#e74c3c" if m == best_name else "#2d6a9f" for m in model_names],
            text=[f"R²={v:.4f}" for v in r2_vals],
            textposition="inside"
        ))
        fig_r2.update_layout(title="R² Score ของแต่ละโมเดล",
                             height=260, xaxis_range=[0, 1],
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>📉 ข้อมูลจริง vs ทำนาย (Best Model)</div>", unsafe_allow_html=True)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=annual["Year"], y=annual["Total_vis"],
        mode="lines+markers", name="ข้อมูลจริง",
        line=dict(color="#2d6a9f", width=3)
    ))
    fig_pred.add_trace(go.Scatter(
        x=annual["Year"], y=model_results[best_name]["y_pred"],
        mode="lines+markers", name=f"ทำนาย ({best_name})",
        line=dict(color="#e67e22", width=2, dash="dot")
    ))
    fig_pred.add_trace(go.Scatter(
        x=[2569], y=[pred_2569_custom],
        mode="markers", name="พยากรณ์ 2569",
        marker=dict(size=16, color="#e74c3c", symbol="star"),
    ))
    fig_pred.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="ปี (พ.ศ.)", yaxis_title="นักท่องเที่ยว (คน)")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Sensitivity — what if MotoGP or no MotoGP?
    st.markdown("<div class='section-header'>🔧 วิเคราะห์ความไว (Sensitivity)</div>", unsafe_allow_html=True)

    scenarios = []
    for motogp in [0, 1]:
        for covid in [0, 1]:
            for marathon in [0, 1]:
                for phanomrung in [0, 1]:
                    ev = {"MotoGP": motogp, "Covid": covid,
                          "Marathon": marathon, "PhanomRung_Festival": phanomrung}
                    p = predict_2569(best_model, annual, ev)
                    label = (f"MotoGP={'✓' if motogp else '✗'} | "
                             f"COVID={'✓' if covid else '✗'} | "
                             f"Marathon={'✓' if marathon else '✗'} | "
                             f"PhanomRung={'✓' if phanomrung else '✗'}")
                    scenarios.append({"Scenario": label, "Prediction": p})

    df_sc = pd.DataFrame(scenarios).sort_values("Prediction", ascending=False)
    fig_sc = px.bar(df_sc.head(10), x="Prediction", y="Scenario",
                    orientation="h", color="Prediction",
                    color_continuous_scale="Blues",
                    title="Top 10 สถานการณ์คาดการณ์ปี 2569")
    fig_sc.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ── TAB: รายเดือน ──
# ─────────────────────────────────────────────────────────
elif selected_tab == "📅 รายเดือน":
    st.markdown("<div class='section-header'>📅 สถิตินักท่องเที่ยวรายเดือน</div>", unsafe_allow_html=True)

    # ── Filter to monthly rows only (ปี 2567–2568 มีรายเดือน) ──
    monthly_df = df[~df["is_quarterly"]].copy()
    quarterly_df = df[df["is_quarterly"]].copy()

    years_monthly = sorted(monthly_df["Year"].unique(), reverse=True)
    years_quarterly = sorted(quarterly_df["Year"].unique(), reverse=True)

    mode = st.radio("ประเภทข้อมูล", ["รายเดือน (2567–2568)", "รายไตรมาส (2556–2566)"], horizontal=True)

    if mode == "รายเดือน (2567–2568)":
        sel_year = st.selectbox("เลือกปี (พ.ศ.)", years_monthly)
        sub = monthly_df[monthly_df["Year"] == sel_year].copy()
        sub["month_label"] = sub["month_num"].map(MONTH_TH)
        sub = sub.sort_values("month_num")

        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=sub["month_label"], y=sub["Total_vis"],
            marker_color="#2d6a9f",
            text=sub["Total_vis"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside"
        ))
        # Mark events
        for _, row in sub.iterrows():
            events_here = []
            if row["MotoGP"] == 1: events_here.append("🏍️ MotoGP")
            if row["Marathon"] == 1: events_here.append("🏃 Marathon")
            if row["PhanomRung_Festival"] == 1: events_here.append("🏯 PhanomRung")
            if events_here:
                fig_m.add_annotation(
                    x=MONTH_TH.get(row["month_num"], ""),
                    y=row["Total_vis"] * 1.12,
                    text="<br>".join(events_here),
                    showarrow=False,
                    font=dict(size=9, color="#c0392b"),
                    bgcolor="rgba(255,255,255,0.8)"
                )

        fig_m.update_layout(title=f"จำนวนนักท่องเที่ยวรายเดือน ปี {sel_year}",
                             height=420, xaxis_title="เดือน", yaxis_title="จำนวน (คน)",
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_m, use_container_width=True)

        # Table
        st.dataframe(
            sub[["month_label", "Total_vis", "Thai_vis", "Foreign_vis",
                 "MotoGP", "Marathon", "PhanomRung_Festival"]]
            .rename(columns={"month_label": "เดือน", "Total_vis": "รวม",
                             "Thai_vis": "ไทย", "Foreign_vis": "ต่างชาติ"}),
            use_container_width=True, hide_index=True
        )

    else:
        sel_year = st.selectbox("เลือกปี (พ.ศ.)", years_quarterly)
        sub = quarterly_df[quarterly_df["Year"] == sel_year].copy()
        sub["quarter_label"] = sub["month_num"].map(QUARTER_LABEL)
        sub = sub.sort_values("month_num")

        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=sub["quarter_label"], y=sub["Total_vis"],
            marker_color="#27ae60",
            text=sub["Total_vis"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside"
        ))
        fig_q.update_layout(title=f"จำนวนนักท่องเที่ยวรายไตรมาส ปี {sel_year}",
                             height=400, xaxis_title="ไตรมาส", yaxis_title="จำนวน (คน)",
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_q, use_container_width=True)

    # ── Multi-year Monthly Heatmap (2567 vs 2568) ──
    st.markdown("<div class='section-header'>🌡️ Heatmap เปรียบเทียบรายเดือน</div>", unsafe_allow_html=True)
    pivot = monthly_df.pivot_table(index="Year", columns="month_num", values="Total_vis", aggfunc="sum")
    pivot.columns = [MONTH_TH.get(c, c) for c in pivot.columns]
    fig_heat = px.imshow(pivot, text_auto=".2s", color_continuous_scale="Blues",
                         aspect="auto", title="Heatmap นักท่องเที่ยวรายเดือน-รายปี")
    fig_heat.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ── TAB: ผลกระทบเหตุการณ์ ──
# ─────────────────────────────────────────────────────────
elif selected_tab == "🎪 ผลกระทบเหตุการณ์":
    st.markdown("<div class='section-header'>🎪 ผลกระทบของเหตุการณ์ต่อจำนวนนักท่องเที่ยว</div>", unsafe_allow_html=True)

    events_list = ["MotoGP", "Marathon", "PhanomRung_Festival", "Covid"]
    event_labels = {"MotoGP": "🏍️ MotoGP", "Marathon": "🏃 Marathon",
                    "PhanomRung_Festival": "🏯 Phanom Rung Festival", "Covid": "🦠 COVID-19"}

    # Annual with events
    annual_ev = annual.copy()

    for ev in events_list:
        has = annual_ev[annual_ev[ev] == 1]["Total_vis"]
        no = annual_ev[annual_ev[ev] == 0]["Total_vis"]
        st.markdown(f"#### {event_labels[ev]}")
        c1, c2, c3 = st.columns(3)
        c1.metric("เฉลี่ยเมื่อมีเหตุการณ์", f"{has.mean():,.0f}")
        c2.metric("เฉลี่ยเมื่อไม่มีเหตุการณ์", f"{no.mean():,.0f}")
        diff = has.mean() - no.mean()
        c3.metric("ผลต่าง", f"{diff:+,.0f}", delta_color="normal" if diff > 0 else "inverse")

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 เปรียบเทียบแต่ละเหตุการณ์ต่อปี</div>", unsafe_allow_html=True)

    sel_event = st.selectbox("เลือกเหตุการณ์", events_list, format_func=lambda x: event_labels[x])

    # Show years where event happened vs next year
    ev_years = annual[annual[sel_event] == 1]["Year"].tolist()

    rows = []
    for yr in ev_years:
    curr_row = annual[annual["Year"] == yr]
    curr = curr_row["Total_vis"].values

    nxt_yr = yr + 1
    nxt_row = annual[annual["Year"] == nxt_yr]
    nxt = nxt_row["Total_vis"].values

    # 🔒 เช็คก่อนว่าปีถัดไปมีข้อมูลไหม
    if not nxt_row.empty:
        pred_nxt = None  # มีข้อมูลจริงแล้ว ไม่ต้อง predict
    else:
        pred_nxt = predict_2569(best_model, annual)

    rows.append({
        "ปี": yr,
        f"จำนวน (ปีที่มี {sel_event})": float(curr[0]) if len(curr) else None,
        "จำนวนปีถัดไป (จริง)": float(nxt[0]) if len(nxt) else None,
        "จำนวนปีถัดไป (คาดการณ์)": float(pred_nxt) if pred_nxt else None
    })

    df_ev = pd.DataFrame(rows).dropna()
    if not df_ev.empty:
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Bar(
            x=df_ev["ปี"].astype(str),
            y=df_ev[f"จำนวน (ปีที่มี {sel_event})"],
            name=f"ปีที่มี {sel_event}", marker_color="#2d6a9f"
        ))
        fig_ev.add_trace(go.Bar(
            x=df_ev["ปี"].astype(str),
            y=df_ev["จำนวนปีถัดไป (จริง)"],
            name="ปีถัดไป (จริง)", marker_color="#e67e22"
        ))
        # Add 2569 prediction
        if annual["Year"].iloc[-1] in ev_years or True:
            fig_ev.add_trace(go.Scatter(
                x=["2568 → 2569"],
                y=[pred_2569_custom],
                mode="markers", name="ทำนายปี 2569",
                marker=dict(size=14, color="#e74c3c", symbol="star")
            ))
        fig_ev.update_layout(barmode="group", height=400,
                             title=f"ผลกระทบของ {event_labels[sel_event]} ต่อจำนวนนักท่องเที่ยว",
                             plot_bgcolor="white", paper_bgcolor="white",
                             xaxis_title="ปี (พ.ศ.)", yaxis_title="จำนวน (คน)")
        st.plotly_chart(fig_ev, use_container_width=True)
    else:
        st.warning(f"ไม่มีข้อมูล {sel_event} เพียงพอในการวิเคราะห์")

    # Football section
    st.markdown("---")
    st.markdown("<div class='section-header'>⚽ ข้อมูลการแข่งขันฟุตบอล</div>", unsafe_allow_html=True)
    football_df = df[df["Football_match"].notna() & (df["Football_match"] != "")].copy()
    if not football_df.empty:
        football_df = football_df[["Year", "Football_Month", "Football_date", "Football_match"]].drop_duplicates()
        st.dataframe(football_df.rename(columns={
            "Year": "ปี", "Football_Month": "เดือน",
            "Football_date": "วันที่", "Football_match": "คู่แข่งขัน"
        }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────
# ── TAB: เปรียบเทียบโมเดล ──
# ─────────────────────────────────────────────────────────
elif selected_tab == "🤖 เปรียบเทียบโมเดล":
    st.markdown("<div class='section-header'>🤖 เปรียบเทียบประสิทธิภาพโมเดล ML</div>", unsafe_allow_html=True)

    # Metrics table
    metrics_data = []
    for name, res in model_results.items():
        metrics_data.append({
            "โมเดล": name,
            "MAE": f"{res['MAE']:,.0f}",
            "RMSE": f"{res['RMSE']:,.0f}",
            "R²": f"{res['R2']:.4f}",
            "ทำนาย 2569": f"{predict_2569(res['model'], annual, event_settings):,.0f}",
            "🏆": "✅ ดีที่สุด" if name == best_name else ""
        })
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Radar chart
    fig_radar = go.Figure()
    for name, res in model_results.items():
        r2_norm = max(0, res["R2"])
        mae_norm = 1 - min(1, res["MAE"] / annual["Total_vis"].mean())
        rmse_norm = 1 - min(1, res["RMSE"] / annual["Total_vis"].mean())
        fig_radar.add_trace(go.Scatterpolar(
            r=[r2_norm, mae_norm, rmse_norm, r2_norm],
            theta=["R²", "1-MAE (norm)", "1-RMSE (norm)", "R²"],
            fill="toself", name=name
        ))
    fig_radar.update_layout(title="Radar Chart เปรียบเทียบโมเดล",
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            height=450)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Error distribution
    st.markdown("<div class='section-header'>📉 Residual Plot (ความคลาดเคลื่อน)</div>", unsafe_allow_html=True)

    sel_model_name = st.selectbox("เลือกโมเดล", list(model_results.keys()))
    residuals = annual["Total_vis"].values - model_results[sel_model_name]["y_pred"]

    fig_res = make_subplots(rows=1, cols=2,
                            subplot_titles=("Residuals over Years", "Distribution of Residuals"))
    fig_res.add_trace(go.Scatter(x=annual["Year"], y=residuals,
                                 mode="lines+markers", name="Residuals",
                                 line=dict(color="#e74c3c")), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_res.add_trace(go.Histogram(x=residuals, nbinsx=10,
                                   marker_color="#2d6a9f", name="Distribution"), row=1, col=2)
    fig_res.update_layout(height=380, showlegend=False,
                           plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance (RF/GB only)
    if sel_model_name in ["Random Forest", "Gradient Boosting"]:
        st.markdown("<div class='section-header'>📊 Feature Importance</div>", unsafe_allow_html=True)
        importances = model_results[sel_model_name]["model"].feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feat_cols, "Importance": importances
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                        title=f"Feature Importance — {sel_model_name}")
        fig_fi.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fi, use_container_width=True)
