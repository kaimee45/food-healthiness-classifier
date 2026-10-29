import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import norm, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Healthiness Classifier",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8eaf6;
}

/* ── Sidebar: force always visible, never collapse ─────────────────────── */
section[data-testid="stSidebar"] {
    transform: none !important;
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    display: flex !important;
    visibility: visible !important;
    background: linear-gradient(180deg, #1a1740 0%, #1e1b4b 100%) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.2) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 280px !important;
}
[data-testid="stSidebar"] * {
    color: #e8eaf6 !important;
}

/* Hide the collapse button — nav should always be visible */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="stBaseButton-headerNoPadding"] {
    display: none !important;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: linear-gradient(135deg, rgba(100,80,255,0.25), rgba(60,200,200,0.15));
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #b0bec5;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e0f7fa;
}

/* Section headers */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #80cbc4;
    border-left: 4px solid #26c6da;
    padding-left: 0.7rem;
    margin-bottom: 1rem;
}

/* Prediction result */
.pred-healthy {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a5d6a7;
    border: 1px solid #66bb6a;
}
.pred-unhealthy {
    background: linear-gradient(135deg, #b71c1c, #c62828);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffcdd2;
    border: 1px solid #ef5350;
}

/* Streamlit elements */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}
div[data-testid="stMetric"] label {
    color: #b0bec5 !important;
}
div[data-testid="stMetric"] div {
    color: #e0f7fa !important;
}

/* Slider + inputs */
[data-testid="stSlider"] > div > div > div {
    background: #26c6da !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #26c6da, #7c4dff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(38,198,218,0.4);
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px;
    padding: 4px;
}
[data-baseweb="tab"] {
    color: #b0bec5 !important;
    border-radius: 8px !important;
}
[aria-selected="true"] {
    background: rgba(38,198,218,0.2) !important;
    color: #26c6da !important;
    font-weight: 600 !important;
}

/* Plot backgrounds */
.stPlotlyChart, .stImage { border-radius: 12px; overflow: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(38,198,218,0.4); border-radius: 4px; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; background: transparent !important; }
header [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Matplotlib dark style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#3a3a5c",
    "axes.labelcolor":   "#b0bec5",
    "xtick.color":       "#b0bec5",
    "ytick.color":       "#b0bec5",
    "text.color":        "#e8eaf6",
    "grid.color":        "#2a2a4a",
    "grid.linewidth":    0.5,
    "axes.titlecolor":   "#80cbc4",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "font.family":       "sans-serif",
})
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#16213e",
    "figure.facecolor": "#1a1a2e",
    "text.color": "#e8eaf6",
    "axes.labelcolor": "#b0bec5",
    "xtick.color": "#b0bec5",
    "ytick.color": "#b0bec5",
    "grid.color": "#2a2a4a",
})

PALETTE = ["#26c6da", "#7c4dff", "#ff6f00", "#66bb6a", "#ef5350", "#ab47bc"]
# External fast-food items (unseen data — NOT in McDonald's training set)
EXTERNAL_FOODS = pd.DataFrame([
    {"Brand": "Burger King", "Food": "BK Whopper", "Calories": 657, "Protein": 28, "Fat": 40, "Saturated_Fat": 12, "Sugar": 11, "Sodium": 980},
    {"Brand": "Burger King", "Food": "BK Chicken Royale", "Calories": 570, "Protein": 25, "Fat": 32, "Saturated_Fat": 5, "Sugar": 8, "Sodium": 1050},
    {"Brand": "Burger King", "Food": "BK Veggie Burger", "Calories": 390, "Protein": 14, "Fat": 17, "Saturated_Fat": 3, "Sugar": 7, "Sodium": 680},
    {"Brand": "Burger King", "Food": "BK Onion Rings (Medium)", "Calories": 410, "Protein": 5, "Fat": 20, "Saturated_Fat": 4, "Sugar": 6, "Sodium": 570},
    {"Brand": "Burger King", "Food": "BK Chocolate Shake (Medium)", "Calories": 580, "Protein": 11, "Fat": 14, "Saturated_Fat": 9, "Sugar": 72, "Sodium": 390},
    {"Brand": "Subway", "Food": "Subway 6\" Veggie Delite", "Calories": 230, "Protein": 9, "Fat": 3, "Saturated_Fat": 1, "Sugar": 4, "Sodium": 310},
    {"Brand": "Subway", "Food": "Subway 6\" Turkey Breast", "Calories": 280, "Protein": 18, "Fat": 4, "Saturated_Fat": 1, "Sugar": 5, "Sodium": 700},
    {"Brand": "Subway", "Food": "Subway 6\" Chicken Teriyaki", "Calories": 370, "Protein": 26, "Fat": 5, "Saturated_Fat": 2, "Sugar": 14, "Sodium": 810},
    {"Brand": "Subway", "Food": "Subway 6\" Tuna Sub", "Calories": 480, "Protein": 19, "Fat": 25, "Saturated_Fat": 5, "Sugar": 5, "Sodium": 640},
    {"Brand": "Subway", "Food": "Subway Chocolate Chip Cookie", "Calories": 220, "Protein": 2, "Fat": 10, "Saturated_Fat": 5, "Sugar": 18, "Sodium": 160},
    {"Brand": "KFC", "Food": "KFC Original Recipe Chicken Breast", "Calories": 390, "Protein": 39, "Fat": 21, "Saturated_Fat": 5, "Sugar": 0, "Sodium": 1190},
    {"Brand": "KFC", "Food": "KFC Popcorn Chicken (Regular)", "Calories": 400, "Protein": 21, "Fat": 24, "Saturated_Fat": 4, "Sugar": 1, "Sodium": 1160},
    {"Brand": "KFC", "Food": "KFC Zinger Burger", "Calories": 450, "Protein": 22, "Fat": 21, "Saturated_Fat": 4, "Sugar": 6, "Sodium": 1050},
    {"Brand": "KFC", "Food": "KFC Coleslaw (Individual)", "Calories": 170, "Protein": 1, "Fat": 12, "Saturated_Fat": 2, "Sugar": 10, "Sodium": 180},
    {"Brand": "KFC", "Food": "KFC Mashed Potatoes & Gravy", "Calories": 120, "Protein": 2, "Fat": 4, "Saturated_Fat": 1, "Sugar": 0, "Sodium": 530},
    {"Brand": "Domino's", "Food": "Domino's Margherita (2 slices)", "Calories": 420, "Protein": 16, "Fat": 16, "Saturated_Fat": 7, "Sugar": 6, "Sodium": 900},
    {"Brand": "Domino's", "Food": "Domino's Pepperoni (2 slices)", "Calories": 500, "Protein": 20, "Fat": 22, "Saturated_Fat": 10, "Sugar": 6, "Sodium": 1100},
    {"Brand": "Domino's", "Food": "Domino's Veggie Paradise (2 sl)", "Calories": 380, "Protein": 14, "Fat": 13, "Saturated_Fat": 5, "Sugar": 5, "Sodium": 750},
    {"Brand": "Domino's", "Food": "Domino's Garlic Breadsticks", "Calories": 330, "Protein": 8, "Fat": 14, "Saturated_Fat": 4, "Sugar": 2, "Sodium": 560},
    {"Brand": "Domino's", "Food": "Domino's Choco Lava Cake", "Calories": 350, "Protein": 4, "Fat": 14, "Saturated_Fat": 6, "Sugar": 38, "Sodium": 280},
    {"Brand": "Pizza Hut", "Food": "Pizza Hut Veg Supreme (2 sl)", "Calories": 440, "Protein": 16, "Fat": 18, "Saturated_Fat": 7, "Sugar": 6, "Sodium": 920},
    {"Brand": "Pizza Hut", "Food": "Pizza Hut Chicken Tikka (2 sl)", "Calories": 480, "Protein": 22, "Fat": 18, "Saturated_Fat": 8, "Sugar": 5, "Sodium": 1000},
    {"Brand": "Pizza Hut", "Food": "Pizza Hut Garlic Bread", "Calories": 310, "Protein": 6, "Fat": 14, "Saturated_Fat": 4, "Sugar": 3, "Sodium": 480},
    {"Brand": "Pizza Hut", "Food": "Pizza Hut Tandoori Paneer (2 sl)", "Calories": 460, "Protein": 18, "Fat": 20, "Saturated_Fat": 9, "Sugar": 4, "Sodium": 880},
    {"Brand": "Taco Bell", "Food": "Taco Bell Crunchy Taco", "Calories": 170, "Protein": 8, "Fat": 10, "Saturated_Fat": 4, "Sugar": 1, "Sodium": 310},
    {"Brand": "Taco Bell", "Food": "Taco Bell Bean Burrito", "Calories": 380, "Protein": 14, "Fat": 10, "Saturated_Fat": 4, "Sugar": 4, "Sodium": 1060},
    {"Brand": "Taco Bell", "Food": "Taco Bell Crunchwrap Supreme", "Calories": 540, "Protein": 16, "Fat": 23, "Saturated_Fat": 8, "Sugar": 6, "Sodium": 1200},
    {"Brand": "Taco Bell", "Food": "Taco Bell Nachos BellGrande", "Calories": 740, "Protein": 16, "Fat": 38, "Saturated_Fat": 7, "Sugar": 4, "Sodium": 1050},
    {"Brand": "Wendy's", "Food": "Wendy's Grilled Chicken Wrap", "Calories": 270, "Protein": 20, "Fat": 10, "Saturated_Fat": 4, "Sugar": 2, "Sodium": 730},
    {"Brand": "Wendy's", "Food": "Wendy's Baconator", "Calories": 940, "Protein": 57, "Fat": 66, "Saturated_Fat": 30, "Sugar": 9, "Sodium": 1650},
    {"Brand": "Wendy's", "Food": "Wendy's Jr. Cheeseburger", "Calories": 290, "Protein": 16, "Fat": 14, "Saturated_Fat": 6, "Sugar": 6, "Sodium": 620},
    {"Brand": "Chick-fil-A", "Food": "Chick-fil-A Grilled Nuggets (8pc)", "Calories": 140, "Protein": 25, "Fat": 3, "Saturated_Fat": 1, "Sugar": 1, "Sodium": 440},
    {"Brand": "Chick-fil-A", "Food": "Chick-fil-A Spicy Chicken Sandwich", "Calories": 450, "Protein": 28, "Fat": 19, "Saturated_Fat": 4, "Sugar": 6, "Sodium": 1620},
])



def get_dataset_key(source_path=None):
    if source_path and os.path.exists(source_path):
        return f"{source_path}_{os.path.getmtime(source_path)}_{os.path.getsize(source_path)}"
    if os.path.exists("Menu.csv"):
        return f"Menu.csv_{os.path.getmtime('Menu.csv')}_{os.path.getsize('Menu.csv')}"
    if os.path.exists("India_Menu.csv"):
        return f"India_Menu.csv_{os.path.getmtime('India_Menu.csv')}_{os.path.getsize('India_Menu.csv')}"
    return "online_github_url"


# ─── Data Loading & Processing ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(dataset_key: str = "", filepath: str = "", uploaded_bytes: bytes = None):
    """Load the menu dataset (local file, uploaded file, or online fallback), clean data, handle outliers, and compute health scores and labels."""
    df = None
    if uploaded_bytes is not None:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    elif filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    elif os.path.exists("Menu.csv"):
        df = pd.read_csv("Menu.csv")
    elif os.path.exists("India_Menu.csv"):
        df = pd.read_csv("India_Menu.csv")
    else:
        primary_url = (
            "https://raw.githubusercontent.com/Garvitjoshi1/"
            "McDonalds-India-Menu-Nutrition-Dataset/refs/heads/main/India_Menu.csv"
        )
        df = pd.read_csv(primary_url)

    # Standardise column names
    df = df.rename(columns={
        "Menu Category": "Category",
        "Menu Items": "Food",
        "Per Serve Size": "Serving_Size",
        "Energy (kCal)": "Calories",
        "Protein (g)": "Protein",
        "Total fat (g)": "Fat",
        "Sat Fat (g)": "Saturated_Fat",
        "Total Sugars (g)": "Sugar",
        "Sodium (mg)": "Sodium",
    })

    # Define numeric features
    features = ["Calories", "Protein", "Fat", "Saturated_Fat", "Sugar", "Sodium"]
    # Ensure category, food, serving size exist
    if "Category" not in df.columns:
        df["Category"] = "General Menu"
    if "Food" not in df.columns:
        df["Food"] = [f"Item {i+1}" for i in range(len(df))]
    if "Serving_Size" not in df.columns:
        df["Serving_Size"] = "Standard"

    # Keep only required columns
    food = df[["Category", "Food", "Serving_Size"] + features].copy()
    # Coerce to numeric (some rows may contain non‑numeric strings)
    for col in features:
        food[col] = pd.to_numeric(food[col], errors="coerce")
    # Drop rows with missing feature values and duplicates
    food = food.dropna(subset=features).drop_duplicates().reset_index(drop=True)

    # ── Outlier handling (IQR Winsorisation) for Calories & Sodium ──────────
    outlier_cols = ["Calories", "Sodium"]
    outlier_info = []
    for col in outlier_cols:
        Q1 = food[col].quantile(0.25)
        Q3 = food[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_low = int((food[col] < lower).sum())
        n_high = int((food[col] > upper).sum())
        # Preserve original for visualisation
        food[col + "_raw"] = food[col].copy()
        # Cap values
        food[col] = food[col].clip(lower=lower, upper=upper)
        outlier_info.append({
            "Feature": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "Lower Fence": round(lower, 2),
            "Upper Fence": round(upper, 2),
            "Capped Low": n_low,
            "Capped High": n_high,
            "Total Capped": n_low + n_high,
        })
    outlier_df = pd.DataFrame(outlier_info)

    # ── Health scoring and label ───────────────────────────────────────────────
    cond = pd.DataFrame({
        "Calories": food["Calories"] <= 400,
        "Fat": food["Fat"] <= 15,
        "Sat_Fat": food["Saturated_Fat"] <= 5,
        "Sugar": food["Sugar"] <= 10,
        "Sodium": food["Sodium"] <= 600,
        "Protein": food["Protein"] >= 8,
    })
    food["Health_Score"] = cond.sum(axis=1)
    food["Healthy"]      = np.where(food["Health_Score"] >= 4, 1, 0)
    food["Health_Label"] = np.where(food["Healthy"] == 1, "Healthy", "Unhealthy")

    return food, features, outlier_df


@st.cache_resource(show_spinner=False)
def train_models(_food, _features, dataset_key: str = ""):
    X = _food[_features]
    y = _food["Healthy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    return lr, dt, X_train, X_test, y_train, y_test, lr_pred, dt_pred


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥗 Food Health AI")
    st.markdown("**SI Lab Final Project**")
    st.markdown("---")

    # Dataset Selection & Management
    st.markdown("### 📁 Dataset Source")

    available_sources = []
    if os.path.exists("Menu.csv"):
        available_sources.append("Menu.csv (New / Uploaded Dataset)")
    if os.path.exists("India_Menu.csv"):
        available_sources.append("India_Menu.csv (Original Dataset)")
    available_sources.append("Online GitHub URL")

    selected_source_label = st.selectbox(
        "Active Dataset",
        available_sources,
        index=0,
        help="Choose the dataset to analyze. Defaults to Menu.csv if present."
    )

    uploaded_file = st.file_uploader(
        "Upload Custom CSV",
        type=["csv"],
        help="Upload a new fast-food or nutritional menu dataset"
    )

    if st.button("🔄 Clear Cache & Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Determine file key and source
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        dataset_key = f"upload_{uploaded_file.name}_{len(raw_bytes)}"
        source_display = f"Uploaded: {uploaded_file.name}"
        active_filepath = ""
        active_bytes = raw_bytes
    elif "Menu.csv" in selected_source_label and os.path.exists("Menu.csv"):
        dataset_key = get_dataset_key("Menu.csv")
        source_display = "Menu.csv"
        active_filepath = "Menu.csv"
        active_bytes = None
    elif "India_Menu.csv" in selected_source_label and os.path.exists("India_Menu.csv"):
        dataset_key = get_dataset_key("India_Menu.csv")
        source_display = "India_Menu.csv"
        active_filepath = "India_Menu.csv"
        active_bytes = None
    else:
        dataset_key = "online_github_url"
        source_display = "Online GitHub Dataset"
        active_filepath = ""
        active_bytes = None

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Data Explorer",
         "📈 Visualizations",
         "🔬 Statistical Analysis",
         "⚖️ Class Imbalance",
         "🤖 Model Training",
         "🎯 Live Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Kaivalya Maske**  \nB2-B3-48 · AIDS")


# ─── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    food, features, outlier_df = load_and_process(dataset_key, active_filepath, active_bytes)

with st.spinner("Training models…"):
    lr_model, dt_model, X_train, X_test, y_train, y_test, lr_pred, dt_pred = train_models(
        food, features, dataset_key
    )

with st.sidebar:
    st.success(f"📊 **{len(food):,} items** loaded\n\n(`{source_display}`)")


# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <div style='font-size:3.5rem;'>🥗</div>
        <h1 style='font-size:2.6rem; font-weight:700; color:#80cbc4; margin:0;'>
            Food Healthiness Classifier
        </h1>
        <p style='color:#b0bec5; font-size:1.1rem; margin-top:0.4rem;'>
            Statistical Inference Lab · Final Project
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    healthy_n   = int(food["Healthy"].sum())
    unhealthy_n = int(len(food) - healthy_n)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Items",   len(food))
    col2.metric("✅ Healthy",       healthy_n)
    col3.metric("❌ Unhealthy",     unhealthy_n)
    col4.metric("🔢 Features Used", len(features))

    st.markdown("<br>", unsafe_allow_html=True)

    # Two info cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class='card'>
            <b style='color:#26c6da;'>📌 Objective</b><br><br>
            Classify fast-food menu items (<b>{len(food):,} items</b> from <b>{source_display}</b>) as <b>Healthy</b> or <b>Unhealthy</b>
            using nutritional data, statistical inference techniques, and ML models.
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class='card'>
            <b style='color:#7c4dff;'>🧪 SI Concepts Used</b><br><br>
            Descriptive Statistics · Data Visualization · Normal Distribution ·
            Central Limit Theorem · Bootstrap Estimation
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <b style='color:#26c6da;'>⚕️ Health Score Rules</b><br><br>
        Each food item scores <b>+1</b> for each condition it meets:<br><br>
        <table style='width:100%; color:#e8eaf6; border-collapse:collapse;'>
        <tr><td>Calories ≤ 400 kcal</td><td>Fat ≤ 15 g</td><td>Saturated Fat ≤ 5 g</td></tr>
        <tr><td>Sugar ≤ 10 g</td><td>Sodium ≤ 600 mg</td><td>Protein ≥ 8 g</td></tr>
        </table><br>
        <b>Score 4–6 → 🟢 Healthy &nbsp;&nbsp;&nbsp; Score 0–3 → 🔴 Unhealthy</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='card'><b style='color:#ff6f00;'>⚠️ Disclaimer</b> – "
                "Healthy/Unhealthy is an academic classification for this project, "
                "not medical or dietary advice.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("<div class='section-title'>📊 Data Explorer</div>", unsafe_allow_html=True)

    tabs = st.tabs(["🗂 Raw Data", "📐 Descriptive Stats", "🔍 Filter & Search"])

    with tabs[0]:
        st.caption(f"Showing {len(food)} cleaned records")
        label_filter = st.selectbox("Filter by label", ["All", "Healthy", "Unhealthy"])
        view = food if label_filter == "All" else food[food["Health_Label"] == label_filter]
        st.dataframe(
            view[["Food", "Category"] + features + ["Health_Score", "Health_Label"]],
            use_container_width=True, height=420
        )

    with tabs[1]:
        desc = food[features].describe().T
        desc["median"]   = food[features].median()
        desc["variance"] = food[features].var()
        desc["IQR"]      = food[features].quantile(0.75) - food[features].quantile(0.25)
        st.dataframe(
            desc[["count","mean","median","std","variance","min","25%","50%","75%","max","IQR"]].round(2),
            use_container_width=True
        )

    with tabs[2]:
        col_a, col_b = st.columns(2)
        with col_a:
            feat = st.selectbox("Sort by", features)
        with col_b:
            order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True)
        asc = (order == "Lowest first")
        st.dataframe(
            food[["Food", "Category"] + features + ["Health_Label"]]
            .sort_values(feat, ascending=asc),
            use_container_width=True, height=400
        )


# ══════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown("<div class='section-title'>📈 Visualizations</div>", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Distributions", "📦 Boxplots", "🔵 Scatter", "🌡️ Heatmap", "🚨 Outliers"])

    with tabs[0]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes = axes.flatten()
        for i, f in enumerate(features):
            axes[i].hist(food[f], bins=15, color=PALETTE[i], alpha=0.85, edgecolor="#0f0c29")
            axes[i].set_title(f)
            axes[i].set_xlabel(f)
            axes[i].set_ylabel("Count")
        fig.suptitle("Distribution of Nutritional Features", fontsize=15, color="#80cbc4")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(12, 5))
        food_melted = food[features].melt(var_name="Nutrient", value_name="Value")
        sns.boxplot(data=food_melted, x="Nutrient", y="Value",
                    palette=PALETTE, ax=ax, linewidth=1.2)
        ax.set_title("Boxplots of Nutritional Features")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[2]:
        col_a, col_b = st.columns(2)
        with col_a:
            x_feat = st.selectbox("X axis", features, index=0)
        with col_b:
            y_feat = st.selectbox("Y axis", features, index=5)
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, color in [("Healthy","#66bb6a"), ("Unhealthy","#ef5350")]:
            sub = food[food["Health_Label"] == label]
            ax.scatter(sub[x_feat], sub[y_feat], label=label,
                       color=color, alpha=0.75, s=60, edgecolors="#0f0c29", linewidth=0.5)
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_title(f"{x_feat} vs {y_feat}")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[3]:
        corr = food[features].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.zeros_like(corr, dtype=bool)
        np.fill_diagonal(mask, False)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, square=True, ax=ax,
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Correlation Heatmap of Nutritional Features")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📖 How to read this?"):
            st.markdown("""
            - **+1 (dark red)** → strong positive correlation (both features rise together)
            - **−1 (dark blue)** → strong negative correlation (one rises, other falls)
            - **~0 (white)** → little to no linear relationship
            """)

    # ── Outliers tab ────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### 🚨 Outlier Detection & Handling — Calories & Sodium")
        st.markdown(
            "Outliers are detected using the **IQR (Interquartile Range)** method. "
            "Any value below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` is **capped (Winsorized)** "
            "to the fence value before model training."
        )

        # Stats table
        st.markdown("#### 📋 Outlier Summary")
        st.dataframe(
            outlier_df.set_index("Feature"),
            use_container_width=True
        )

        # Before / After boxplots side by side for each feature
        for col in ["Calories", "Sodium"]:
            raw_col = col + "_raw"
            units   = "kcal" if col == "Calories" else "mg"

            row_info = outlier_df[outlier_df["Feature"] == col].iloc[0]
            total_capped = int(row_info["Total Capped"])

            st.markdown(f"#### 🔍 {col} — {total_capped} value(s) capped")

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

            # Before
            axes[0].boxplot(food[raw_col].dropna(), vert=False,
                            patch_artist=True,
                            boxprops=dict(facecolor="#7c4dff", color="#e8eaf6"),
                            medianprops=dict(color="#26c6da", linewidth=2),
                            whiskerprops=dict(color="#b0bec5"),
                            capprops=dict(color="#b0bec5"),
                            flierprops=dict(marker="o", color="#ef5350",
                                           markerfacecolor="#ef5350", markersize=7))
            axes[0].set_title(f"{col} — Before Outlier Handling", color="#ef5350")
            axes[0].set_xlabel(units)

            # After
            axes[1].boxplot(food[col].dropna(), vert=False,
                            patch_artist=True,
                            boxprops=dict(facecolor="#26c6da", color="#e8eaf6"),
                            medianprops=dict(color="#ff6f00", linewidth=2),
                            whiskerprops=dict(color="#b0bec5"),
                            capprops=dict(color="#b0bec5"),
                            flierprops=dict(marker="o", color="#66bb6a",
                                           markerfacecolor="#66bb6a", markersize=7))
            axes[1].set_title(f"{col} — After Outlier Capping (IQR)", color="#66bb6a")
            axes[1].set_xlabel(units)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("""<div class='card'>
            <b style='color:#26c6da;'>📌 Why IQR Winsorization?</b><br><br>
            Instead of <b>removing</b> outlier rows (losing data), we <b>clip</b> them to the fence.
            This keeps all food items in the dataset while preventing extreme values from
            skewing the model's learned weights — especially important for
            <b>Calories</b> (wide range across beverages vs meals) and
            <b>Sodium</b> (some items are extremely high in salt).
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Analysis":
    st.markdown("<div class='section-title'>🔬 Statistical Analysis</div>", unsafe_allow_html=True)

    tabs = st.tabs(["📐 Normal Distribution", "📊 CLT", "🥾 Bootstrap", "🔲 Chi-Square Test"])

    # ── Normal Distribution
    with tabs[0]:
        feat_nd = st.selectbox("Select feature", features, key="nd_feat")
        data    = food[feat_nd].dropna()
        mu, sigma = data.mean(), data.std()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(data, bins=15, density=True, alpha=0.65,
                color="#26c6da", edgecolor="#0f0c29", label=f"Actual {feat_nd}")
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, norm.pdf(x, mu, sigma), color="#ff6f00", linewidth=2.5,
                label="Normal Distribution")
        ax.axvline(mu, color="#7c4dff", linestyle="--", linewidth=1.6, label=f"Mean={mu:.1f}")
        ax.set_xlabel(feat_nd)
        ax.set_ylabel("Density")
        ax.set_title(f"{feat_nd}: Actual vs Normal Distribution")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        c1, c2 = st.columns(2)
        c1.metric("Mean (μ)", f"{mu:.2f}")
        c2.metric("Std Dev (σ)", f"{sigma:.2f}")

    # ── CLT
    with tabs[1]:
        n_sims = st.slider("Number of simulations", 500, 5000, 2000, 500, key="clt_sims")
        feat_clt = st.selectbox("Select feature", features, key="clt_feat")
        population  = food[feat_clt].dropna().values
        pop_mean    = np.mean(population)
        sample_sizes = [10, 30, 50]

        fig, ax = plt.subplots(figsize=(10, 5))
        clt_rows = []
        for i, n in enumerate(sample_sizes):
            means = [np.mean(np.random.choice(population, size=n, replace=True))
                     for _ in range(n_sims)]
            means = np.array(means)
            ax.hist(means, bins=30, alpha=0.45, color=PALETTE[i], label=f"n={n}")
            clt_rows.append({
                "Sample Size": n,
                "Mean of Means": round(np.mean(means), 2),
                "Std of Means":  round(np.std(means),  2)
            })
        ax.axvline(pop_mean, color="#ff6f00", linestyle="--", linewidth=2,
                   label=f"Population Mean = {pop_mean:.1f}")
        ax.set_xlabel(f"Sample Mean ({feat_clt})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"CLT: Sampling Distribution of Mean {feat_clt}")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(pd.DataFrame(clt_rows), use_container_width=True)

    # ── Bootstrap
    with tabs[2]:
        feat_bs   = st.selectbox("Select feature", features, key="bs_feat")
        samp_size = st.slider("Sample size", 20, 80, 50, key="bs_n")
        n_boot    = st.slider("Bootstrap iterations", 1000, 5000, 3000, 500, key="bs_iter")

        np.random.seed(42)
        population = food[feat_bs].dropna().values
        pop_mean   = np.mean(population)
        sample     = np.random.choice(population, size=samp_size, replace=False)
        boot_means = np.array([
            np.mean(np.random.choice(sample, size=samp_size, replace=True))
            for _ in range(n_boot)
        ])
        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(boot_means, bins=35, color="#7c4dff", alpha=0.75, edgecolor="#0f0c29",
                label="Bootstrap Means")
        ax.axvline(pop_mean, color="#ff6f00", linestyle="--", linewidth=2,
                   label=f"Population Mean = {pop_mean:.1f}")
        ax.axvline(lower, color="#66bb6a", linestyle="--", linewidth=1.8,
                   label=f"2.5% = {lower:.1f}")
        ax.axvline(upper, color="#66bb6a", linestyle="--", linewidth=1.8,
                   label=f"97.5% = {upper:.1f}")
        ax.set_xlabel(f"Bootstrap Mean ({feat_bs})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Bootstrap Distribution of Mean {feat_bs}")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sample Mean",       f"{np.mean(sample):.2f}")
        c2.metric("Bootstrap Estimate",f"{np.mean(boot_means):.2f}")
        c3.metric("Population Mean",   f"{pop_mean:.2f}")
        c4.metric("95% CI",            f"[{lower:.1f}, {upper:.1f}]")

        inside = lower <= pop_mean <= upper
        if inside:
            st.success("✅ Population mean falls **inside** the 95% bootstrap interval.")
        else:
            st.warning("⚠️ Population mean falls **outside** the 95% bootstrap interval.")

    # ── Chi-Square Test of Independence
    with tabs[3]:
        st.markdown("### 🔲 Chi-Square Test of Independence")
        st.markdown(
            "**Research Question:** Is there a statistically significant association between "
            "**Menu Category** and **Health Label** (Healthy / Unhealthy)?  \n"
            "*Example: Are McCafe items healthier than Regular Menu items?*"
        )

        # ── Contingency table
        ct = pd.crosstab(food["Category"], food["Health_Label"])
        # Ensure both columns exist
        for col in ["Healthy", "Unhealthy"]:
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[["Healthy", "Unhealthy"]]
        ct["Total"]        = ct["Healthy"] + ct["Unhealthy"]
        ct["% Healthy"]    = (ct["Healthy"] / ct["Total"] * 100).round(1)
        ct["% Unhealthy"]  = (ct["Unhealthy"] / ct["Total"] * 100).round(1)

        st.markdown("#### 📋 Contingency Table")
        st.dataframe(ct, use_container_width=True)

        # ── Run chi-square test
        chi2_obs = ct[["Healthy", "Unhealthy"]]
        chi2_stat, p_val, dof, expected = chi2_contingency(chi2_obs)
        expected_df = pd.DataFrame(
            expected.round(2),
            index=ct.index,
            columns=["Expected Healthy", "Expected Unhealthy"]
        )

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("χ² Statistic",  f"{chi2_stat:.4f}")
        k2.metric("p-value",        f"{p_val:.4f}")
        k3.metric("Degrees of Freedom", dof)
        k4.metric("Significant?",   "✅ Yes" if p_val < 0.05 else "❌ No")

        # ── Bar chart: % Healthy per category
        st.markdown("#### 📊 Healthy % by Menu Category")
        cat_order = ct["% Healthy"].sort_values(ascending=True).index
        fig, ax = plt.subplots(figsize=(10, max(4, len(cat_order) * 0.55)))
        colors = [
            "#66bb6a" if v >= 50 else "#ef5350"
            for v in ct.loc[cat_order, "% Healthy"]
        ]
        bars = ax.barh(
            cat_order,
            ct.loc[cat_order, "% Healthy"],
            color=colors, edgecolor="#0f0c29", alpha=0.88
        )
        ax.axvline(50, color="#ff6f00", linestyle="--", linewidth=1.5, label="50% threshold")
        for bar, val in zip(bars, ct.loc[cat_order, "% Healthy"]):
            ax.text(
                val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9, fontweight="bold"
            )
        ax.set_xlabel("% Healthy items")
        ax.set_title("Healthiness by Menu Category")
        ax.set_xlim(0, 115)
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Standardised residuals heatmap
        st.markdown("#### 🌡️ Standardised Residuals Heatmap")
        st.caption(
            "Values > +2 (green) → category has **more** healthy items than expected.  "
            "Values < −2 (red) → category has **fewer** healthy items than expected."
        )
        obs    = chi2_obs.values.astype(float)
        std_res = (obs - expected) / np.sqrt(expected)
        std_res_df = pd.DataFrame(
            std_res.round(3),
            index=ct.index,
            columns=["Healthy", "Unhealthy"]
        )
        fig2, ax2 = plt.subplots(figsize=(6, max(3, len(ct) * 0.45)))
        sns.heatmap(
            std_res_df, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            ax=ax2, cbar_kws={"label": "Std. Residual"},
            annot_kws={"size": 10, "weight": "bold"}
        )
        ax2.set_title("Standardised Residuals (Observed − Expected) / √Expected")
        ax2.set_xlabel("Health Label")
        ax2.set_ylabel("Category")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Expected counts table
        with st.expander("📐 View Expected Counts"):
            st.dataframe(expected_df, use_container_width=True)

        # ── Interpretation card
        alpha = 0.05
        if p_val < alpha:
            conclusion = (
                f"Since **p = {p_val:.4f} < α = {alpha}**, we **reject H₀**. "
                "There is a **statistically significant association** between Menu Category "
                "and Health Label — some categories are significantly healthier than others."
            )
            color = "#26c6da"
        else:
            conclusion = (
                f"Since **p = {p_val:.4f} ≥ α = {alpha}**, we **fail to reject H₀**. "
                "There is **no statistically significant association** between Menu Category "
                "and Health Label at the 5% significance level."
            )
            color = "#ff6f00"

        st.markdown(f"""<div class='card'>
            <b style='color:{color};'>📌 Statistical Interpretation</b><br><br>
            <b>H₀</b>: Menu Category and Health Label are independent.<br>
            <b>H₁</b>: Menu Category and Health Label are associated.<br><br>
            {conclusion}<br><br>
            χ²({dof}) = {chi2_stat:.4f}, p = {p_val:.4f}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: CLASS IMBALANCE
# ══════════════════════════════════════════════════════════════
elif page == "⚖️ Class Imbalance":
    st.markdown("<div class='section-title'>⚖️ Class Imbalance</div>", unsafe_allow_html=True)

    counts = food["Health_Label"].value_counts()
    ratio  = round(counts.max() / counts.min(), 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Healthy",         int(counts.get("Healthy",   0)))
    c2.metric("❌ Unhealthy",       int(counts.get("Unhealthy", 0)))
    c3.metric("📐 Imbalance Ratio", f"{ratio}×")

    if ratio > 1.5:
        st.warning(f"⚠️ Class imbalance detected (ratio {ratio}×). Both models use "
                   "`class_weight='balanced'` to compensate.")
    else:
        st.success("✅ Classes are roughly balanced.")

    col_map = {"Healthy": "#66bb6a", "Unhealthy": "#ef5350"}
    pie_colors = [col_map[l] for l in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar
    bars = axes[0].bar(counts.index, counts.values,
                       color=[col_map[l] for l in counts.index],
                       edgecolor="#0f0c29", linewidth=0.6)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Healthy vs Unhealthy – Count")
    axes[0].set_ylabel("Count")

    # Pie
    axes[1].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=pie_colors, startangle=90,
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2}
    )
    axes[1].set_title("Class Proportion")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""<div class='card'>
        <b style='color:#26c6da;'>How is imbalance handled?</b><br><br>
        Both <b>Logistic Regression</b> and <b>Decision Tree</b> are trained with
        <code>class_weight='balanced'</code>. This makes sklearn automatically compute
        weights inversely proportional to class frequency — minority class items
        receive a higher weight during training so the model doesn't ignore them.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("<div class='section-title'>🤖 Model Training & Evaluation</div>",
                unsafe_allow_html=True)

    tabs = st.tabs(["📋 Comparison", "🌳 Decision Tree", "🔢 Confusion Matrices",
                    "📊 Feature Importance"])

    with tabs[0]:
        results = pd.DataFrame({
            "Model":     ["Logistic Regression", "Decision Tree"],
            "Accuracy":  [accuracy_score(y_test, lr_pred),
                          accuracy_score(y_test, dt_pred)],
            "Precision": [precision_score(y_test, lr_pred, zero_division=0),
                          precision_score(y_test, dt_pred, zero_division=0)],
            "Recall":    [recall_score(y_test, lr_pred, zero_division=0),
                          recall_score(y_test, dt_pred, zero_division=0)],
            "F1 Score":  [f1_score(y_test, lr_pred, zero_division=0),
                          f1_score(y_test, dt_pred, zero_division=0)],
        })
        st.dataframe(results.round(3).set_index("Model"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        x = np.arange(len(metrics))
        w = 0.3
        ax.bar(x - w/2, results.iloc[0][metrics], w,
               label="Logistic Regression", color="#26c6da", alpha=0.85)
        ax.bar(x + w/2, results.iloc[1][metrics], w,
               label="Decision Tree",        color="#7c4dff", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[1]:
        st.caption("Decision Tree (max_depth=4, class_weight='balanced')")
        fig, ax = plt.subplots(figsize=(18, 8))
        plot_tree(dt_model, feature_names=features,
                  class_names=["Unhealthy", "Healthy"],
                  filled=True, rounded=True, ax=ax, fontsize=8)
        ax.set_title("Decision Tree for Food Healthiness", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[2]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (preds, title) in zip(axes, [
            (lr_pred, "Logistic Regression"),
            (dt_pred, "Decision Tree")
        ]):
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Unhealthy","Healthy"],
                        yticklabels=["Unhealthy","Healthy"],
                        ax=ax, cbar=False,
                        annot_kws={"size": 14, "weight": "bold"})
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[3]:
        importances = dt_model.feature_importances_
        feat_imp    = pd.Series(importances, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(feat_imp.index, feat_imp.values,
                       color=PALETTE[:len(features)], alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title("Decision Tree Feature Importances")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════

# Use the global EXTERNAL_FOODS defined earlier for live predictor.


        # External foods are defined globally above.
        EXTERNAL_FOODS = pd.DataFrame([

        {"Brand": "Burger King", "Food": "BK Whopper",                  "Calories": 657, "Protein": 28, "Fat": 40, "Saturated_Fat": 12, "Sugar": 11, "Sodium": 980},
        {"Brand": "Burger King", "Food": "BK Chicken Royale",           "Calories": 570, "Protein": 25, "Fat": 32, "Saturated_Fat": 5,  "Sugar": 8,  "Sodium": 1050},
        {"Brand": "Burger King", "Food": "BK Veggie Burger",            "Calories": 390, "Protein": 14, "Fat": 17, "Saturated_Fat": 3,  "Sugar": 7,  "Sodium": 680},
        {"Brand": "Burger King", "Food": "BK Onion Rings (Medium)",     "Calories": 410, "Protein": 5,  "Fat": 20, "Saturated_Fat": 4,  "Sugar": 6,  "Sodium": 570},
        {"Brand": "Burger King", "Food": "BK Chocolate Shake (Medium)", "Calories": 580, "Protein": 11, "Fat": 14, "Saturated_Fat": 9,  "Sugar": 72, "Sodium": 390},
        # ── Subway ───────────────────────────────────────────────
        {"Brand": "Subway", "Food": "Subway 6\" Veggie Delite",        "Calories": 230, "Protein": 9,  "Fat": 3,  "Saturated_Fat": 1,  "Sugar": 4,  "Sodium": 310},
        {"Brand": "Subway", "Food": "Subway 6\" Turkey Breast",        "Calories": 280, "Protein": 18, "Fat": 4,  "Saturated_Fat": 1,  "Sugar": 5,  "Sodium": 700},
        {"Brand": "Subway", "Food": "Subway 6\" Chicken Teriyaki",     "Calories": 370, "Protein": 26, "Fat": 5,  "Saturated_Fat": 2,  "Sugar": 14, "Sodium": 810},
        {"Brand": "Subway", "Food": "Subway 6\" Tuna Sub",             "Calories": 480, "Protein": 19, "Fat": 25, "Saturated_Fat": 5,  "Sugar": 5,  "Sodium": 640},
        {"Brand": "Subway", "Food": "Subway Chocolate Chip Cookie",     "Calories": 220, "Protein": 2,  "Fat": 10, "Saturated_Fat": 5,  "Sugar": 18, "Sodium": 160},
        # ── KFC ──────────────────────────────────────────────────
        {"Brand": "KFC", "Food": "KFC Original Recipe Chicken Breast",  "Calories": 390, "Protein": 39, "Fat": 21, "Saturated_Fat": 5,  "Sugar": 0,  "Sodium": 1190},
        {"Brand": "KFC", "Food": "KFC Popcorn Chicken (Regular)",       "Calories": 400, "Protein": 21, "Fat": 24, "Saturated_Fat": 4,  "Sugar": 1,  "Sodium": 1160},
        {"Brand": "KFC", "Food": "KFC Zinger Burger",                   "Calories": 450, "Protein": 22, "Fat": 21, "Saturated_Fat": 4,  "Sugar": 6,  "Sodium": 1050},
        {"Brand": "KFC", "Food": "KFC Coleslaw (Individual)",           "Calories": 170, "Protein": 1,  "Fat": 12, "Saturated_Fat": 2,  "Sugar": 10, "Sodium": 180},
        {"Brand": "KFC", "Food": "KFC Mashed Potatoes & Gravy",        "Calories": 120, "Protein": 2,  "Fat": 4,  "Saturated_Fat": 1,  "Sugar": 0,  "Sodium": 530},
        # ── Domino's ─────────────────────────────────────────────
        {"Brand": "Domino's", "Food": "Domino's Margherita (2 slices)", "Calories": 420, "Protein": 16, "Fat": 16, "Saturated_Fat": 7,  "Sugar": 6,  "Sodium": 900},
        {"Brand": "Domino's", "Food": "Domino's Pepperoni (2 slices)",  "Calories": 500, "Protein": 20, "Fat": 22, "Saturated_Fat": 10, "Sugar": 6,  "Sodium": 1100},
        {"Brand": "Domino's", "Food": "Domino's Veggie Paradise (2 sl)","Calories": 380, "Protein": 14, "Fat": 13, "Saturated_Fat": 5,  "Sugar": 5,  "Sodium": 750},
        {"Brand": "Domino's", "Food": "Domino's Garlic Breadsticks",    "Calories": 330, "Protein": 8,  "Fat": 14, "Saturated_Fat": 4,  "Sugar": 2,  "Sodium": 560},
        {"Brand": "Domino's", "Food": "Domino's Choco Lava Cake",       "Calories": 350, "Protein": 4,  "Fat": 14, "Saturated_Fat": 6,  "Sugar": 38, "Sodium": 280},
        # ── Pizza Hut ────────────────────────────────────────────
        {"Brand": "Pizza Hut", "Food": "Pizza Hut Veg Supreme (2 sl)",  "Calories": 440, "Protein": 16, "Fat": 18, "Saturated_Fat": 7,  "Sugar": 6,  "Sodium": 920},
        {"Brand": "Pizza Hut", "Food": "Pizza Hut Chicken Tikka (2 sl)","Calories": 480, "Protein": 22, "Fat": 18, "Saturated_Fat": 8,  "Sugar": 5,  "Sodium": 1000},
        {"Brand": "Pizza Hut", "Food": "Pizza Hut Garlic Bread",        "Calories": 310, "Protein": 6,  "Fat": 14, "Saturated_Fat": 4,  "Sugar": 3,  "Sodium": 480},
        {"Brand": "Pizza Hut", "Food": "Pizza Hut Tandoori Paneer (2 sl)", "Calories": 460, "Protein": 18, "Fat": 20, "Saturated_Fat": 9, "Sugar": 4, "Sodium": 880},
        # ── Taco Bell ────────────────────────────────────────────
        {"Brand": "Taco Bell", "Food": "Taco Bell Crunchy Taco",        "Calories": 170, "Protein": 8,  "Fat": 10, "Saturated_Fat": 4,  "Sugar": 1,  "Sodium": 310},
        {"Brand": "Taco Bell", "Food": "Taco Bell Bean Burrito",        "Calories": 380, "Protein": 14, "Fat": 10, "Saturated_Fat": 4,  "Sugar": 4,  "Sodium": 1060},
        {"Brand": "Taco Bell", "Food": "Taco Bell Crunchwrap Supreme",  "Calories": 540, "Protein": 16, "Fat": 23, "Saturated_Fat": 8,  "Sugar": 6,  "Sodium": 1200},
        {"Brand": "Taco Bell", "Food": "Taco Bell Nachos BellGrande",   "Calories": 740, "Protein": 16, "Fat": 38, "Saturated_Fat": 7,  "Sugar": 4,  "Sodium": 1050},
        # ── Miscellaneous ────────────────────────────────────────
        {"Brand": "Wendy's",   "Food": "Wendy's Grilled Chicken Wrap",  "Calories": 270, "Protein": 20, "Fat": 10, "Saturated_Fat": 4,  "Sugar": 2,  "Sodium": 730},
        {"Brand": "Wendy's",   "Food": "Wendy's Baconator",             "Calories": 940, "Protein": 57, "Fat": 66, "Saturated_Fat": 30, "Sugar": 9,  "Sodium": 1650},
        {"Brand": "Wendy's",   "Food": "Wendy's Jr. Cheeseburger",      "Calories": 290, "Protein": 16, "Fat": 14, "Saturated_Fat": 6,  "Sugar": 6,  "Sodium": 620},
        {"Brand": "Chick-fil-A","Food":"Chick-fil-A Grilled Nuggets (8pc)","Calories":140,"Protein": 25, "Fat": 3,  "Saturated_Fat": 1,  "Sugar": 1,  "Sodium": 440},
        {"Brand": "Chick-fil-A","Food":"Chick-fil-A Spicy Chicken Sandwich","Calories":450,"Protein":28,"Fat": 19, "Saturated_Fat": 4,  "Sugar": 6,  "Sodium": 1620},
    ])

elif page == "🎯 Live Predictor":
    st.markdown("<div class='section-title'>🎯 Live Food Health Predictor</div>",
                unsafe_allow_html=True)
    st.markdown("Test the models using an existing food item from the dataset, external fast-food items (**unseen data**), or custom nutritional values.")

    pred_tab1, pred_tab3, pred_tab2 = st.tabs([
        "📋 Select from Dataset",
        "🍕 Test on Unseen Data",
        "🎛️ Manual Sliders"
    ])

    with pred_tab1:
        food_names = food["Food"].tolist()
        selected_food_name = st.selectbox("🍔 Select a food item", food_names)
        row = food[food["Food"] == selected_food_name].iloc[0]

        cal  = float(row["Calories"])
        prot = float(row["Protein"])
        fat  = float(row["Fat"])
        sat  = float(row["Saturated_Fat"])
        sug  = float(row["Sugar"])
        sod  = float(row["Sodium"])

        cat_label = row["Category"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🔥 Calories",   f"{cal} kcal")
        c2.metric("💪 Protein",    f"{prot} g")
        c3.metric("🧈 Fat",        f"{fat} g")
        c4.metric("🐄 Sat Fat",    f"{sat} g")
        c5.metric("🍬 Sugar",      f"{sug} g")
        c6.metric("🧂 Sodium",     f"{sod} mg")
        st.caption(f"Category: **{cat_label}**")

        run_pred_1 = st.button("🔍 Predict Selected Food", use_container_width=True, key="pred_btn_1")

    with pred_tab3:
        st.markdown("""<div class='card'>
            <b style='color:#ff6f00;'>🍕 Unseen Data Testing</b><br><br>
            These items are from <b>Burger King, Subway, KFC, Domino's, Pizza Hut,
            Taco Bell, Wendy's & Chick-fil-A</b> — none of them were in the
            McDonald's India training set. This tests how well the model
            <b>generalises</b> to new, unseen fast-food items.
        </div>""", unsafe_allow_html=True)

        brand_filter = st.selectbox("🏪 Filter by Brand", ["All Brands"] + sorted(EXTERNAL_FOODS["Brand"].unique().tolist()))
        if brand_filter != "All Brands":
            ext_filtered = EXTERNAL_FOODS[EXTERNAL_FOODS["Brand"] == brand_filter]
        else:
            ext_filtered = EXTERNAL_FOODS

        ext_food_names = ext_filtered["Food"].tolist()
        selected_ext = st.selectbox("🍔 Select an external food item", ext_food_names, key="ext_food_sel")
        ext_row = ext_filtered[ext_filtered["Food"] == selected_ext].iloc[0]

        cal  = float(ext_row["Calories"])
        prot = float(ext_row["Protein"])
        fat  = float(ext_row["Fat"])
        sat  = float(ext_row["Saturated_Fat"])
        sug  = float(ext_row["Sugar"])
        sod  = float(ext_row["Sodium"])

        st.caption(f"Brand: **{ext_row['Brand']}**  ·  ⚠️ Not in training data")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🔥 Calories",   f"{cal} kcal")
        c2.metric("💪 Protein",    f"{prot} g")
        c3.metric("🧈 Fat",        f"{fat} g")
        c4.metric("🐄 Sat Fat",    f"{sat} g")
        c5.metric("🍬 Sugar",      f"{sug} g")
        c6.metric("🧂 Sodium",     f"{sod} mg")

        run_pred_1 = st.button("🔍 Predict External Food", use_container_width=True, key="pred_btn_3")

    with pred_tab2:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                cal_m = st.slider("🔥 Calories (kcal)", 0, 1200, 350)
                prot_m = st.slider("💪 Protein (g)",   0, 60,   12)

            with col2:
                fat_m  = st.slider("🧈 Total Fat (g)",       0, 80, 10)
                sat_m  = st.slider("🐄 Saturated Fat (g)",   0, 40,  4)

            with col3:
                sug_m  = st.slider("🍬 Sugar (g)",    0, 100,  8)
                sod_m  = st.slider("🧂 Sodium (mg)",  0, 2500, 450)

            run_pred_2 = st.form_submit_button("🔍 Predict Health Label", use_container_width=True)

        if run_pred_2:
            cal, prot, fat, sat, sug, sod = cal_m, prot_m, fat_m, sat_m, sug_m, sod_m
            run_pred_1 = True

    if 'run_pred_1' in locals() and run_pred_1:
        new_food = pd.DataFrame({
            "Calories":      [cal],
            "Protein":       [prot],
            "Fat":           [fat],
            "Saturated_Fat": [sat],
            "Sugar":         [sug],
            "Sodium":        [sod],
        })

        lr_class = lr_model.predict(new_food)[0]
        lr_prob  = lr_model.predict_proba(new_food)[0]
        dt_class = dt_model.predict(new_food)[0]

        # Manual health score
        score = int(cal <= 400) + int(fat <= 15) + int(sat <= 5) + \
                int(sug <= 10) + int(sod <= 600) + int(prot >= 8)
        score_label = "Healthy" if score >= 4 else "Unhealthy"

        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            cls_html = "pred-healthy" if lr_class == 1 else "pred-unhealthy"
            icon = "✅" if lr_class == 1 else "❌"
            label = "Healthy" if lr_class == 1 else "Unhealthy"
            st.markdown(f"<div class='{cls_html}'>{icon} Logistic Regression<br>{label}</div>",
                        unsafe_allow_html=True)
        with r2:
            cls_html = "pred-healthy" if dt_class == 1 else "pred-unhealthy"
            icon = "✅" if dt_class == 1 else "❌"
            label = "Healthy" if dt_class == 1 else "Unhealthy"
            st.markdown(f"<div class='{cls_html}'>{icon} Decision Tree<br>{label}</div>",
                        unsafe_allow_html=True)
        with r3:
            cls_html = "pred-healthy" if score >= 4 else "pred-unhealthy"
            icon = "✅" if score >= 4 else "❌"
            st.markdown(f"<div class='{cls_html}'>{icon} Health Score<br>{score_label} ({score}/6)</div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.barh(["Unhealthy", "Healthy"], lr_prob,
                color=["#ef5350", "#66bb6a"], edgecolor="#0f0c29", height=0.45)
        for i, v in enumerate(lr_prob):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.set_title("Logistic Regression – Probability")
        ax.set_xlabel("Probability")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Nutritional breakdown
        st.markdown("**Your input vs healthy thresholds:**")
        thresholds = {
            "Calories": (cal, 400, "≤", "kcal"),
            "Protein":  (prot, 8,  "≥", "g"),
            "Fat":      (fat,  15, "≤", "g"),
            "Sat Fat":  (sat,  5,  "≤", "g"),
            "Sugar":    (sug,  10, "≤", "g"),
            "Sodium":   (sod,  600,"≤", "mg"),
        }
        cols = st.columns(6)
        for col, (name, (val, thresh, op, unit)) in zip(cols, thresholds.items()):
            if op == "≤":
                ok = val <= thresh
            else:
                ok = val >= thresh
            icon = "✅" if ok else "❌"
            col.metric(f"{icon} {name}", f"{val} {unit}", f"Threshold: {op}{thresh}")
