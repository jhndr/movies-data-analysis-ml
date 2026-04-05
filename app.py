import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movies ML Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #1F3864;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4fa; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #2E75B6; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1F3864; }
    .metric-label { font-size: 0.8rem; color: #666; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1F3864;
        border-bottom: 2px solid #2E75B6; padding-bottom: 4px; margin-bottom: 1rem;
    }
    .best-badge {
        background: #d4edda; color: #155724; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_movies.csv")
    df = df.dropna(subset=["YEAR", "PRIMARY_GENRE", "RATING"])
    df = df[df["RunTime"] > 0]
    df["LOG_VOTES"] = np.log1p(df["VOTES"])
    df["LOG_GROSS"] = np.log1p(df["Gross_M"])
    le = LabelEncoder()
    df["GENRE_ENC"] = le.fit_transform(df["PRIMARY_GENRE"])
    df["GENRE_LABELS"] = df["PRIMARY_GENRE"]
    return df, le

@st.cache_resource
def train_models(df):
    features = ["YEAR", "GENRE_ENC", "LOG_VOTES", "RunTime", "LOG_GROSS"]
    X = df[features].values
    y = df["RATING"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Neural Network":    MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                          solver="adam", max_iter=500, random_state=42,
                                          early_stopping=True, validation_fraction=0.1)
    }

    results, predictions = {}, {}
    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        results[name] = {
            "MAE":  mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2":   r2_score(y_test, y_pred)
        }
        predictions[name] = y_pred

    rf = models["Random Forest"]
    feature_names = ["Year", "Genre", "Log Votes", "Runtime", "Log Gross"]
    importances   = rf.feature_importances_

    return models, scaler, results, predictions, y_test, y_train, feature_names, importances

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg",
             width=120)
    st.markdown("## 🎬 Movies ML Dashboard")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "🏠 Overview",
        "📊 Data Explorer",
        "🤖 Model Results",
        "🔮 Predict a Rating"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** `cleaned_movies.csv`")
    st.markdown("**Models:** Linear Regression, Random Forest, Neural Network")
    st.markdown("**Task:** IMDb Rating Prediction")

# ── Load ──────────────────────────────────────────────────────────────────────
df, le = load_data()

with st.spinner("Training models — please wait..."):
    models, scaler, results, predictions, y_test, y_train, feature_names, importances = train_models(df)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-title">🎬 Movies ML Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">IMDb Rating Prediction using Machine Learning · movies_csv.xlsx</div>', unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Titles</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["PRIMARY_GENRE"].nunique()}</div><div class="metric-label">Unique Genres</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["RATING"].mean():.2f}</div><div class="metric-label">Mean Rating</div></div>', unsafe_allow_html=True)
    with c4:
        best_model = max(results, key=lambda m: results[m]["R2"])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{results[best_model]["R2"]:.3f}</div><div class="metric-label">Best R² ({best_model})</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{results[best_model]["MAE"]:.3f}</div><div class="metric-label">Best MAE ({best_model})</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["RATING"], bins=30, color="#2E75B6", edgecolor="white", linewidth=0.4)
        ax.axvline(df["RATING"].mean(),   color="red",    linestyle="--", linewidth=1.4, label=f"Mean: {df['RATING'].mean():.2f}")
        ax.axvline(df["RATING"].median(), color="orange", linestyle="--", linewidth=1.4, label=f"Median: {df['RATING'].median():.2f}")
        ax.set_xlabel("IMDb Rating"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Top 10 Genres by Count</div>', unsafe_allow_html=True)
        genre_counts = df["PRIMARY_GENRE"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        palette = sns.color_palette("Blues_r", 10)
        ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color=palette)
        for i, v in enumerate(genre_counts.values[::-1]):
            ax.text(v + 5, i, str(v), va="center", fontsize=7)
        ax.set_xlabel("Number of Titles"); ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    summary_data = []
    for name, r in results.items():
        summary_data.append({
            "Model": name,
            "MAE":  round(r["MAE"], 4),
            "RMSE": round(r["RMSE"], 4),
            "R²":   round(r["R2"], 4),
            "Best": "⭐ Best" if name == best_model else ""
        })
    st.dataframe(pd.DataFrame(summary_data).set_index("Model"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<div class="main-title">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore and filter the cleaned movies dataset</div>', unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔍 Filter Dataset", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            all_genres = sorted(df["PRIMARY_GENRE"].dropna().unique())
            sel_genres = st.multiselect("Genre", all_genres, default=all_genres[:5])
        with fc2:
            year_min, year_max = int(df["YEAR"].min()), int(df["YEAR"].max())
            sel_years = st.slider("Year Range", year_min, year_max, (2010, 2023))
        with fc3:
            sel_rating = st.slider("Rating Range", 1.0, 10.0, (1.0, 10.0), step=0.1)

    fdf = df[
        (df["PRIMARY_GENRE"].isin(sel_genres)) &
        (df["YEAR"] >= sel_years[0]) & (df["YEAR"] <= sel_years[1]) &
        (df["RATING"] >= sel_rating[0]) & (df["RATING"] <= sel_rating[1])
    ]

    st.markdown(f"**{len(fdf):,} titles** match your filters.")
    st.dataframe(fdf[["MOVIES","YEAR","PRIMARY_GENRE","RATING","VOTES","RunTime","Gross_M"]].reset_index(drop=True), use_container_width=True, height=280)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Rating by Genre", "📅 Yearly Trend", "⏱ Runtime vs Rating"])

    with tab1:
        avg_rating = fdf.groupby("PRIMARY_GENRE")["RATING"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#2E75B6" if v == avg_rating.max() else "#7BAFD4" for v in avg_rating.values]
        bars = ax.bar(avg_rating.index, avg_rating.values, color=colors)
        ax.set_ylim(0, 10); ax.set_ylabel("Avg IMDb Rating")
        ax.set_title("Average Rating by Genre (filtered)")
        ax.set_xticklabels(avg_rating.index, rotation=35, ha="right", fontsize=8)
        for bar, val in zip(bars, avg_rating.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", fontsize=7)
        ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        year_df = fdf[(fdf["YEAR"] >= 2000) & (fdf["YEAR"] <= 2023)].dropna(subset=["YEAR"])
        year_counts = year_df["YEAR"].astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(year_counts.index, year_counts.values, marker="o", color="teal",
                linewidth=2, markersize=5)
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color="teal")
        ax.set_xlabel("Year"); ax.set_ylabel("Number of Titles")
        ax.set_title("Titles Released per Year (filtered, 2000–2023)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        rt_df = fdf[(fdf["RunTime"] > 0) & (fdf["RunTime"] <= 300)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(rt_df["RunTime"], rt_df["RATING"],
                        alpha=0.3, s=10, c=rt_df["RATING"], cmap="RdYlGn", vmin=1, vmax=10)
        plt.colorbar(sc, ax=ax, label="Rating")
        ax.set_xlabel("Runtime (minutes)"); ax.set_ylabel("IMDb Rating")
        ax.set_title("Runtime vs IMDb Rating (filtered)")
        ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<div class="main-title">🤖 Model Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Training results, feature importance, and evaluation charts</div>', unsafe_allow_html=True)

    # ── Metric cards ───────────────────────────────────────────────────────
    best_model = max(results, key=lambda m: results[m]["R2"])
    cols = st.columns(3)
    colors = {"Linear Regression": "#ED7D31", "Random Forest": "#2E75B6", "Neural Network": "#70AD47"}
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            badge = " ⭐ Best" if name == best_model else ""
            st.markdown(f"""
            <div style="background:#f0f4fa;border-radius:12px;padding:1rem;border-top:4px solid {colors[name]};text-align:center;">
                <div style="font-weight:700;font-size:1rem;color:#1F3864;">{name}{badge}</div>
                <div style="margin-top:0.6rem;">
                    <span style="font-size:0.8rem;color:#666;">MAE</span><br>
                    <span style="font-size:1.4rem;font-weight:700;color:{colors[name]};">{r['MAE']:.4f}</span>
                </div>
                <div style="margin-top:0.4rem;">
                    <span style="font-size:0.8rem;color:#666;">RMSE</span><br>
                    <span style="font-size:1.4rem;font-weight:700;color:{colors[name]};">{r['RMSE']:.4f}</span>
                </div>
                <div style="margin-top:0.4rem;">
                    <span style="font-size:0.8rem;color:#666;">R²</span><br>
                    <span style="font-size:1.4rem;font-weight:700;color:{colors[name]};">{r['R2']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", "🎯 Actual vs Predicted",
        "📉 Residuals", "⭐ Feature Importance"
    ])

    with tab1:
        model_labels = list(results.keys())
        mae_vals  = [results[m]["MAE"]  for m in model_labels]
        rmse_vals = [results[m]["RMSE"] for m in model_labels]
        r2_vals   = [results[m]["R2"]   for m in model_labels]
        x = np.arange(len(model_labels)); w = 0.25

        fig, ax = plt.subplots(figsize=(9, 4.5))
        b1 = ax.bar(x - w,   mae_vals,  w, label="MAE",  color="#2E75B6")
        b2 = ax.bar(x,       rmse_vals, w, label="RMSE", color="#ED7D31")
        b3 = ax.bar(x + w,   r2_vals,   w, label="R²",   color="#70AD47")
        ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylabel("Score"); ax.set_title("Model Comparison: MAE, RMSE, R²")
        ax.legend(fontsize=9)
        for bar in [*b1, *b2, *b3]:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7)
        ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        sel_model = st.selectbox("Select model", list(results.keys()), index=1)
        y_pred = predictions[sel_model]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=colors[sel_model])
        mn, mx = float(y_test.min()), float(y_test.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Rating"); ax.set_ylabel("Predicted Rating")
        ax.set_title(f"Actual vs Predicted — {sel_model}")
        ax.text(0.05, 0.93, f"R² = {results[sel_model]['R2']:.4f}", transform=ax.transAxes,
                fontsize=9, color="darkred", fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        sel_model2 = st.selectbox("Select model ", list(results.keys()), index=1)
        residuals = y_test - predictions[sel_model2]
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.hist(residuals, bins=40, color="mediumslateblue", edgecolor="white", linewidth=0.4)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.4, label="Zero error")
            ax.axvline(residuals.mean(), color="orange", linestyle="--", linewidth=1.4,
                       label=f"Mean: {residuals.mean():.3f}")
            ax.set_xlabel("Residual (Actual − Predicted)"); ax.set_ylabel("Count")
            ax.set_title(f"Residuals Distribution — {sel_model2}")
            ax.legend(fontsize=8); ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(predictions[sel_model2], residuals, alpha=0.3, s=10, color="steelblue")
            ax.axhline(0, color="red", linestyle="--", linewidth=1.4)
            ax.set_xlabel("Predicted Rating"); ax.set_ylabel("Residual")
            ax.set_title(f"Residuals vs Predicted — {sel_model2}")
            ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        **Residual stats for {sel_model2}:**  
        Mean = `{residuals.mean():.4f}` &nbsp;&nbsp; Std = `{residuals.std():.4f}` &nbsp;&nbsp;
        Min = `{residuals.min():.4f}` &nbsp;&nbsp; Max = `{residuals.max():.4f}`
        """)

    with tab4:
        fi_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = ["#2E75B6" if i == fi_idx[0] else "#7BAFD4" for i in range(len(feature_names))]
        bars = ax.bar([feature_names[i] for i in fi_idx],
                      [importances[i] for i in fi_idx], color=bar_colors)
        for bar, val in zip(bars, [importances[i] for i in fi_idx]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("Importance Score")
        ax.set_title("Feature Importance (Random Forest)")
        ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
            "Rank": [np.where(fi_idx == i)[0][0] + 1 for i in range(len(feature_names))]
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df["Importance"] = fi_df["Importance"].round(4)
        st.dataframe(fi_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT A RATING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Rating":
    st.markdown('<div class="main-title">🔮 Predict a Rating</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter movie/show details to get a predicted IMDb rating from all three models</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎬 Enter Movie Details")

        title_input = st.text_input("Movie / Show Title (optional)", placeholder="e.g. Inception")
        year_input  = st.number_input("Release Year", min_value=1930, max_value=2025, value=2020)

        all_genres  = sorted(df["PRIMARY_GENRE"].dropna().unique())
        genre_input = st.selectbox("Primary Genre", all_genres)

        votes_input   = st.number_input("Estimated Number of Votes", min_value=0, max_value=2_000_000, value=50000, step=1000)
        runtime_input = st.number_input("Runtime (minutes)", min_value=1, max_value=900, value=120)
        gross_input   = st.number_input("Gross Revenue (millions $, 0 if unknown)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5)

        predict_btn = st.button("🔮 Predict Rating", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📊 Prediction Results")

        if predict_btn:
            genre_enc   = le.transform([genre_input])[0]
            log_votes   = np.log1p(votes_input)
            log_gross   = np.log1p(gross_input)
            input_raw   = np.array([[year_input, genre_enc, log_votes, runtime_input, log_gross]])
            input_scaled = scaler.transform(input_raw)

            preds = {}
            for name, model in models.items():
                if name == "Random Forest":
                    preds[name] = float(model.predict(input_raw)[0])
                else:
                    preds[name] = float(model.predict(input_scaled)[0])

            # Clip to valid range
            preds = {k: np.clip(v, 1.0, 10.0) for k, v in preds.items()}
            best_pred = preds["Random Forest"]

            # Big result
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1F3864,#2E75B6);border-radius:14px;
                        padding:1.5rem;text-align:center;color:white;margin-bottom:1rem;">
                <div style="font-size:0.9rem;opacity:0.8;">
                    {'&ldquo;' + title_input + '&rdquo; &nbsp;|&nbsp;' if title_input else ''}{genre_input} · {year_input}
                </div>
                <div style="font-size:3rem;font-weight:800;margin:0.3rem 0;">{best_pred:.1f} / 10</div>
                <div style="font-size:0.85rem;opacity:0.75;">Predicted IMDb Rating (Random Forest)</div>
            </div>
            """, unsafe_allow_html=True)

            # Per-model breakdown
            colors_map = {"Linear Regression": "#ED7D31", "Random Forest": "#2E75B6", "Neural Network": "#70AD47"}
            for name, pred in preds.items():
                bar_pct = int(pred / 10 * 100)
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-weight:600;font-size:0.9rem;">{name}</span>
                        <span style="font-weight:700;color:{colors_map[name]};">{pred:.2f}</span>
                    </div>
                    <div style="background:#e0e0e0;border-radius:6px;height:10px;">
                        <div style="background:{colors_map[name]};width:{bar_pct}%;height:10px;border-radius:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Rating context
            avg = np.mean(list(preds.values()))
            if avg >= 8.0:
                label, color = "🌟 Excellent — likely to be very well received", "#155724"
                bg = "#d4edda"
            elif avg >= 6.5:
                label, color = "👍 Good — above average audience reception expected", "#0c5460"
                bg = "#d1ecf1"
            elif avg >= 5.0:
                label, color = "😐 Average — mixed audience reception expected", "#856404"
                bg = "#fff3cd"
            else:
                label, color = "👎 Below average — likely to receive poor reviews", "#721c24"
                bg = "#f8d7da"

            st.markdown(f"""
            <div style="background:{bg};color:{color};padding:0.8rem 1rem;
                        border-radius:8px;margin-top:1rem;font-weight:600;">
                {label}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Input summary:**")
            st.json({
                "Title": title_input or "(not provided)",
                "Year": year_input,
                "Genre": genre_input,
                "Votes": votes_input,
                "Runtime (min)": runtime_input,
                "Gross ($M)": gross_input,
            })
        else:
            st.info("👈 Fill in the movie details on the left and click **Predict Rating** to see results from all three models.")

            st.markdown("#### 💡 Tips for better predictions")
            st.markdown("""
            - **Votes** has a strong influence — enter your best estimate of expected audience size
            - **Runtime** is the most important feature discovered by Random Forest
            - **Genre** affects the baseline rating significantly (Documentary tends to score higher)
            - Gross revenue has minimal impact on the predicted rating
            """)
