# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config(page_title="Dengue Risk Analysis", layout="wide")
st.title("Dengue Risk Analysis Dashboard")

# --- Load dataset ---
@st.cache_data
def load_data(path="clean_dengue_india_regions2.csv"):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()
    return df

df = load_data()

if df.empty:
    st.warning("Dataset is empty or could not be loaded.")
else:
    st.subheader("Raw Data")
    st.dataframe(df.head(10))

# --- Helper: Clean columns ---
def clean_columns(df, numeric_cols, required_cols):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    return df

# --- Scatter Plot: Lyapunov vs Growth ---
scatter_cols = ["growth", "lyap_all", "Region", "Year", "Cases"]
df_scatter = clean_columns(df, ["growth", "lyap_all"], scatter_cols)

if not df_scatter.empty:
    fig_scatter = px.scatter(
        df_scatter,
        x="growth",
        y="lyap_all",
        color="Region" if "Region" in df_scatter.columns else None,
        hover_data=[col for col in ["Year", "Cases"] if col in df_scatter.columns],
        title="Lyapunov vs Growth Rate Across Regions"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("Not enough data for scatter plot.")

# --- Heatmap: Lyapunov values by Region and Year ---
heatmap_cols = ["Region", "Year", "lyap_all"]
df_heatmap = df[heatmap_cols].copy()
df_heatmap = df_heatmap.dropna()

if not df_heatmap.empty:
    heatmap_data = df_heatmap.pivot_table(
        index="Region", columns="Year", values="lyap_all", aggfunc="mean"
    )
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale="Viridis"
    ))
    fig_heatmap.update_layout(title="Lyapunov Heatmap by Region and Year")
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("Not enough data for heatmap.")

# --- Additional charts (example: growth over years) ---
line_cols = ["Year", "growth", "Region"]
df_line = clean_columns(df, ["growth"], line_cols)

if not df_line.empty:
    fig_line = px.line(
        df_line,
        x="Year",
        y="growth",
        color="Region" if "Region" in df_line.columns else None,
        title="Growth Rate Over Years by Region",
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning("Not enough data for line chart.")

# --- Footer ---
st.markdown("---")
st.markdown("Dashboard created by Aabha K. | Dengue Risk Analysis")
