import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Dengue Outbreak Analytics",
    page_icon="🦟",
    layout="wide"
)

# ------------------------------------------------
# STYLE
# ------------------------------------------------
st.markdown("""
<style>
.main-title{
font-size:42px;
font-weight:700;
text-align:center;
}

.subtitle{
text-align:center;
color:gray;
margin-bottom:30px;
}

div[data-testid="stMetric"]{
background-color:white;
border-radius:12px;
padding:10px;
box-shadow:0px 4px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown('<div class="main-title">Dengue Outbreak Dynamics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Stochastic Analysis and Prediction of Dengue Cases</div>', unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv("clean_dengue_india_regions2.csv")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("Controls")
regions = sorted(df["Region"].unique())
region = st.sidebar.selectbox("Select Region", regions)
data = df[df["Region"] == region].sort_values("Year")

# ------------------------------------------------
# GROWTH RATE
# ------------------------------------------------
data["growth"] = data["Cases"].pct_change()
growth = data["growth"].replace([np.inf,-np.inf],np.nan).dropna()
avg_growth = growth.median() if len(growth) > 0 else 0

# ------------------------------------------------
# LYAPUNOV EXPONENT
# ------------------------------------------------
st.header("Lyapunov Stability Analysis")
st.latex(r"\lambda = mean(\log(1 + growth))")
growth_clean = growth[growth > -0.99]
lyapunov = np.mean(np.log(1 + growth_clean)) if len(growth_clean) > 0 else 0

# ------------------------------------------------
# METRICS
# ------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Growth Rate", round(avg_growth,3))
c2.metric("Lyapunov Exponent", round(lyapunov,4))
c3.metric("Lyapunov Time (Years)", round(1/abs(lyapunov),2) if lyapunov !=0 else "∞")

# ------------------------------------------------
# STABILITY CLASSIFICATION
# ------------------------------------------------
def classify_lyap(l):
    if l < -0.01:
        return "Declining"
    elif -0.01 <= l <= 0.01:
        return "Stable"
    elif 0.01 < l <= 0.08:
        return "Growing"
    else:
        return "Volatile"

status = classify_lyap(lyapunov)
st.metric("System Stability", status)

# ------------------------------------------------
# YEARWISE CASE GRAPH
# ------------------------------------------------
st.header("Year-wise Dengue Cases")
fig_bar = px.bar(
    data,
    x="Year",
    y="Cases",
    color="Cases",
    color_continuous_scale="RdPu"
)
fig_bar.update_layout(template="plotly_white")
st.plotly_chart(fig_bar,use_container_width=True)

# ------------------------------------------------
# ROLLING TREND
# ------------------------------------------------
st.header("Smoothed Outbreak Trend")
data["rolling"]=data["Cases"].rolling(3).mean()
fig_trend = go.Figure()
fig_trend.add_trace(
    go.Scatter(
        x=data["Year"],
        y=data["Cases"],
        mode="lines+markers",
        name="Actual Cases",
        line=dict(color="#ff4da6")
    )
)
fig_trend.add_trace(
    go.Scatter(
        x=data["Year"],
        y=data["rolling"],
        mode="lines",
        name="3-Year Moving Avg",
        line=dict(color="#7a0177",width=4)
    )
)
fig_trend.update_layout(template="plotly_white")
st.plotly_chart(fig_trend,use_container_width=True)

# ------------------------------------------------
# HEATMAP
# ------------------------------------------------
st.header("Dengue Outbreak Heatmap")
heatmap_df = df.pivot_table(index="Region", columns="Year", values="Cases", aggfunc="sum")
fig_heat = px.imshow(
    heatmap_df,
    color_continuous_scale="RdPu",
    aspect="auto",
    labels=dict(x="Year",y="Region",color="Cases")
)
fig_heat.update_layout(height=700,template="plotly_white")
st.plotly_chart(fig_heat,use_container_width=True)

# ------------------------------------------------
# MONTE CARLO SIMULATION
# ------------------------------------------------
st.header("Monte Carlo Outbreak Simulation")
st.latex(r"Cases_{t+1}=Cases_t(1+G+\epsilon)")
last_cases=data["Cases"].iloc[-1]
last_year=int(data["Year"].max())
future_years=5
simulations=200
years=list(range(last_year+1,last_year+future_years+1))
paths=[]
for s in range(simulations):
    current=last_cases
    path=[]
    for y in years:
        noise=np.random.normal(0,0.05)
        current=current*(1+avg_growth+noise)
        path.append(current)
    paths.append(path)
paths=np.array(paths)
mean_path=paths.mean(axis=0)
upper=np.percentile(paths,95,axis=0)
lower=np.percentile(paths,5,axis=0)
fig_sim=go.Figure()
fig_sim.add_trace(go.Scatter(x=years,y=upper,line=dict(width=0),showlegend=False))
fig_sim.add_trace(
    go.Scatter(
        x=years,
        y=lower,
        fill="tonexty",
        fillcolor="rgba(255,0,150,0.2)",
        line=dict(width=0),
        name="Uncertainty"
    )
)
fig_sim.add_trace(
    go.Scatter(
        x=years,
        y=mean_path,
        mode="lines+markers",
        line=dict(color="#ff4da6",width=4),
        name="Expected Cases"
    )
)
fig_sim.update_layout(
    template="plotly_white",
    xaxis_title="Year",
    yaxis_title="Predicted Cases"
)
st.plotly_chart(fig_sim,use_container_width=True)

# ------------------------------------------------
# FUTURE PREDICTION
# ------------------------------------------------
st.header("Future Growth Prediction")
st.latex(r"Cases_{t+1}=Cases_t(1+G)")
future=[]
current=last_cases
for i in range(1,6):
    current=current*(1+avg_growth)
    future.append({"Year":last_year+i,"Cases":current})
future_df=pd.DataFrame(future)
combined=pd.concat([data[["Year","Cases"]],future_df])
combined["Type"]=["Actual"]*len(data)+["Predicted"]*len(future_df)
fig_pred=px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True,
    color_discrete_sequence=["#ff4da6","#ff99cc"]
)
fig_pred.update_layout(template="plotly_white")
st.plotly_chart(fig_pred,use_container_width=True)

# ------------------------------------------------
# ROLLING LYAPUNOV & HISTOGRAM
# ------------------------------------------------
st.header("Rolling Lyapunov & Distribution")
data['rolling_growth'] = data['Cases'].pct_change().rolling(3).mean()
data['rolling_lyapunov'] = np.log(1 + data['rolling_growth']).replace([np.inf,-np.inf],0)
fig_lyap = go.Figure()
fig_lyap.add_trace(go.Scatter(
    x=data['Year'],
    y=data['rolling_lyapunov'],
    mode='lines+markers',
    line=dict(color="#7a0177", width=3),
    name="Rolling Lyapunov"
))
fig_lyap.update_layout(template="plotly_white", title="Year-wise Lyapunov Exponent", yaxis_title="λ")
st.plotly_chart(fig_lyap,use_container_width=True)
fig_hist = px.histogram(data, x='rolling_lyapunov', nbins=20, title="Distribution of Lyapunov Exponents")
st.plotly_chart(fig_hist,use_container_width=True)

# ------------------------------------------------
# STABILITY CLASSIFICATION OVER YEARS
# ------------------------------------------------
st.header("Year-wise Stability Classification")
data['Lyapunov_Status'] = data['rolling_lyapunov'].apply(classify_lyap)
fig_status = px.scatter(data, x='Year', y='Cases', color='Lyapunov_Status',
                        title="Year-wise Stability Classification",
                        color_discrete_sequence=["#2ca02c","#1f77b4","#ff7f0e","#d62728"])
st.plotly_chart(fig_status,use_container_width=True)

# ------------------------------------------------
# SENSITIVITY TO INITIAL CONDITIONS
# ------------------------------------------------
st.header("Sensitivity to Initial Conditions")
t_future = np.arange(1,6)
sensitivity = np.exp(lyapunov * t_future)
fig_sens = px.line(x=t_future, y=sensitivity,
                   labels={"x":"Years Ahead", "y":"Sensitivity"},
                   title="Expected Divergence Over Time")
st.plotly_chart(fig_sens,use_container_width=True)

# ------------------------------------------------
# LYAPUNOV VS GROWTH RATE SCATTER
# ------------------------------------------------
st.header("Lyapunov vs Growth Rate Across Regions")
df['lyap_all'] = np.log(1 + df['Cases'].pct_change().replace([np.inf,-np.inf],0))
fig_scatter = px.scatter(df, x="growth", y="lyap_all",
                         color="Region", hover_data=["Year", "Cases"],
                         title="Lyapunov vs Growth Rate Across Regions")
st.plotly_chart(fig_scatter,use_container_width=True)

# ------------------------------------------------
# DATASET
# ------------------------------------------------
st.header("Dataset")
st.dataframe(data)
