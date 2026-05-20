import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="DataWizard Dashboard", page_icon="⚡", layout="wide")

@st.cache_data
def load_data():
    try:
        return pd.read_excel("data/cleaned_data.xlsx")
    except Exception:
        return pd.read_csv("data/cleaned_data.csv")

df = load_data()

# ── Sidebar ──
st.sidebar.title("⚙️ Filtreler")
st.sidebar.markdown(f"**Toplam Kayıt:** {len(df):,}")
st.sidebar.divider()

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

for col in cat_cols[:4]:
    opts = sorted(df[col].dropna().unique().tolist())
    sel = st.sidebar.multiselect(col, opts, default=opts)
    df = df[df[col].isin(sel)] if sel else df

# ── Header ──
st.title("⚡ DataWizard Dashboard")
st.caption(f"{len(df):,} kayıt gösteriliyor")
st.divider()

# ── Metrik Kartlar ──
if num_cols:
    cols_m = st.columns(min(len(num_cols), 4))
    for i, col in enumerate(num_cols[:4]):
        cols_m[i].metric(col, f"{df[col].mean():.2f}", help="Ortalama")
    st.divider()

# ── Grafikler ──
tab_names = []
if cat_cols and num_cols: tab_names.append("Bar")
if num_cols:              tab_names.append("Histogram")
if len(num_cols) >= 2:    tab_names.append("Scatter")
if cat_cols:              tab_names.append("Pasta")
if len(num_cols) >= 2:    tab_names.append("Korelasyon")
if num_cols:              tab_names.append("Box Plot")

if tab_names:
    tabs = st.tabs(tab_names)
    idx = 0

    if cat_cols and num_cols:
        with tabs[idx]:
            c, n = cat_cols[0], num_cols[0]
            bd = df.groupby(c)[n].mean().reset_index()
            st.plotly_chart(
                px.bar(bd, x=c, y=n, color=n, color_continuous_scale="Blues",
                       title=f"{c} Bazında Ortalama {n}"),
                use_container_width=True
            )
        idx += 1

    if num_cols:
        with tabs[idx]:
            n = num_cols[0]
            st.plotly_chart(
                px.histogram(df, x=n, nbins=30,
                             color_discrete_sequence=["#3b82f6"],
                             title=f"{n} Dağılımı"),
                use_container_width=True
            )
        idx += 1

    if len(num_cols) >= 2:
        with tabs[idx]:
            n1, n2 = num_cols[0], num_cols[1]
            carg = {"color": cat_cols[0]} if cat_cols else {}
            st.plotly_chart(
                px.scatter(df, x=n1, y=n2, **carg, opacity=0.7,
                           title=f"{n1} vs {n2}"),
                use_container_width=True
            )
        idx += 1

    if cat_cols:
        with tabs[idx]:
            c = cat_cols[0]
            pie_d = df[c].value_counts().reset_index()
            pie_d.columns = [c, "Adet"]
            st.plotly_chart(
                px.pie(pie_d, names=c, values="Adet", title=f"{c} Dağılımı"),
                use_container_width=True
            )
        idx += 1

    if len(num_cols) >= 2:
        with tabs[idx]:
            corr = df[num_cols[:8]].corr()
            st.plotly_chart(
                px.imshow(corr, text_auto=True,
                          color_continuous_scale="RdBu_r",
                          title="Korelasyon Matrisi"),
                use_container_width=True
            )
        idx += 1

    if num_cols:
        with tabs[idx]:
            n = num_cols[0]
            carg = {"color": cat_cols[0]} if cat_cols else {}
            st.plotly_chart(
                px.box(df, y=n, **carg, title=f"{n} İstatistiksel Özet"),
                use_container_width=True
            )

st.divider()

# ── Ham Veri ──
with st.expander("📋 Ham Veriyi Göster"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV İndir", csv, "veri.csv", "text/csv")
