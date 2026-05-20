import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AutoAnaliz ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card {
    background: #1e2433; border-radius: 12px;
    padding: 16px 20px; margin: 4px 0;
    border: 1px solid #2a2d3a;
}
.insight { 
    background: #0f2027; border-left: 4px solid #3b82f6;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    margin: 6px 0; font-size: 0.88rem; color: #cbd5e1;
}
.warn { 
    background: #1f0f0f; border-left: 4px solid #ef4444;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    margin: 6px 0; font-size: 0.88rem; color: #fca5a5;
}
.good { 
    background: #0f1f12; border-left: 4px solid #22c55e;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    margin: 6px 0; font-size: 0.88rem; color: #86efac;
}
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #3b82f6;
    margin: 20px 0 8px 0; padding-bottom: 4px;
    border-bottom: 1px solid #2a2d3a;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# AKILLI KOLON ALGILAMA
# ══════════════════════════════════════════════════════

def detect_columns(df):
    """Kolonları akıllıca sınıflandır"""
    info = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "year": [],
        "binary": [],
        "id": [],
        "text": [],
    }

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        # Tarih tespiti
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            info["datetime"].append(col)
            continue

        # Sayısal tip
        if pd.api.types.is_numeric_dtype(df[col]):
            nuniq = df[col].nunique()
            # Yıl tespiti: 1900-2100 arası, az unique değer
            if nuniq <= 50 and df[col].between(1900, 2100).mean() > 0.9:
                info["year"].append(col)
            # Binary: 0/1 veya 2 unique
            elif nuniq == 2:
                info["binary"].append(col)
            # ID tespiti: her satır farklı, sıralı
            elif nuniq == len(df) and df[col].is_monotonic_increasing:
                info["id"].append(col)
            else:
                info["numeric"].append(col)
            continue

        # String kolonlar
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            nuniq = df[col].nunique()

            # Tarih string tespiti
            try:
                parsed = pd.to_datetime(series.head(20), errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                    info["datetime"].append(col)
                    continue
            except Exception:
                pass

            # Yıl string tespiti
            try:
                as_int = pd.to_numeric(series, errors="coerce")
                if as_int.notna().mean() > 0.9 and as_int.between(1900, 2100).mean() > 0.9 and nuniq <= 50:
                    df[col] = as_int.astype("Int64")
                    info["year"].append(col)
                    continue
            except Exception:
                pass

            # Binary string
            low_vals = set(series.str.lower().unique())
            binary_sets = [
                {"evet","hayır"}, {"yes","no"}, {"true","false"},
                {"var","yok"}, {"sağ","ölü"}, {"erkek","dişi"},
                {"male","female"}, {"1","0"}, {"aktif","pasif"},
                {"açık","kapalı"}
            ]
            if any(low_vals <= bs for bs in binary_sets) or nuniq == 2:
                info["binary"].append(col)
                continue

            # ID tespiti
            if nuniq == len(df):
                info["id"].append(col)
                continue

            # Uzun metin
            avg_len = series.astype(str).str.len().mean()
            if avg_len > 50:
                info["text"].append(col)
                continue

            # Kategorik
            info["categorical"].append(col)

    return info, df


def clean_data(df):
    """Otomatik veri temizleme"""
    log = []
    original_rows = len(df)

    # Tamamen boş sil
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Duplicate sil
    dup = df.duplicated().sum()
    if dup > 0:
        df = df.drop_duplicates()
        log.append(f"🗑️ {dup} duplicate satır silindi")

    # Türkçe encoding düzelt
    enc_map = {
        "Ã¼":"ü","Ã¶":"ö","Ã§":"ç","ÅŸ":"ş","Ä±":"ı",
        "Ä°":"İ","Ãœ":"Ü","Ã–":"Ö","Ã‡":"Ç","ÄŸ":"ğ","Äž":"Ğ"
    }
    for col in df.select_dtypes(include="object").columns:
        for bad, good in enc_map.items():
            try:
                df[col] = df[col].astype(str).str.replace(bad, good, regex=False)
            except Exception:
                pass

    # Baş/son boşluk temizle
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Virgüllü sayıları düzelt
    for col in df.select_dtypes(include="object").columns:
        try:
            conv = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
            if conv.notna().sum() > len(df) * 0.6:
                df[col] = conv
                log.append(f"🔢 '{col}' sayısallaştırıldı")
        except Exception:
            pass

    removed = original_rows - len(df)
    if removed > 0:
        log.append(f"🗑️ Toplam {removed} satır temizlendi")

    return df, log


def generate_insights(df, col_info):
    """Otomatik içgörü üret"""
    insights = []

    # Sayısal kolonlar için istatistiksel tespitler
    for col in col_info["numeric"][:6]:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        if outliers > 0:
            insights.append(("warn", f"'{col}' kolonunda {outliers} aykırı değer tespit edildi"))

        skew = s.skew()
        if abs(skew) > 1:
            yon = "sağa" if skew > 0 else "sola"
            insights.append(("info", f"'{col}' dağılımı {yon} çarpık (çarpıklık: {skew:.2f})"))

    # Eksik veri uyarıları
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for col, cnt in missing.items():
        pct = cnt / len(df) * 100
        if pct > 20:
            insights.append(("warn", f"'{col}' kolonunda %{pct:.0f} eksik veri var"))

    # Kategorik baskınlık
    for col in col_info["categorical"][:4]:
        vc = df[col].value_counts()
        if len(vc) > 0:
            top_pct = vc.iloc[0] / len(df) * 100
            if top_pct > 60:
                insights.append(("info", f"'{col}' kolonunda '{vc.index[0]}' değeri baskın (%{top_pct:.0f})"))

    # Yıl trendi
    if col_info["year"] and col_info["numeric"]:
        yr_col = col_info["year"][0]
        num_col = col_info["numeric"][0]
        try:
            trend = df.groupby(yr_col)[num_col].mean()
            if len(trend) >= 2:
                change = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
                yon = "arttı" if change > 0 else "azaldı"
                insights.append(("good" if change > 0 else "warn",
                    f"'{num_col}', {int(trend.index[0])}→{int(trend.index[-1])} arasında %{abs(change):.1f} {yon}"))
        except Exception:
            pass

    return insights


# ══════════════════════════════════════════════════════
# GRAFİK ÜRETİCİLERİ
# ══════════════════════════════════════════════════════

def plot_time_series(df, year_col, num_cols, cat_col=None):
    figs = []
    for num_col in num_cols[:4]:
        try:
            if cat_col and df[cat_col].nunique() <= 12:
                data = df.groupby([year_col, cat_col])[num_col].mean().reset_index()
                fig = px.line(data, x=year_col, y=num_col, color=cat_col,
                              markers=True,
                              title=f"📈 {num_col} — Yıllara Göre ({cat_col} bazlı)",
                              color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                data = df.groupby(year_col)[num_col].agg(["mean","min","max"]).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data[year_col], y=data["mean"],
                                         mode="lines+markers", name="Ortalama",
                                         line=dict(color="#3b82f6", width=3)))
                fig.add_trace(go.Scatter(x=data[year_col], y=data["max"],
                                         mode="lines", name="Maks",
                                         line=dict(color="#22c55e", dash="dot")))
                fig.add_trace(go.Scatter(x=data[year_col], y=data["min"],
                                         mode="lines", name="Min",
                                         line=dict(color="#ef4444", dash="dot"),
                                         fill="tonexty", fillcolor="rgba(59,130,246,0.08)"))
                fig.update_layout(title=f"📈 {num_col} — Yıllara Göre Trend")
            fig.update_layout(hovermode="x unified", plot_bgcolor="#0f1117",
                              paper_bgcolor="#0f1117", font_color="#e0e0e0")
            figs.append((f"Yıl Trendi: {num_col}", fig))
        except Exception:
            pass
    return figs


def plot_category_comparison(df, cat_col, num_cols):
    figs = []
    nuniq = df[cat_col].nunique()
    if nuniq < 2 or nuniq > 50:
        return figs

    for num_col in num_cols[:4]:
        try:
            agg = df.groupby(cat_col)[num_col].agg(["mean","count","std"]).reset_index()
            agg.columns = [cat_col, "Ortalama", "Adet", "StdSapma"]
            agg = agg.sort_values("Ortalama", ascending=False)

            if nuniq <= 15:
                fig = px.bar(agg, x=cat_col, y="Ortalama",
                             color="Ortalama", color_continuous_scale="Blues",
                             text="Ortalama",
                             title=f"📊 {cat_col} — {num_col} Karşılaştırması",
                             hover_data=["Adet"])
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            else:
                fig = px.bar(agg.head(20), x="Ortalama", y=cat_col,
                             orientation="h", color="Ortalama",
                             color_continuous_scale="Blues",
                             title=f"📊 {cat_col} — {num_col} (Top 20)")
            fig.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                              font_color="#e0e0e0", showlegend=False)
            figs.append((f"{cat_col} × {num_col}", fig))
        except Exception:
            pass
    return figs


def plot_distribution(df, num_cols):
    figs = []
    for col in num_cols[:6]:
        try:
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=(f"{col} — Histogram", f"{col} — Box Plot"))
            hist_data = df[col].dropna()
            fig.add_trace(
                go.Histogram(x=hist_data, nbinsx=30,
                             marker_color="#3b82f6", opacity=0.8, name=col),
                row=1, col=1
            )
            fig.add_trace(
                go.Box(y=hist_data, marker_color="#7c3aed",
                       boxmean=True, name=col),
                row=1, col=2
            )
            fig.update_layout(
                title=f"📉 {col} Dağılım Analizi",
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font_color="#e0e0e0", showlegend=False
            )
            figs.append((f"Dağılım: {col}", fig))
        except Exception:
            pass
    return figs


def plot_binary_analysis(df, binary_cols, num_cols, cat_cols):
    figs = []
    for b_col in binary_cols[:3]:
        # Pasta: genel dağılım
        try:
            vc = df[b_col].value_counts().reset_index()
            vc.columns = [b_col, "Adet"]
            fig_pie = px.pie(vc, names=b_col, values="Adet",
                             title=f"🥧 {b_col} Dağılımı",
                             color_discrete_sequence=["#3b82f6","#ef4444","#22c55e","#f59e0b"])
            fig_pie.update_layout(paper_bgcolor="#0f1117", font_color="#e0e0e0")
            figs.append((f"{b_col} Dağılımı", fig_pie))
        except Exception:
            pass

        # Sayısal kolonlarla karşılaştırma
        for num_col in num_cols[:2]:
            try:
                fig = px.box(df, x=b_col, y=num_col, color=b_col,
                             title=f"📦 {b_col} → {num_col} İlişkisi",
                             color_discrete_sequence=["#3b82f6","#ef4444"])
                fig.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                  font_color="#e0e0e0")
                figs.append((f"{b_col} × {num_col}", fig))
            except Exception:
                pass

        # Kategorik ile karşılaştırma
        for cat_col in cat_cols[:2]:
            try:
                if df[cat_col].nunique() <= 15:
                    ct = df.groupby([cat_col, b_col]).size().reset_index(name="Adet")
                    fig = px.bar(ct, x=cat_col, y="Adet", color=b_col,
                                 barmode="group",
                                 title=f"📊 {cat_col} × {b_col} Dağılımı",
                                 color_discrete_sequence=["#3b82f6","#ef4444"])
                    fig.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                      font_color="#e0e0e0")
                    figs.append((f"{cat_col} × {b_col}", fig))
            except Exception:
                pass
    return figs


def plot_correlation(df, num_cols):
    figs = []
    cols = num_cols[:10]
    if len(cols) < 2:
        return figs
    try:
        corr = df[cols].corr()
        fig = px.imshow(corr, text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="🌡️ Korelasyon Matrisi",
                        zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor="#0f1117", font_color="#e0e0e0")
        figs.append(("Korelasyon", fig))

        # En güçlü korelasyonları bul
        strong = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong.append((cols[i], cols[j], val))
        strong.sort(key=lambda x: abs(x[2]), reverse=True)
        for c1, c2, val in strong[:3]:
            fig_sc = px.scatter(df, x=c1, y=c2,
                                trendline="ols",
                                title=f"🔵 {c1} × {c2} (r={val:.2f})",
                                color_discrete_sequence=["#3b82f6"])
            fig_sc.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                  font_color="#e0e0e0")
            figs.append((f"Scatter: {c1}×{c2}", fig_sc))
    except Exception:
        pass
    return figs


def plot_datetime_analysis(df, dt_cols, num_cols):
    figs = []
    for dt_col in dt_cols[:2]:
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            df_t = df.dropna(subset=[dt_col]).copy()
            df_t["_ay"] = df_t[dt_col].dt.to_period("M").astype(str)

            for num_col in num_cols[:2]:
                monthly = df_t.groupby("_ay")[num_col].mean().reset_index()
                fig = px.line(monthly, x="_ay", y=num_col,
                              title=f"📅 {num_col} — Aylık Trend ({dt_col})",
                              markers=True)
                fig.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                  font_color="#e0e0e0")
                figs.append((f"Aylık: {num_col}", fig))
        except Exception:
            pass
    return figs


def plot_top_bottom(df, cat_col, num_col, n=10):
    figs = []
    try:
        agg = df.groupby(cat_col)[num_col].mean().reset_index()
        agg.columns = [cat_col, "Ortalama"]
        top = agg.nlargest(n, "Ortalama")
        bot = agg.nsmallest(n, "Ortalama")

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=(f"Top {n} — En Yüksek", f"Bottom {n} — En Düşük"))
        fig.add_trace(
            go.Bar(x=top["Ortalama"], y=top[cat_col], orientation="h",
                   marker_color="#22c55e", name="En Yüksek"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=bot["Ortalama"], y=bot[cat_col], orientation="h",
                   marker_color="#ef4444", name="En Düşük"),
            row=1, col=2
        )
        fig.update_layout(
            title=f"🏆 {cat_col} — {num_col} Sıralama",
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#e0e0e0", showlegend=False
        )
        figs.append((f"Sıralama: {cat_col}", fig))
    except Exception:
        pass
    return figs


# ══════════════════════════════════════════════════════
# ANA UYGULAMA
# ══════════════════════════════════════════════════════

st.title("⚡ AutoAnaliz — Akıllı Veri Analizi")
st.caption("Excel veya CSV yükle → sistem otomatik analiz eder")

uploaded = st.file_uploader(
    "📂 Dosya Yükle (Excel veya CSV)",
    type=["xlsx", "xls", "csv"],
    help="Herhangi bir veri dosyası yükleyin"
)

if uploaded is None:
    st.info("👆 Dosya yükleyerek başlayın — sistem veriyi otomatik analiz edecek")
    st.markdown("""
    **Bu uygulama otomatik olarak şunları yapar:**
    - 🧹 Veriyi temizler (duplicate, encoding, format)
    - 🔍 Kolon tiplerini algılar (yıl, kategori, sayı, tarih, ikili)
    - 📊 Her kolon tipine uygun grafik üretir
    - 💡 Otomatik içgörüler ve uyarılar üretir
    - 📥 Temizlenmiş veriyi indirme imkanı sunar
    """)
    st.stop()

# ── Veri Yükleme & Temizleme ──
with st.spinner("Veri yükleniyor ve temizleniyor..."):
    try:
        if uploaded.name.endswith(".csv"):
            df = None
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1254"]:
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, encoding=enc)
                    break
                except Exception:
                    continue
            if df is None:
                st.error("CSV dosyası okunamadı.")
                st.stop()
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Dosya okunamadı: {e}")
        st.stop()

    df, clean_log = clean_data(df)
    col_info, df = detect_columns(df)

# ── Sidebar ──
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    st.markdown(f"**Dosya:** {uploaded.name}")
    st.markdown(f"**Satır:** {len(df):,} | **Kolon:** {len(df.columns)}")
    st.divider()

    st.markdown("**🔍 Algılanan Kolon Tipleri**")
    type_labels = {
        "numeric": "🔢 Sayısal",
        "categorical": "🏷️ Kategorik",
        "year": "📅 Yıl",
        "binary": "⚖️ İkili (Evet/Hayır)",
        "datetime": "🗓️ Tarih",
        "text": "📝 Metin",
        "id": "🆔 ID",
    }
    for key, label in type_labels.items():
        cols_list = col_info.get(key, [])
        if cols_list:
            st.markdown(f"**{label}:** {', '.join(f'`{c}`' for c in cols_list)}")

    st.divider()

    # Filtreler
    st.markdown("**🎛️ Filtreler**")
    active_filters = {}
    for col in col_info["categorical"][:3]:
        opts = sorted(df[col].dropna().unique().tolist())
        if len(opts) <= 30:
            sel = st.multiselect(col, opts, default=opts, key=f"f_{col}")
            active_filters[col] = sel

    for col in col_info["year"][:2]:
        years = sorted(df[col].dropna().unique().astype(int).tolist())
        if len(years) >= 2:
            yr_range = st.select_slider(
                col, options=years,
                value=(years[0], years[-1]),
                key=f"y_{col}"
            )
            active_filters[col] = yr_range

    # Filtreyi uygula
    df_f = df.copy()
    for col, val in active_filters.items():
        if isinstance(val, tuple):
            df_f = df_f[df_f[col].between(val[0], val[1])]
        elif val:
            df_f = df_f[df_f[col].isin(val)]

    st.divider()
    st.markdown(f"**Filtrelenmiş:** {len(df_f):,} satır")

    # Temiz veri indir
    st.divider()
    csv_out = df_f.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Temiz Veriyi İndir", csv_out, "temiz_veri.csv", "text/csv")

# ── Temizlik Logu ──
if clean_log:
    with st.expander("🧹 Temizlik Raporu", expanded=False):
        for l in clean_log:
            st.markdown(l)

# ── Özet Metrikler ──
st.markdown('<div class="section-title">📊 Genel Özet</div>', unsafe_allow_html=True)
summary_cols = col_info["numeric"][:6]
if summary_cols:
    mcols = st.columns(min(len(summary_cols), 4))
    for i, col in enumerate(summary_cols[:4]):
        with mcols[i]:
            val = df_f[col].mean()
            total = df_f[col].sum()
            st.metric(
                label=col,
                value=f"{val:.2f}",
                delta=f"Toplam: {total:,.0f}"
            )

# ── İçgörüler ──
insights = generate_insights(df_f, col_info)
if insights:
    st.markdown('<div class="section-title">💡 Otomatik İçgörüler</div>', unsafe_allow_html=True)
    icols = st.columns(2)
    for i, (typ, msg) in enumerate(insights[:8]):
        css = "insight" if typ == "info" else ("warn" if typ == "warn" else "good")
        icon = "ℹ️" if typ == "info" else ("⚠️" if typ == "warn" else "✅")
        with icols[i % 2]:
            st.markdown(f'<div class="{css}">{icon} {msg}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# OTOMATİK GRAFİKLER
# ══════════════════════════════════════════════
all_figs = []

# 1. Yıl trendi
if col_info["year"] and col_info["numeric"]:
    yr_col = col_info["year"][0]
    cat_col = col_info["categorical"][0] if col_info["categorical"] else None
    all_figs += plot_time_series(df_f, yr_col, col_info["numeric"], cat_col)

# 2. Kategorik karşılaştırma
for cat_col in col_info["categorical"][:3]:
    if col_info["numeric"]:
        all_figs += plot_category_comparison(df_f, cat_col, col_info["numeric"])
        # Top/Bottom sıralama (çok kategorik var ise)
        if df_f[cat_col].nunique() > 10:
            all_figs += plot_top_bottom(df_f, cat_col, col_info["numeric"][0])

# 3. İkili analizler
if col_info["binary"]:
    all_figs += plot_binary_analysis(
        df_f, col_info["binary"], col_info["numeric"], col_info["categorical"]
    )

# 4. Sayısal dağılımlar
if col_info["numeric"]:
    all_figs += plot_distribution(df_f, col_info["numeric"])

# 5. Korelasyon
if len(col_info["numeric"]) >= 2:
    all_figs += plot_correlation(df_f, col_info["numeric"])

# 6. Tarih serisi
if col_info["datetime"] and col_info["numeric"]:
    all_figs += plot_datetime_analysis(df_f, col_info["datetime"], col_info["numeric"])

# ── Grafikleri Göster ──
if all_figs:
    st.markdown('<div class="section-title">📈 Otomatik Grafikler</div>',
                unsafe_allow_html=True)
    st.caption(f"{len(all_figs)} grafik otomatik üretildi")

    # Tab'larla organize et
    tab_size = 6
    chunks = [all_figs[i:i+tab_size] for i in range(0, len(all_figs), tab_size)]
    if len(chunks) == 1:
        for title, fig in all_figs:
            st.plotly_chart(fig, use_container_width=True)
    else:
        tab_labels = [f"Bölüm {i+1}" for i in range(len(chunks))]
        tabs = st.tabs(tab_labels)
        for tab, chunk in zip(tabs, chunks):
            with tab:
                for title, fig in chunk:
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Grafik üretilemedi — veri yeterli sayısal veya kategorik kolon içermiyor olabilir.")

# ── Ham Veri ──
st.divider()
with st.expander("📋 Ham Veriyi Göster / İndir"):
    st.dataframe(df_f, use_container_width=True)

