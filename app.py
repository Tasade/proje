import sys
import os
import json
import threading
import subprocess
import shutil
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QProgressBar, QTextEdit, QTabWidget, QMessageBox, QComboBox,
    QCheckBox, QGroupBox, QSplitter, QFrame, QScrollArea, QLineEdit,
    QStatusBar, QToolBar, QAction, QDialog, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QLinearGradient

import pandas as pd
import numpy as np
import requests

# ─────────────────────────────────────────────
# THEME & STYLE
# ─────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #0f1117;
}
QWidget {
    background-color: #0f1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    background: #161b27;
}
QTabBar::tab {
    background: #1e2433;
    color: #888;
    padding: 10px 20px;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #2563eb;
    color: white;
}
QTabBar::tab:hover:!selected {
    background: #252d40;
    color: #ccc;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #3b82f6;
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #1e2433;
    color: #444;
}
QPushButton#dangerBtn {
    background-color: #dc2626;
}
QPushButton#dangerBtn:hover {
    background-color: #ef4444;
}
QPushButton#successBtn {
    background-color: #16a34a;
}
QPushButton#successBtn:hover {
    background-color: #22c55e;
}
QPushButton#warningBtn {
    background-color: #d97706;
}
QPushButton#warningBtn:hover {
    background-color: #f59e0b;
}
QPushButton#ghostBtn {
    background-color: transparent;
    border: 1px solid #2563eb;
    color: #3b82f6;
}
QPushButton#ghostBtn:hover {
    background-color: #1e2433;
}
QTableWidget {
    background-color: #161b27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    gridline-color: #2a2d3a;
    color: #e0e0e0;
}
QTableWidget::item:selected {
    background-color: #2563eb;
}
QHeaderView::section {
    background-color: #1e2433;
    color: #94a3b8;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #2a2d3a;
    font-weight: 600;
    font-size: 12px;
}
QTextEdit {
    background-color: #161b27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    color: #e0e0e0;
    padding: 10px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
}
QLineEdit {
    background-color: #1e2433;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    color: #e0e0e0;
    padding: 8px 12px;
}
QLineEdit:focus {
    border: 1px solid #2563eb;
}
QProgressBar {
    background-color: #1e2433;
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:1 #7c3aed);
    border-radius: 4px;
}
QGroupBox {
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
    color: #94a3b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #3b82f6;
}
QCheckBox {
    color: #e0e0e0;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #2a2d3a;
    background: #1e2433;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border: 1px solid #2563eb;
}
QComboBox {
    background-color: #1e2433;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    color: #e0e0e0;
    padding: 8px 12px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QStatusBar {
    background-color: #0d1117;
    color: #64748b;
    border-top: 1px solid #2a2d3a;
}
QScrollBar:vertical {
    background: #0f1117;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2a2d3a;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QSplitter::handle {
    background: #2a2d3a;
    width: 2px;
}
"""

# ─────────────────────────────────────────────
# WORKER THREADS
# ─────────────────────────────────────────────
class SmartAnalysisWorker(QThread):
    """API gerektirmeden yerel akilli analiz + otomatik temizlik + streamlit uretimi"""
    analysis_done = pyqtSignal(str)
    cleaning_done = pyqtSignal(object)
    streamlit_done = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, df, project_path, chart_types):
        super().__init__()
        self.df = df.copy()
        self.project_path = project_path
        self.chart_types = chart_types

    def run(self):
        try:
            self.progress.emit(10, "Veri analiz ediliyor...")
            report, clean_opts = self._analyze()
            self.analysis_done.emit(report)

            self.progress.emit(35, "Otomatik temizlik uygulanıyor...")
            df_clean = self._auto_clean(clean_opts)
            self.cleaning_done.emit(df_clean)

            self.progress.emit(65, "Streamlit projesi olusturuluyor...")
            project_path = self._build_streamlit(df_clean)

            self.progress.emit(88, "ZIP arsivi hazirlaniyor...")
            zip_path = self._make_zip(project_path)

            self.progress.emit(100, "Her sey hazir!")
            self.streamlit_done.emit(zip_path)
        except Exception as e:
            self.error.emit(str(e))

    def _analyze(self):
        df = self.df
        total_nulls = int(df.isnull().sum().sum())
        total_dups = int(df.duplicated().sum())
        null_pct = (total_nulls / max(df.size, 1)) * 100
        dup_pct = (total_dups / max(len(df), 1)) * 100
        quality = max(0, 100 - int(null_pct * 0.7) - int(dup_pct * 0.3))
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object"]).columns)
        problems, clean_opts = [], {}

        if total_dups > 0:
            sev = "Kritik" if dup_pct > 10 else "Orta"
            problems.append(f"- {sev} — {total_dups} duplicate satir (%{dup_pct:.1f})")
            clean_opts["remove_duplicates"] = True
        if total_nulls > 0:
            sev = "Kritik" if null_pct > 20 else "Orta"
            problems.append(f"- {sev} — {total_nulls} bos deger (%{null_pct:.1f})")
            clean_opts.update({"fill_numeric": True, "fill_text": True,
                               "remove_empty_rows": True, "remove_empty_cols": True})
        for col in cat_cols[:5]:
            sample = df[col].dropna().astype(str).head(20).str.cat()
            if any(c in sample for c in ["Ã", "Å", "Ä"]):
                problems.append("- Orta — Turkce karakter encoding sorunu")
                clean_opts["fix_turkish"] = True
                break
        for col in cat_cols[:5]:
            try:
                if df[col].dropna().astype(str).head(10).str.contains(r"^\d+,\d+$", regex=True).any():
                    problems.append("- Dusuk — Virgul kullanan sayisal degerler")
                    clean_opts["fix_numeric_format"] = True
                    break
            except Exception:
                pass
        clean_opts["strip_whitespace"] = True
        clean_opts["fix_dates"] = any(
            k in col.lower() for col in df.columns
            for k in ["tarih", "date", "dt", "zaman", "time"]
        )

        col_lines = []
        for col in df.columns:
            nc = int(df[col].isnull().sum())
            pct = (nc / max(len(df), 1)) * 100
            dtype = str(df[col].dtype)
            issues = []
            if nc > 0:
                issues.append(f"%{pct:.0f} bos")
            if df[col].duplicated().sum() > len(df) * 0.9:
                issues.append("yuksek tekrar")
            issue_str = ", ".join(issues) if issues else "temiz"
            col_lines.append(f"| `{col}` | {dtype} | {nc} | {issue_str} |")

        problem_str = "\n".join(problems) if problems else "- Ciddi sorun tespit edilmedi"
        applied = "\n".join([f"- {k.replace('_',' ').title()}" for k, v in clean_opts.items() if v])
        col_table = "\n".join(col_lines)

        report = f"""## Veri Kalite Raporu

### Genel Ozet
| Metrik | Deger |
|--------|-------|
| Toplam Satir | {len(df):,} |
| Toplam Kolon | {len(df.columns)} |
| Sayisal Kolon | {len(num_cols)} |
| Kategorik Kolon | {len(cat_cols)} |
| Bos Deger | {total_nulls:,} (%{null_pct:.1f}) |
| Duplicate Satir | {total_dups:,} (%{dup_pct:.1f}) |
| **Veri Kalitesi** | **{quality}/100** |

---

### Tespit Edilen Sorunlar
{problem_str}

---

### Kolon Analizi
| Kolon | Tip | Bos Deger | Durum |
|-------|-----|-----------|-------|
{col_table}

---

### Uygulanan Otomatik Temizlik
{applied}

---
Temizlenmis veri ve Streamlit projesi ZIP olarak hazirlanmistir.
"""
        return report, clean_opts

    def _auto_clean(self, opts):
        df = self.df.copy()
        if opts.get("remove_duplicates"):
            df = df.drop_duplicates()
        if opts.get("remove_empty_rows"):
            df = df.dropna(how="all")
        if opts.get("remove_empty_cols"):
            df = df.dropna(axis=1, how="all")
        if opts.get("fill_numeric"):
            nc = df.select_dtypes(include=[np.number]).columns
            df[nc] = df[nc].fillna(df[nc].median())
        if opts.get("fill_text"):
            tc = df.select_dtypes(include=["object"]).columns
            df[tc] = df[tc].fillna("Bilinmiyor")
        if opts.get("fix_turkish"):
            rep = {"Ã¼":"ü","Ã¶":"ö","Ã§":"ç","ÅŸ":"ş","Ä±":"ı",
                   "Ä°":"İ","Ãœ":"Ü","Ã–":"Ö","Ã‡":"Ç","ÄŸ":"ğ","Äž":"Ğ"}
            for col in df.select_dtypes(include=["object"]).columns:
                for bad, good in rep.items():
                    df[col] = df[col].astype(str).str.replace(bad, good, regex=False)
        if opts.get("fix_numeric_format"):
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    conv = pd.to_numeric(df[col].astype(str).str.replace(",",".",regex=False), errors="coerce")
                    if conv.notna().sum() > len(df) * 0.5:
                        df[col] = conv
                except Exception:
                    pass
        if opts.get("fix_dates"):
            for col in df.columns:
                if any(k in col.lower() for k in ["tarih","date","dt","zaman","time"]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        pass
        if opts.get("strip_whitespace"):
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].str.strip()
        return df

    def _build_streamlit(self, df):
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object"]).columns)
        date_cols = list(df.select_dtypes(include=["datetime64"]).columns)
        charts = self.chart_types

        # Tum satirlar NO-indent olarak uretilir (app.py top-level kod)
        lines = []

        # --- imports & config ---
        lines += [
            "import streamlit as st",
            "import pandas as pd",
            "import numpy as np",
            "import plotly.express as px",
            "",
            "st.set_page_config(page_title='DataWizard Dashboard', page_icon='\u26a1', layout='wide')",
            "",
            "@st.cache_data",
            "def load_data():",
            "    try:",
            "        return pd.read_excel('data/cleaned_data.xlsx')",
            "    except Exception:",
            "        return pd.read_csv('data/cleaned_data.csv')",
            "",
            "df = load_data()",
            "",
            "# --- Sidebar ---",
            "st.sidebar.title('Filtreler')",
            f"st.sidebar.markdown('**Toplam Kayit:** {len(df):,}')",
            "st.sidebar.divider()",
        ]

        # Sidebar filters - NO leading spaces
        for i, col in enumerate(cat_cols[:4]):
            s = col.replace("'", "\'")
            lines.append(f"_u{i} = sorted(df['{s}'].dropna().unique().tolist())")
            lines.append(f"_s{i} = st.sidebar.multiselect('{s}', _u{i}, default=_u{i})")
            lines.append(f"df = df[df['{s}'].isin(_s{i})] if _s{i} else df")

        # --- Header ---
        lines += [
            "",
            "st.title('\u26a1 DataWizard Dashboard')",
            f"st.caption(f'{{len(df):,}} kayit gosteriliyor')",
            "st.divider()",
            "",
            "# --- Metrik Kartlar ---",
        ]

        # Metrics - NO leading spaces
        if num_cols:
            n = min(len(num_cols[:4]), 4)
            lines.append(f"_mc = st.columns({n})")
            for i, col in enumerate(num_cols[:n]):
                col_safe = col.replace("'", "\'")
                lines.append(f"_mc[{i}].metric('{col_safe}', str(round(df['{col_safe}'].mean(), 2)))")
                lines.append(metric_str)

        lines += ["", "st.divider()", "", "# --- Grafikler ---"]

        # Charts - NO leading spaces
        if "bar" in charts and cat_cols and num_cols:
            c, n = cat_cols[0].replace("'", "\'"), num_cols[0].replace("'", "\'")
            lines += [
                "st.subheader('Bar Grafik')",
                f"_bd = df.groupby('{c}')['{n}'].mean().reset_index()",
                f"st.plotly_chart(px.bar(_bd, x='{c}', y='{n}', color='{n}', color_continuous_scale='Blues', title='{c} Bazinda Ort. {n}'), use_container_width=True)",
            ]

        if "line" in charts and num_cols:
            lines.append("st.subheader('Cizgi Grafik')")
            if date_cols:
                d, n = date_cols[0].replace("'", "\'"), num_cols[0].replace("'", "\'")
                lines.append(f"st.plotly_chart(px.line(df.sort_values('{d}'), x='{d}', y='{n}', title='{n} Zaman Serisi'), use_container_width=True)")
            else:
                n = num_cols[0].replace("'", "\'")
                lines.append(f"st.plotly_chart(px.line(df.reset_index(), x='index', y='{n}', title='{n} Trend'), use_container_width=True)")

        if "scatter" in charts and len(num_cols) >= 2:
            n1, n2 = num_cols[0].replace("'", "\'"), num_cols[1].replace("'", "\'")
            carg = f", color='{cat_cols[0]}'" if cat_cols else ""
            lines += [
                "st.subheader('Scatter Plot')",
                f"st.plotly_chart(px.scatter(df, x='{n1}', y='{n2}'{carg}, opacity=0.7, title='{n1} vs {n2}'), use_container_width=True)",
            ]

        if "pie" in charts and cat_cols:
            c = cat_cols[0].replace("'", "\'")
            lines += [
                "st.subheader('Pasta Grafik')",
                f"_pie = df['{c}'].value_counts().reset_index()",
                f"_pie.columns = ['{c}', 'Adet']",
                f"st.plotly_chart(px.pie(_pie, names='{c}', values='Adet', title='{c} Dagilimi'), use_container_width=True)",
            ]

        if "histogram" in charts and num_cols:
            n = num_cols[0].replace("'", "\'")
            lines += [
                "st.subheader('Histogram')",
                f"st.plotly_chart(px.histogram(df, x='{n}', nbins=30, color_discrete_sequence=['#3b82f6'], title='{n} Dagilimi'), use_container_width=True)",
            ]

        if "heatmap" in charts and len(num_cols) >= 2:
            hc = str(num_cols[:8])
            lines += [
                "st.subheader('Korelasyon Isi Haritasi')",
                f"_corr = df[{hc}].corr()",
                "st.plotly_chart(px.imshow(_corr, text_auto=True, color_continuous_scale='RdBu_r', title='Korelasyon Matrisi'), use_container_width=True)",
            ]

        if "box" in charts and num_cols:
            n = num_cols[0].replace("'", "\'")
            carg = f", color='{cat_cols[0]}'" if cat_cols else ""
            lines += [
                "st.subheader('Box Plot')",
                f"st.plotly_chart(px.box(df, y='{n}'{carg}, title='{n} Istatistiksel Ozet'), use_container_width=True)",
            ]

        # --- Footer ---
        lines += [
            "",
            "with st.expander('Ham Veriyi Goster'):",
            "    st.dataframe(df, use_container_width=True)",
            "    _csv = df.to_csv(index=False).encode('utf-8-sig')",
            "    st.download_button('CSV Indir', _csv, 'veri.csv', 'text/csv')",
        ]

        app_code = "\n".join(lines)

        path = Path(self.project_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "data").mkdir(exist_ok=True)
        (path / "app.py").write_text(app_code, encoding="utf-8")
        (path / "requirements.txt").write_text(
            "streamlit>=1.28.0\npandas>=1.5.0\nplotly>=5.15.0\nopenpyxl>=3.0.0\nnumpy>=1.24.0\n")
        df.to_excel(str(path / "data" / "cleaned_data.xlsx"), index=False)
        return str(path)

    def _make_zip(self, project_path):
        import zipfile
        zip_path = project_path + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(project_path).rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(Path(project_path).parent))
        return zip_path



class CleaningWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, df, options):
        super().__init__()
        self.df = df.copy()
        self.options = options

    def run(self):
        try:
            df = self.df
            step = 0
            total = sum(self.options.values())
            
            if self.options.get("remove_duplicates"):
                self.progress.emit(int(step/total*100), "Tekrar eden satırlar siliniyor...")
                before = len(df)
                df = df.drop_duplicates()
                step += 1
                self.progress.emit(int(step/total*100), f"✓ {before - len(df)} duplicate silindi")

            if self.options.get("remove_empty_rows"):
                self.progress.emit(int(step/total*100), "Tamamen boş satırlar siliniyor...")
                df = df.dropna(how='all')
                step += 1

            if self.options.get("remove_empty_cols"):
                self.progress.emit(int(step/total*100), "Tamamen boş kolonlar siliniyor...")
                df = df.dropna(axis=1, how='all')
                step += 1

            if self.options.get("fill_numeric"):
                self.progress.emit(int(step/total*100), "Sayısal boşluklar dolduruluyor (medyan)...")
                num_cols = df.select_dtypes(include=[np.number]).columns
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                step += 1

            if self.options.get("fill_text"):
                self.progress.emit(int(step/total*100), "Metin boşlukları dolduruluyor...")
                text_cols = df.select_dtypes(include=['object']).columns
                df[text_cols] = df[text_cols].fillna("Bilinmiyor")
                step += 1

            if self.options.get("fix_turkish"):
                self.progress.emit(int(step/total*100), "Türkçe karakter sorunları düzeltiliyor...")
                text_cols = df.select_dtypes(include=['object']).columns
                replacements = {
                    'Ã¼': 'ü', 'Ã¶': 'ö', 'Ã§': 'ç', 'ÅŸ': 'ş',
                    'Ä±': 'ı', 'Ä°': 'İ', 'Ãœ': 'Ü', 'Ã–': 'Ö',
                    'Ã‡': 'Ç', 'ÅŸ': 'Ş', 'ÄŸ': 'ğ', 'Äž': 'Ğ'
                }
                for col in text_cols:
                    for bad, good in replacements.items():
                        df[col] = df[col].astype(str).str.replace(bad, good, regex=False)
                step += 1

            if self.options.get("fix_numeric_format"):
                self.progress.emit(int(step/total*100), "Sayısal format düzeltiliyor (nokta/virgül)...")
                text_cols = df.select_dtypes(include=['object']).columns
                for col in text_cols:
                    try:
                        converted = df[col].astype(str).str.replace(',', '.', regex=False)
                        converted = pd.to_numeric(converted, errors='coerce')
                        if converted.notna().sum() > len(df) * 0.5:
                            df[col] = converted
                    except:
                        pass
                step += 1

            if self.options.get("fix_dates"):
                self.progress.emit(int(step/total*100), "Tarih formatları standardize ediliyor...")
                for col in df.columns:
                    if 'tarih' in col.lower() or 'date' in col.lower() or 'dt' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                step += 1

            if self.options.get("strip_whitespace"):
                self.progress.emit(int(step/total*100), "Baştaki/sondaki boşluklar temizleniyor...")
                text_cols = df.select_dtypes(include=['object']).columns
                df[text_cols] = df[text_cols].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                step += 1

            if self.options.get("normalize_columns"):
                self.progress.emit(int(step/total*100), "Kolon adları normalize ediliyor...")
                df.columns = (
                    df.columns.str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace('ı', 'i').str.replace('ğ', 'g')
                    .str.replace('ü', 'u').str.replace('ş', 's')
                    .str.replace('ö', 'o').str.replace('ç', 'c')
                )
                step += 1

            self.progress.emit(100, "✅ Temizlik tamamlandı!")
            self.finished.emit(df)
            
        except Exception as e:
            self.error.emit(str(e))


class StreamlitGeneratorWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, df, project_path, api_key, chart_types):
        super().__init__()
        self.df = df
        self.project_path = project_path
        self.api_key = api_key
        self.chart_types = chart_types

    def run(self):
        try:
            num_cols = list(self.df.select_dtypes(include=[np.number]).columns)
            cat_cols = list(self.df.select_dtypes(include=['object']).columns)
            date_cols = list(self.df.select_dtypes(include=['datetime64']).columns)

            code = self._generate_app(num_cols, cat_cols, date_cols)

            path = Path(self.project_path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "data").mkdir(exist_ok=True)
            (path / "app.py").write_text(code, encoding="utf-8")
            (path / "requirements.txt").write_text(
                "streamlit>=1.28.0\npandas>=1.5.0\nplotly>=5.15.0\nopenpyxl>=3.0.0\nnumpy>=1.24.0\n")
            self.df.to_excel(str(path / "data" / "cleaned_data.xlsx"), index=False)
            self.finished.emit(str(path))
        except Exception as e:
            self.error.emit(str(e))

    def _generate_app(self, num_cols, cat_cols, date_cols):
        """Produce a clean, no-indent top-level app.py"""
        charts = self.chart_types
        df = self.df
        lines = [
            "import streamlit as st",
            "import pandas as pd",
            "import numpy as np",
            "import plotly.express as px",
            "",
            "st.set_page_config(page_title='DataWizard Dashboard', page_icon='\u26a1', layout='wide')",
            "",
            "@st.cache_data",
            "def load_data():",
            "    try:",
            "        return pd.read_excel('data/cleaned_data.xlsx')",
            "    except Exception:",
            "        return pd.read_csv('data/cleaned_data.csv')",
            "",
            "df = load_data()",
            "",
            "st.sidebar.title('Filtreler')",
            f"st.sidebar.markdown('**Toplam Kayit:** {len(df):,}')",
            "st.sidebar.divider()",
        ]
        for i, col in enumerate(cat_cols[:4]):
            s = col.replace("'", "\'")
            lines.append(f"_u{i} = sorted(df['{s}'].dropna().unique().tolist())")
            lines.append(f"_s{i} = st.sidebar.multiselect('{s}', _u{i}, default=_u{i})")
            lines.append(f"df = df[df['{s}'].isin(_s{i})] if _s{i} else df")
        lines += [
            "",
            "st.title('\u26a1 DataWizard Dashboard')",
            f"st.caption('{len(df):,} kayit')",
            "st.divider()",
        ]
        if num_cols:
            n = min(len(num_cols), 4)
            lines.append(f"_mc = st.columns({n})")
            for i, col in enumerate(num_cols[:n]):
                col_safe = col.replace("'", "\'")
                lines.append(f"_mc[{i}].metric('{col_safe}', str(round(df['{col_safe}'].mean(), 2)))")
        lines += ["", "st.divider()", ""]
        if "bar" in charts and cat_cols and num_cols:
            c, n = cat_cols[0].replace("'", "\'"), num_cols[0].replace("'", "\'")
            lines += [
                "st.subheader('Bar Grafik')",
                f"_bd = df.groupby('{c}')['{n}'].mean().reset_index()",
                f"st.plotly_chart(px.bar(_bd, x='{c}', y='{n}', color='{n}', color_continuous_scale='Blues'), use_container_width=True)",
            ]
        if "line" in charts and num_cols:
            lines.append("st.subheader('Cizgi Grafik')")
            if date_cols:
                d, n = date_cols[0].replace("'", "\'"), num_cols[0].replace("'", "\'")
                lines.append(f"st.plotly_chart(px.line(df.sort_values('{d}'), x='{d}', y='{n}'), use_container_width=True)")
            else:
                n = num_cols[0].replace("'", "\'")
                lines.append(f"st.plotly_chart(px.line(df.reset_index(), x='index', y='{n}'), use_container_width=True)")
        if "scatter" in charts and len(num_cols) >= 2:
            n1, n2 = num_cols[0].replace("'", "\'"), num_cols[1].replace("'", "\'")
            carg = f", color='{cat_cols[0]}'" if cat_cols else ""
            lines += [
                "st.subheader('Scatter Plot')",
                f"st.plotly_chart(px.scatter(df, x='{n1}', y='{n2}'{carg}, opacity=0.7), use_container_width=True)",
            ]
        if "pie" in charts and cat_cols:
            c = cat_cols[0].replace("'", "\'")
            lines += [
                "st.subheader('Pasta Grafik')",
                f"_pie = df['{c}'].value_counts().reset_index()",
                f"_pie.columns = ['{c}', 'Adet']",
                f"st.plotly_chart(px.pie(_pie, names='{c}', values='Adet'), use_container_width=True)",
            ]
        if "histogram" in charts and num_cols:
            n = num_cols[0].replace("'", "\'")
            lines += [
                "st.subheader('Histogram')",
                f"st.plotly_chart(px.histogram(df, x='{n}', nbins=30, color_discrete_sequence=['#3b82f6']), use_container_width=True)",
            ]
        if "heatmap" in charts and len(num_cols) >= 2:
            hc = str(num_cols[:8])
            lines += [
                "st.subheader('Korelasyon Isi Haritasi')",
                f"_corr = df[{hc}].corr()",
                "st.plotly_chart(px.imshow(_corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)",
            ]
        if "box" in charts and num_cols:
            n = num_cols[0].replace("'", "\'")
            carg = f", color='{cat_cols[0]}'" if cat_cols else ""
            lines += [
                "st.subheader('Box Plot')",
                f"st.plotly_chart(px.box(df, y='{n}'{carg}), use_container_width=True)",
            ]
        lines += [
            "",
            "with st.expander('Ham Veriyi Goster'):",
            "    st.dataframe(df, use_container_width=True)",
            "    st.download_button('CSV Indir', df.to_csv(index=False).encode('utf-8-sig'), 'veri.csv', 'text/csv')",
        ]
        return "\n".join(lines)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, api_key=""):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Ayarlar")
        self.setMinimumWidth(480)
        self.setStyleSheet(DARK_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("API Ayarları")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #e0e0e0;")
        layout.addWidget(title)

        group = QGroupBox("Anthropic Claude API")
        form = QFormLayout(group)
        form.setSpacing(12)

        self.api_input = QLineEdit()
        self.api_input.setPlaceholderText("sk-ant-api03-...")
        self.api_input.setText(api_key)
        self.api_input.setEchoMode(QLineEdit.Password)
        form.addRow("API Key:", self.api_input)

        hint = QLabel("🔗 <a href='https://console.anthropic.com' style='color:#3b82f6'>console.anthropic.com</a>'dan alabilirsiniz")
        hint.setOpenExternalLinks(True)
        hint.setStyleSheet("color: #64748b; font-size: 11px;")
        form.addRow("", hint)

        layout.addWidget(group)

        btns = QHBoxLayout()
        cancel_btn = QPushButton("İptal")
        cancel_btn.setObjectName("ghostBtn")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("💾 Kaydet")
        save_btn.clicked.connect(self.accept)
        btns.addWidget(cancel_btn)
        btns.addWidget(save_btn)
        layout.addLayout(btns)

    def get_api_key(self):
        return self.api_input.text().strip()


# ─────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────
class DataWizardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df_original = None
        self.df_clean = None
        self.api_key = self._load_api_key()
        self.worker = None
        self.zip_path = None
        
        self.setWindowTitle("⚡ DataWizard — Excel Analiz & Streamlit Üretici")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)
        
        self._build_ui()
        self._update_status("Hazır — Excel dosyası yükleyin")

    def _load_api_key(self):
        config_path = Path.home() / ".datawizard_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text()).get("api_key", "")
            except:
                return ""
        return ""

    def _save_api_key(self, key):
        config_path = Path.home() / ".datawizard_config.json"
        config_path.write_text(json.dumps({"api_key": key}))

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Header ──
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f1117, stop:0.5 #1a1f35, stop:1 #0f1117);
                border-bottom: 1px solid #2a2d3a;
                min-height: 64px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 12, 24, 12)

        logo = QLabel("⚡ DataWizard")
        logo.setStyleSheet("font-size: 22px; font-weight: 800; color: #3b82f6; letter-spacing: -0.5px;")
        
        subtitle = QLabel("Excel Analiz • Veri Temizleme • Streamlit Üretici")
        subtitle.setStyleSheet("color: #64748b; font-size: 12px; margin-left: 12px;")

        header_layout.addWidget(logo)
        header_layout.addWidget(subtitle)
        header_layout.addStretch()

        settings_btn = QPushButton("⚙️ API Ayarları")
        settings_btn.setObjectName("ghostBtn")
        settings_btn.setFixedWidth(140)
        settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_btn)

        main_layout.addWidget(header)

        # ── Content ──
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        # Left panel (upload + options)
        left_panel = self._build_left_panel()
        left_panel.setFixedWidth(300)
        content_layout.addWidget(left_panel)

        # Right panel (tabs)
        right_panel = self._build_right_panel()
        content_layout.addWidget(right_panel)

        main_layout.addWidget(content)

        # ── Status Bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # Upload section
        upload_group = QGroupBox("📂 Dosya")
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(8)

        self.upload_btn = QPushButton("📂 Excel / CSV Yükle")
        self.upload_btn.setFixedHeight(44)
        self.upload_btn.clicked.connect(self._upload_file)
        upload_layout.addWidget(self.upload_btn)

        self.file_label = QLabel("Dosya seçilmedi")
        self.file_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self.file_label.setWordWrap(True)
        upload_layout.addWidget(self.file_label)

        self.analyze_btn = QPushButton("🤖 Analiz Et + Temizle + Streamlit")
        self.analyze_btn.setFixedHeight(44)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._analyze_with_claude)
        upload_layout.addWidget(self.analyze_btn)

        layout.addWidget(upload_group)

        # Cleaning options
        clean_group = QGroupBox("🧹 Temizlik Seçenekleri")
        clean_layout = QVBoxLayout(clean_group)
        clean_layout.setSpacing(6)

        self.clean_options = {}
        options = [
            ("remove_duplicates", "Duplicate satırları sil"),
            ("remove_empty_rows", "Boş satırları sil"),
            ("remove_empty_cols", "Boş kolonları sil"),
            ("fill_numeric", "Sayısal boşluk → medyan"),
            ("fill_text", "Metin boşluk → 'Bilinmiyor'"),
            ("fix_turkish", "Türkçe karakter düzelt"),
            ("fix_numeric_format", "Virgül→Nokta (sayısal)"),
            ("fix_dates", "Tarih formatı standardize"),
            ("strip_whitespace", "Baş/son boşlukları temizle"),
            ("normalize_columns", "Kolon adları normalize"),
        ]
        
        for key, label in options:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.clean_options[key] = cb
            clean_layout.addWidget(cb)

        layout.addWidget(clean_group)

        # Clean & Download buttons
        self.clean_btn = QPushButton("🧹 Veriyi Temizle")
        self.clean_btn.setFixedHeight(44)
        self.clean_btn.setEnabled(False)
        self.clean_btn.clicked.connect(self._clean_data)
        layout.addWidget(self.clean_btn)

        self.download_btn = QPushButton("⬇️ Temiz Veriyi İndir")
        self.download_btn.setObjectName("successBtn")
        self.download_btn.setFixedHeight(44)
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._download_clean)
        layout.addWidget(self.download_btn)

        layout.addStretch()
        return panel

    def _build_right_panel(self):
        self.tabs = QTabWidget()

        # Tab 1: Data Preview
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(12, 12, 12, 12)

        self.stats_label = QLabel("Veri yüklendikten sonra istatistikler burada görünecek")
        self.stats_label.setStyleSheet("color: #64748b; font-size: 12px; padding: 8px;")
        preview_layout.addWidget(self.stats_label)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        preview_layout.addWidget(self.table)

        self.tabs.addTab(preview_tab, "📊 Veri Önizleme")

        # Tab 2: Claude Analysis
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(12, 12, 12, 12)

        self.analysis_text = QTextEdit()
        self.analysis_text.setPlaceholderText("Claude analizi burada görünecek...\n\nÖnce Excel dosyası yükleyin, sonra '🤖 Claude ile Analiz Et' butonuna tıklayın.")
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)

        self.tabs.addTab(analysis_tab, "🤖 Claude Analizi")

        # Tab 3: Streamlit Generator
        streamlit_tab = self._build_streamlit_tab()
        self.tabs.addTab(streamlit_tab, "⚡ Streamlit Üretici")

        # Tab 4: Console / Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(12, 12, 12, 12)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("İşlem logları burada görünecek...")
        log_layout.addWidget(self.log_text)

        self.tabs.addTab(log_tab, "📋 Log")

        return self.tabs

    def _build_streamlit_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Chart selection
        chart_group = QGroupBox("📈 Grafik Türleri")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setSpacing(6)

        self.chart_options = {}
        charts = [
            ("bar", "Bar Grafik (kategorik karşılaştırma)"),
            ("line", "Çizgi Grafik (zaman serisi)"),
            ("scatter", "Scatter Plot (korelasyon)"),
            ("pie", "Pasta Grafik (dağılım)"),
            ("histogram", "Histogram (frekans dağılımı)"),
            ("heatmap", "Isı Haritası (korelasyon matrisi)"),
            ("box", "Box Plot (istatistiksel özet)"),
        ]
        
        for key, label in charts:
            cb = QCheckBox(label)
            cb.setChecked(key in ["bar", "line", "scatter"])
            self.chart_options[key] = cb
            chart_layout.addWidget(cb)

        layout.addWidget(chart_group)

        # Project path
        path_group = QGroupBox("📁 Proje Konumu")
        path_layout = QHBoxLayout(path_group)
        
        self.project_path_input = QLineEdit()
        self.project_path_input.setPlaceholderText("Proje klasörü seçin...")
        self.project_path_input.setText(str(Path.home() / "Desktop" / "streamlit_project"))
        
        browse_btn = QPushButton("📂")
        browse_btn.setFixedWidth(40)
        browse_btn.clicked.connect(self._browse_project_path)
        
        path_layout.addWidget(self.project_path_input)
        path_layout.addWidget(browse_btn)
        layout.addWidget(path_group)

        # Generate button
        self.generate_btn = QPushButton("⚡ Streamlit Projesi Oluştur (Claude ile)")
        self.generate_btn.setFixedHeight(48)
        self.generate_btn.setObjectName("warningBtn")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._generate_streamlit)
        layout.addWidget(self.generate_btn)

        # Output area
        self.streamlit_output = QTextEdit()
        self.streamlit_output.setReadOnly(True)
        self.streamlit_output.setPlaceholderText("Oluşturulan proje bilgileri burada görünecek...")
        layout.addWidget(self.streamlit_output)

        # ZIP Download button
        self.download_zip_btn = QPushButton("📦 ZIP İndir (Streamlit Projesi + Temiz Veri)")
        self.download_zip_btn.setObjectName("successBtn")
        self.download_zip_btn.setFixedHeight(48)
        self.download_zip_btn.setEnabled(False)
        self.download_zip_btn.clicked.connect(self._download_zip)
        layout.addWidget(self.download_zip_btn)

        # Run button
        self.run_btn = QPushButton("▶️ Streamlit'i Çalıştır")
        self.run_btn.setObjectName("ghostBtn")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_streamlit)
        layout.addWidget(self.run_btn)

        return tab

    # ─────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────
    def _upload_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel / CSV Dosyası Seç", "",
            "Veri Dosyaları (*.xlsx *.xls *.csv);;Tümü (*.*)"
        )
        if not path:
            return
        
        try:
            self._log(f"📂 Dosya yükleniyor: {path}")
            if path.endswith(".csv"):
                # Try different encodings
                for enc in ['utf-8', 'latin-1', 'cp1254', 'iso-8859-9']:
                    try:
                        self.df_original = pd.read_csv(path, encoding=enc)
                        break
                    except:
                        continue
            else:
                self.df_original = pd.read_excel(path)
            
            self.df_clean = None
            fname = Path(path).name
            self.file_label.setText(f"✅ {fname}\n{len(self.df_original)} satır × {len(self.df_original.columns)} kolon")
            self.file_label.setStyleSheet("color: #22c55e; font-size: 11px;")
            
            self._show_dataframe(self.df_original)
            self._update_stats(self.df_original)
            
            self.analyze_btn.setEnabled(True)
            self.clean_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
            
            self._update_status(f"✅ {fname} yüklendi — {len(self.df_original)} satır, {len(self.df_original.columns)} kolon")
            self._log(f"✅ Başarıyla yüklendi: {len(self.df_original)} satır × {len(self.df_original.columns)} kolon")
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Dosya yüklenirken hata:\n{str(e)}")
            self._log(f"❌ Hata: {str(e)}")

    def _analyze_with_claude(self):
        if self.df_original is None:
            return
        chart_types = [k for k, cb in self.chart_options.items() if cb.isChecked()]
        project_path = self.project_path_input.text()
        if not project_path:
            QMessageBox.warning(self, "Uyarı", "Streamlit sekmesinden proje klasörü seçin!")
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳ Analiz + Temizlik + Streamlit...")
        self.tabs.setCurrentIndex(1)
        self.analysis_text.setPlainText("⏳ Veri analiz ediliyor, temizleniyor, Streamlit projesi hazırlanıyor...")
        self._show_progress(True)
        self.progress_bar.setValue(0)

        self.worker = SmartAnalysisWorker(self.df_original, project_path, chart_types)
        self.worker.analysis_done.connect(self._on_analysis_done)
        self.worker.cleaning_done.connect(self._on_smart_clean_done)
        self.worker.streamlit_done.connect(self._on_smart_streamlit_done)
        self.worker.progress.connect(self._on_cleaning_progress)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_analysis_done(self, text):
        self.analysis_text.setMarkdown(text)
        self._log("✅ Analiz tamamlandı")

    def _on_smart_clean_done(self, df):
        self.df_clean = df
        self._show_dataframe(df)
        self._update_stats(df)
        self.download_btn.setEnabled(True)
        removed_rows = len(self.df_original) - len(df)
        self._log(f"✅ Temizlik tamamlandı — {removed_rows} satır kaldırıldı")

    def _on_smart_streamlit_done(self, zip_path):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🤖 Analiz Et + Streamlit Üret")
        self.run_btn.setEnabled(True)
        self._show_progress(False)
        self.zip_path = zip_path

        self.tabs.setCurrentIndex(2)
        self.streamlit_output.setPlainText(
            f"✅ Her şey hazır!\n\n"
            f"📦 ZIP: {zip_path}\n\n"
            f"İçerik:\n"
            f"  ├── app.py\n"
            f"  ├── requirements.txt\n"
            f"  └── data/cleaned_data.xlsx\n\n"
            f"ZIP'i indirmek için aşağıdaki butona tıklayın."
        )
        self.download_zip_btn.setEnabled(True)
        self._update_status("✅ Analiz, temizlik ve Streamlit projesi hazır!")
        self._log(f"✅ ZIP hazır: {zip_path}")

    def _clean_data(self):
        if self.df_original is None:
            return
        
        options = {key: cb.isChecked() for key, cb in self.clean_options.items()}
        
        self.clean_btn.setEnabled(False)
        self.clean_btn.setText("⏳ Temizleniyor...")
        self._show_progress(True)
        self.progress_bar.setValue(0)
        
        self.worker = CleaningWorker(self.df_original, options)
        self.worker.finished.connect(self._on_cleaning_done)
        self.worker.progress.connect(self._on_cleaning_progress)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_cleaning_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self._update_status(msg)
        self._log(msg)

    def _on_cleaning_done(self, df):
        self.df_clean = df
        self._show_dataframe(df)
        self._update_stats(df)
        
        removed_rows = len(self.df_original) - len(df)
        removed_cols = len(self.df_original.columns) - len(df.columns)
        
        self.clean_btn.setEnabled(True)
        self.clean_btn.setText("🧹 Veriyi Temizle")
        self.download_btn.setEnabled(True)
        self._show_progress(False)
        
        msg = f"✅ Temizlik tamamlandı — {removed_rows} satır, {removed_cols} kolon kaldırıldı"
        self._update_status(msg)
        self._log(msg)
        self.tabs.setCurrentIndex(0)

    def _download_clean(self):
        if self.df_clean is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Temiz Veriyi Kaydet", "temiz_veri.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)"
        )
        if not path:
            return
        
        try:
            if path.endswith(".csv"):
                self.df_clean.to_csv(path, index=False, encoding='utf-8-sig')
            else:
                self.df_clean.to_excel(path, index=False)
            
            self._update_status(f"✅ Kaydedildi: {path}")
            self._log(f"✅ Dosya kaydedildi: {path}")
            QMessageBox.information(self, "Başarılı", f"Dosya kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))

    def _generate_streamlit(self):
        df = self.df_clean if self.df_clean is not None else self.df_original
        if df is None:
            QMessageBox.warning(self, "Uyarı", "Önce Excel dosyası yükleyin!")
            return
        chart_types = [k for k, cb in self.chart_options.items() if cb.isChecked()]
        project_path = self.project_path_input.text()
        if not project_path:
            QMessageBox.warning(self, "Uyarı", "Proje klasörü seçin!")
            return
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ Streamlit oluşturuluyor...")
        self._show_progress(True)
        self.worker = StreamlitGeneratorWorker(df, project_path, self.api_key, chart_types)
        self.worker.finished.connect(self._on_streamlit_done)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_streamlit_done(self, path):
        import zipfile as zf
        zip_path = path + ".zip"
        with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
            for f in Path(path).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(Path(path).parent))
        self.zip_path = zip_path
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("⚡ Streamlit Projesi Oluştur")
        self.run_btn.setEnabled(True)
        self.download_zip_btn.setEnabled(True)
        self._show_progress(False)
        self.streamlit_output.setPlainText(
            f"✅ Streamlit projesi hazır!\n\nZIP: {zip_path}\n\n"
            f"İndirmek için 'ZIP İndir' butonuna tıklayın."
        )
        self._update_status("✅ Streamlit projesi oluşturuldu")
        self._log(f"✅ Proje hazır: {path}")

    def _run_streamlit(self):
        path = self.project_path_input.text()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Uyarı", "Proje klasörü bulunamadı!")
            return
        
        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["streamlit", "run", "app.py"],
                    cwd=path,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            elif sys.platform == "darwin":
                # macOS → yeni Terminal penceresi aç
                script = f'tell application "Terminal" to do script "cd \\"{path}\\" && streamlit run app.py"'
                subprocess.Popen(["osascript", "-e", script])
            else:
                # Linux
                subprocess.Popen(
                    ["streamlit", "run", "app.py"],
                    cwd=path
                )
            self._update_status("▶️ Streamlit başlatılıyor...")
            self._log("▶️ Streamlit çalıştırıldı")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Streamlit başlatılamadı:\n{str(e)}\n\nTerminalden manuel çalıştırın:\ncd \"{path}\"\nstreamlit run app.py")

    def _download_zip(self):
        if not hasattr(self, 'zip_path') or not self.zip_path:
            QMessageBox.warning(self, "Uyarı", "Önce proje oluşturun!")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "ZIP Dosyasını Kaydet",
            str(Path.home() / "Desktop" / "streamlit_project.zip"),
            "ZIP Dosyası (*.zip)"
        )
        if not save_path:
            return
        import shutil
        shutil.copy2(self.zip_path, save_path)
        self._update_status(f"✅ ZIP kaydedildi: {save_path}")
        self._log(f"✅ ZIP kaydedildi: {save_path}")
        QMessageBox.information(self, "İndirildi!",
            f"ZIP dosyası kaydedildi:\n{save_path}\n\n"
            f"Açmak için:\n  pip install -r requirements.txt\n  streamlit run app.py")

    def _browse_project_path(self):
        path = QFileDialog.getExistingDirectory(self, "Proje Klasörü Seç", str(Path.home() / "Desktop"))
        if path:
            self.project_path_input.setText(path + "/streamlit_project")

    def _open_settings(self):
        dialog = SettingsDialog(self, self.api_key)
        if dialog.exec_() == QDialog.Accepted:
            self.api_key = dialog.get_api_key()
            self._save_api_key(self.api_key)
            self._log("✅ API Key kaydedildi")

    def _prompt_api_key(self):
        QMessageBox.information(
            self, "API Key Gerekli",
            "Claude özelliklerini kullanmak için Anthropic API Key gerekli.\n\n"
            "⚙️ API Ayarları butonuna tıklayarak ekleyebilirsiniz."
        )
        self._open_settings()

    def _on_worker_error(self, msg):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🤖 Claude ile Analiz Et")
        self.clean_btn.setEnabled(True)
        self.clean_btn.setText("🧹 Veriyi Temizle")
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("⚡ Streamlit Projesi Oluştur (Claude ile)")
        self._show_progress(False)
        self._update_status(f"❌ Hata: {msg}")
        self._log(f"❌ Hata: {msg}")
        QMessageBox.critical(self, "Hata", msg)

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _show_dataframe(self, df):
        display_df = df.head(500)
        self.table.setRowCount(len(display_df))
        self.table.setColumnCount(len(display_df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in display_df.columns])
        
        for i, row in display_df.iterrows():
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val) if pd.notna(val) else "")
                if pd.isna(val):
                    item.setForeground(QColor("#dc2626"))
                self.table.setItem(i if isinstance(i, int) else list(display_df.index).index(i), j, item)
        
        self.table.resizeColumnsToContents()

    def _update_stats(self, df):
        nulls = df.isnull().sum().sum()
        dups = df.duplicated().sum()
        quality = max(0, 100 - int((nulls / max(df.size, 1)) * 100) - int((dups / max(len(df), 1)) * 20))
        
        self.stats_label.setText(
            f"  📊 {len(df):,} satır × {len(df.columns)} kolon  "
            f"│  ⚠️ {nulls:,} boş değer  "
            f"│  🔁 {dups:,} duplicate  "
            f"│  ✨ Kalite: %{quality}"
        )
        self.stats_label.setStyleSheet("color: #94a3b8; font-size: 12px; padding: 8px; background: #1e2433; border-radius: 6px;")

    def _update_status(self, msg):
        self.status_bar.showMessage(f"  {msg}")

    def _show_progress(self, show):
        self.progress_bar.setVisible(show)
        if not show:
            self.progress_bar.setValue(0)

    def _log(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DataWizard")
    
    window = DataWizardApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
