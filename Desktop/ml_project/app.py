"""
Streamlit UI — ML Trainer & Predictor  (modern dark theme)
Run with: streamlit run app.py
"""
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sample_data import generate_sample_data
from data_loader import load_file, ACCEPTED_EXTENSIONS
from ml_pipeline import (
    MODEL_CATALOGUE,
    train,
    predict_single,
    predict_proba,
    get_feature_importances,
    audit_columns,
)
from theme import (
    apply_theme, page_header, section_title,
    metric_card, metrics_row, status_badge,
    empty_state, divider, tag, C,
)
import supabase_client as sb

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C["surface"],
        font=dict(family="Inter", color=C["muted2"], size=12),
        xaxis=dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"]),
        colorway=[C["primary"], C["secondary"], C["success"], C["warning"], C["error"], C["violet"]],
        margin=dict(l=16, r=16, t=40, b=16),
    )
)

def styled_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C["surface"],
        font=dict(family="Inter", color=C["muted2"], size=12),
        margin=dict(l=16, r=16, t=44, b=16),
        title_font=dict(size=14, color=C["text"], family="Inter"),
    )
    fig.update_xaxes(gridcolor=C["border"], linecolor=C["border2"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border2"], zerolinecolor=C["border"])
    return fig


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "df":            None,
    "result":        None,
    "trained":       False,
    "col_overrides": {},
    "target_col":    None,
    "feature_cols":  [],
    "filename":      "sample data",
    "cfg": {
        "model_name":               list(MODEL_CATALOGUE.keys())[0],
        "test_pct":                 20,
        "task_override":            "Auto-detect",
        "classification_threshold": 15,
        "ohe_max_cardinality":      20,
        "high_card_threshold":      50,
        "viz_sample":               5000,
        "bar_chart_limit":          30,
        "predict_cols_per_row":     3,
        "model_params":             {},
    },
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pure helpers ──────────────────────────────────────────────────────────────
_ICONS = ["🟢", "🔵", "🟡", "🟠", "🔴", "🟣", "🟤", "⚫", "⚪"]

def class_icon(cls, all_classes):
    idx = sorted([str(c) for c in all_classes]).index(str(cls)) \
          if str(cls) in [str(c) for c in all_classes] else 0
    return _ICONS[idx % len(_ICONS)]

def col_role_badge(role):
    return {
        "numeric":        "🔵 numeric",
        "bool":           "🟤 boolean",
        "datetime":       "🕐 datetime",
        "low-card-cat":   "🟣 categorical",
        "high-card-cat":  "🟠 high-cardinality",
        "very-high-card": "🔴 very-high-cardinality",
        "constant":       "⚪ constant (will be dropped)",
    }.get(role, role)

def suggest_target(df):
    n = len(df)
    best_idx, best_score = len(df.columns) - 1, -1
    for i, col in enumerate(df.columns):
        s     = df[col]
        ratio = s.nunique() / max(n, 1)
        score = (1 - ratio) * (2 if not pd.api.types.is_numeric_dtype(s) else 1)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

def infer_widget_type(series, role, override, cfg_dict):
    if override == "categorical":
        return "text_input" if series.nunique() > cfg_dict["high_card_threshold"] else "selectbox"
    if override == "numeric":
        return "float"
    if role == "bool":         return "bool"
    if role == "datetime":     return "text_input"
    if role in ("low-card-cat","high-card-cat"): return "selectbox"
    if role == "very-high-card": return "text_input"
    if role == "numeric":
        return "integer" if pd.api.types.is_integer_dtype(series) else "float"
    return "selectbox" if series.nunique() <= cfg_dict["ohe_max_cardinality"] else "text_input"

def reset_on_new_data(df):
    st.session_state.df            = df
    st.session_state.result        = None
    st.session_state.trained       = False
    st.session_state.col_overrides = {}
    idx = suggest_target(df)
    st.session_state.target_col    = df.columns[idx]
    st.session_state.feature_cols  = [c for c in df.columns if c != st.session_state.target_col]

def cfg():
    return st.session_state.cfg


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 8px 0">
        <div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            🧠 ML Studio
        </div>
        <div style="font-size:11px;color:#475569;font-weight:500;margin-top:2px;letter-spacing:0.05em">
            INTELLIGENT TRAINING PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{C["border"]};margin:8px 0 20px 0"></div>',
                unsafe_allow_html=True)

    # ── Status badge ──────────────────────────────────────────────────────────
    if st.session_state.trained:
        r = st.session_state.result
        m = r["metrics"]
        perf = f"{m['accuracy']*100:.1f}%" if m["task"] == "classification" else f"R²={m['r2']}"
        status_badge(f"Model trained · {perf}", "success")
    elif st.session_state.df is not None:
        status_badge("Dataset loaded · ready to train", "info")
    else:
        status_badge("No data loaded", "warning")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:11px;color:{C["muted"]};font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">01 · Load Data</div>', unsafe_allow_html=True)

    source = st.radio("Source", ["Sample data", "Upload dataset"], label_visibility="collapsed")

    if source == "Sample data":
        n_samples = st.number_input("Rows", min_value=50, max_value=100_000, value=300, step=50)
        if st.button("Generate Dataset", use_container_width=True):
            reset_on_new_data(generate_sample_data(n=int(n_samples)))
            st.session_state["filename"] = "sample data"
            st.success(f"Generated {n_samples:,} rows.")
    else:
        fmt_str = " · ".join(f".{e}" for e in ACCEPTED_EXTENSIONS)
        st.markdown(f'<div style="font-size:11px;color:{C["muted"]};margin-bottom:6px">{fmt_str}</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=ACCEPTED_EXTENSIONS, label_visibility="collapsed")
        if uploaded:
            with st.spinner(f"Reading {uploaded.name}…"):
                try:
                    df_new, sheets = load_file(uploaded)
                    if sheets:
                        st.session_state["_pending_bytes"]  = uploaded.getvalue() if hasattr(uploaded,"getvalue") else b""
                        st.session_state["_sheets"]         = sheets
                    else:
                        reset_on_new_data(df_new)
                        st.session_state["filename"] = uploaded.name
                        st.success(f"{len(df_new):,} rows · {len(df_new.columns)} cols")
                except Exception as e:
                    st.error(f"Read error: {e}")

        if st.session_state.get("_sheets"):
            picked = st.selectbox("Sheet", st.session_state["_sheets"])
            if st.button("Load Sheet", use_container_width=True):
                raw      = st.session_state.get("_pending_bytes", b"")
                df_sheet = pd.read_excel(io.BytesIO(raw), sheet_name=picked)
                reset_on_new_data(df_sheet)
                st.session_state["filename"] = f"{picked} (Excel)"
                st.session_state.pop("_sheets", None)
                st.session_state.pop("_pending_bytes", None)
                st.success(f"Loaded '{picked}'")

    # ── 2. Configure ──────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df   = st.session_state.df
        cols = df.columns.tolist()

        st.markdown(f'<div style="height:1px;background:{C["border"]};margin:20px 0 16px 0"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:{C["muted"]};font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">02 · Configure</div>', unsafe_allow_html=True)

        target_idx = cols.index(st.session_state.target_col) \
                     if st.session_state.target_col in cols else suggest_target(df)
        target_col = st.selectbox("Target column", cols, index=target_idx)
        st.session_state.target_col = target_col

        avail        = [c for c in cols if c != target_col]
        saved        = [c for c in st.session_state.feature_cols if c in avail]
        feature_cols = st.multiselect("Feature columns", avail, default=saved or avail)
        st.session_state.feature_cols = feature_cols

        model_name = st.selectbox("Model",  list(MODEL_CATALOGUE.keys()),
                                  index=list(MODEL_CATALOGUE.keys()).index(cfg()["model_name"]))
        cfg()["model_name"] = model_name

        task_choice = st.radio("Task", ["Auto-detect","Classification","Regression"],
                               index=["Auto-detect","Classification","Regression"].index(cfg()["task_override"]))
        cfg()["task_override"] = task_choice

        # Advanced settings
        with st.expander("⚙️ Advanced settings"):
            cfg()["test_pct"] = st.slider("Test split %", 5, 50, cfg()["test_pct"])
            cfg()["classification_threshold"] = st.slider(
                "Auto-classify if numeric unique ≤", 2, 100, cfg()["classification_threshold"])
            cfg()["ohe_max_cardinality"] = st.slider(
                "One-hot encode ≤ N unique values", 2, 200, cfg()["ohe_max_cardinality"])
            cfg()["high_card_threshold"] = st.slider(
                "Ordinal encode ≤ N unique values",
                cfg()["ohe_max_cardinality"], 500, cfg()["high_card_threshold"])
            cfg()["viz_sample"] = st.number_input(
                "Viz sample size", 100, 100_000, cfg()["viz_sample"], 500)
            cfg()["bar_chart_limit"] = st.number_input(
                "Max bars in charts", 5, 500, cfg()["bar_chart_limit"], 5)
            cfg()["predict_cols_per_row"] = st.slider(
                "Predict form columns per row", 1, 6, cfg()["predict_cols_per_row"])

        # Hyperparameters
        with st.expander("🔬 Hyperparameters"):
            spec   = MODEL_CATALOGUE[model_name]["params"]
            saved_p = cfg()["model_params"].get(model_name, {})
            new_p   = {}
            for pname, pspec in spec.items():
                default = saved_p.get(pname, pspec["default"])
                if pspec["type"] == "int":
                    new_p[pname] = st.slider(pspec["label"], pspec["min"], pspec["max"],
                                             int(default), pspec.get("step",1), key=f"hp_{model_name}_{pname}")
                else:
                    new_p[pname] = st.slider(pspec["label"], float(pspec["min"]), float(pspec["max"]),
                                             float(default), key=f"hp_{model_name}_{pname}")
            cfg()["model_params"][model_name] = new_p

        # Column overrides
        with st.expander("🔧 Column type overrides"):
            overrides = {}
            audit_now = audit_columns(df, feature_cols, cfg()["ohe_max_cardinality"],
                                       cfg()["high_card_threshold"]) if feature_cols else {}
            for col in feature_cols:
                role   = audit_now.get(col, {}).get("role", "?")
                choice = st.selectbox(f"{col}  ({col_role_badge(role)})",
                                      ["auto","categorical","numeric"], key=f"ovr_{col}")
                if choice != "auto":
                    overrides[col] = choice
            st.session_state.col_overrides = overrides

        # Train
        st.markdown(f'<div style="height:1px;background:{C["border"]};margin:20px 0 16px 0"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:{C["muted"]};font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">03 · Train</div>', unsafe_allow_html=True)

        if st.button("🚀  Train Model", use_container_width=True, type="primary"):
            if not feature_cols:
                st.error("Select at least one feature.")
            else:
                task_map = {"Auto-detect": None, "Classification": "classification", "Regression": "regression"}
                with st.spinner("Training…"):
                    try:
                        result = train(
                            df=df, feature_cols=feature_cols, target_col=target_col,
                            test_size=cfg()["test_pct"] / 100,
                            model_name=model_name,
                            model_params=cfg()["model_params"].get(model_name, {}),
                            task_override=task_map[task_choice],
                            classification_threshold=cfg()["classification_threshold"],
                            ohe_max_cardinality=cfg()["ohe_max_cardinality"],
                            high_card_threshold=cfg()["high_card_threshold"],
                            col_overrides=st.session_state.col_overrides,
                        )
                        st.session_state.result  = result
                        st.session_state.trained = True
                        m = result["metrics"]
                        if m["task"] == "classification":
                            st.success(f"Accuracy: {m['accuracy']*100:.1f}%")
                        else:
                            st.success(f"R²: {m['r2']}  MAE: {m['mae']}")
                    except Exception as e:
                        st.error(f"{e}")

        if st.session_state.trained:
            buf = io.BytesIO()
            r   = st.session_state.result
            joblib.dump({k: r[k] for k in ("pipeline","label_encoder","feature_cols",
                                            "target_col","task","model_name","model_params")}, buf)
            st.download_button("💾  Export model (.joblib)", data=buf.getvalue(),
                               file_name="model.joblib", use_container_width=True)

            # ── Save to Supabase ───────────────────────────────────────────
            if sb.is_configured():
                st.markdown(
                    f'<div style="height:1px;background:{C["border"]};margin:16px 0 12px 0"></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:11px;color:{C["muted"]};font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">04 · Save</div>',
                    unsafe_allow_html=True,
                )
                r = st.session_state.result
                default_name = f"{r['model_name']} · {r['target_col']}"
                exp_name = st.text_input("Experiment name", value=default_name, key="exp_name_input")
                if st.button("☁️  Save to Supabase", use_container_width=True):
                    with st.spinner("Saving…"):
                        exp_id = sb.save_experiment(st.session_state.result, exp_name)
                    if exp_id:
                        st.success("Saved!")
                    else:
                        st.error("Save failed — check Supabase credentials.")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_data, tab_quality, tab_metrics, tab_predict, tab_history = st.tabs([
    "  📊  Data  ", "  🔍  Quality  ", "  📈  Metrics  ", "  🔮  Predict  ", "  ☁️  History  "
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    if st.session_state.df is None:
        empty_state("📂", "No dataset loaded",
                    "Generate sample data or upload a file using the sidebar to get started.")
    else:
        df     = st.session_state.df
        n_rows = len(df)
        n_cols = len(df.columns)

        page_header("Dataset Explorer",
                    f"{n_rows:,} rows · {n_cols} columns · {st.session_state.get('filename', 'sample data')}")

        # Top metrics row
        metrics_row([
            {"title": "Total rows",     "value": f"{n_rows:,}",
             "accent": C["primary"]},
            {"title": "Columns",        "value": str(n_cols),
             "accent": C["secondary"]},
            {"title": "Missing cells",  "value": f"{int(df.isnull().sum().sum()):,}",
             "subtitle": f"{df.isnull().mean().mean()*100:.1f}% of all cells",
             "accent": C["warning"]},
            {"title": "Duplicate rows", "value": f"{int(df.duplicated().sum()):,}",
             "accent": C["violet"]},
        ])

        divider()

        # Data preview
        section_title("Data Preview")
        viz_n = int(cfg()["viz_sample"])
        disp  = df if n_rows <= viz_n * 2 else df.sample(min(10_000, n_rows), random_state=0)
        if n_rows > len(disp):
            st.caption(f"Showing {len(disp):,} sampled rows of {n_rows:,}")
        st.dataframe(disp.head(100), use_container_width=True, height=280)

        divider()

        # Distribution
        section_title("Column Distribution")
        vis_col = st.selectbox("Select column", df.columns.tolist(), key="vis_col",
                               label_visibility="collapsed")
        col_s   = df[vis_col]
        viz_df  = df if n_rows <= viz_n else df.sample(viz_n, random_state=0)
        bar_lim = int(cfg()["bar_chart_limit"])
        is_num  = pd.api.types.is_numeric_dtype(col_s) and not pd.api.types.is_bool_dtype(col_s)

        if not is_num or col_s.nunique() <= cfg()["ohe_max_cardinality"]:
            counts = col_s.value_counts().head(bar_lim).reset_index()
            counts.columns = [vis_col, "count"]
            if col_s.nunique() > bar_lim:
                st.caption(f"Top {bar_lim} of {col_s.nunique():,} unique values")
            fig = px.bar(counts, x=vis_col, y="count",
                         color="count", color_continuous_scale=[[0,"#1a2236"],[1,C["primary"]]],
                         title=f"Distribution · {vis_col}")
        else:
            n_bins = st.slider("Bins", 5, 200, 40, key="hist_bins")
            fig    = px.histogram(viz_df, x=vis_col, nbins=n_bins,
                                  color_discrete_sequence=[C["primary"]],
                                  title=f"Distribution · {vis_col}")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

        # Correlation heatmap
        num_df = df.select_dtypes(include=np.number)
        if len(num_df.columns) > 1:
            divider()
            section_title("Correlation Matrix", "Pearson correlations between numeric columns")
            corr = num_df.corr()
            fig2 = px.imshow(corr, text_auto=".2f",
                             color_continuous_scale=[[0,C["error"]],[0.5,C["surface2"]],[1,C["primary"]]],
                             zmin=-1, zmax=1, title="Pearson Correlation")
            st.plotly_chart(styled_fig(fig2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════
with tab_quality:
    if st.session_state.df is None:
        empty_state("🔍", "No dataset loaded", "Load data to run quality checks.")
    else:
        df    = st.session_state.df
        audit = audit_columns(df, df.columns.tolist(),
                               cfg()["ohe_max_cardinality"], cfg()["high_card_threshold"])

        page_header("Data Quality Report", "Automated column-level analysis")

        # Summary metrics
        n_const  = sum(1 for i in audit.values() if i["role"] == "constant")
        n_miss   = sum(1 for i in audit.values() if i["null_pct"] > 0)
        n_hcard  = sum(1 for i in audit.values() if i["role"] in ("very-high-card","high-card-cat"))
        metrics_row([
            {"title": "Total columns",         "value": str(len(audit)),     "accent": C["primary"]},
            {"title": "With missing values",   "value": str(n_miss),         "accent": C["warning"]},
            {"title": "High-cardinality cols", "value": str(n_hcard),        "accent": C["secondary"]},
            {"title": "Constant (auto-drop)",  "value": str(n_const),        "accent": C["error"]},
        ])

        divider()
        section_title("Column Report")

        rows = [{"Column": col, "Detected type": info["dtype"],
                 "Role": col_role_badge(info["role"]),
                 "Unique values": info["n_unique"],
                 "Missing %": f"{info['null_pct']}%"}
                for col, info in audit.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=300)

        # Warnings
        warns = []
        for col, info in audit.items():
            r = info["role"]
            if r == "constant":
                warns.append(("⚪", col, "Only 1 unique value — will be auto-dropped", "warning"))
            if r == "very-high-card":
                warns.append(("🔴", col, f"{info['n_unique']:,} unique values — ordinal encoding applied", "error"))
            if r == "datetime":
                warns.append(("🕐", col, "Datetime — converted to Unix timestamp automatically", "info"))
            if info["null_pct"] > 30:
                warns.append(("⚠️", col, f"{info['null_pct']}% missing — will be auto-imputed", "warning"))

        if warns:
            divider()
            section_title("Auto-actions & Warnings")
            for icon, col, msg, kind in warns:
                color = {"warning": C["warning"], "error": C["error"],
                         "info": C["secondary"], "success": C["success"]}.get(kind, C["muted2"])
                st.markdown(f"""
                <div style="
                    display:flex;align-items:flex-start;gap:12px;
                    background:{color}0d;border:1px solid {color}30;
                    border-radius:10px;padding:12px 16px;margin-bottom:8px
                ">
                    <span style="font-size:16px">{icon}</span>
                    <div>
                        <span style="color:{C['text']};font-weight:600;font-size:13px">{col}</span>
                        <span style="color:{C['muted2']};font-size:13px"> — {msg}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{C['success']}0d;border:1px solid {C['success']}30;
                        border-radius:12px;padding:16px 20px;display:flex;align-items:center;gap:12px">
                <span style="font-size:20px">✅</span>
                <span style="color:{C['text']};font-size:14px">No significant data quality issues detected.</span>
            </div>
            """, unsafe_allow_html=True)

        # Missing values chart
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            divider()
            section_title("Missing Values")
            null_df = null_counts[null_counts > 0].reset_index()
            null_df.columns = ["column", "count"]
            null_df["pct"]  = (null_df["count"] / len(df) * 100).round(1)
            null_df         = null_df.sort_values("pct", ascending=False)
            fig_null = px.bar(null_df, x="column", y="pct",
                              color="pct",
                              color_continuous_scale=[[0,"rgba(245,158,11,0.53)"],[1,C["error"]]],
                              title="Missing % per column",
                              labels={"pct": "Missing %", "column": ""})
            st.plotly_chart(styled_fig(fig_null), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRICS
# ════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if not st.session_state.trained:
        empty_state("📈", "No model trained yet",
                    "Configure your features and target in the sidebar, then hit Train Model.")
    else:
        result  = st.session_state.result
        metrics = result["metrics"]
        task    = result["task"]

        page_header(
            f"{result['model_name']}",
            f"Target: {result['target_col']}  ·  Task: {task}  ·  Features: {len(result['feature_cols'])}"
        )

        # Hyperparams row
        if result.get("model_params"):
            tags_html = "  ".join(tag(f"{k}={v}") for k, v in result["model_params"].items())
            st.markdown(f'<div style="margin-bottom:20px">{tags_html}</div>', unsafe_allow_html=True)

        if result.get("constant_cols"):
            st.markdown(f"""
            <div style="background:{C['warning']}0d;border:1px solid {C['warning']}30;
                        border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px">
                ⚠️ <span style="color:{C['text']}">Dropped constant columns:</span>
                <span style="color:{C['muted2']}">{', '.join(result['constant_cols'])}</span>
            </div>
            """, unsafe_allow_html=True)

        if task == "classification":
            acc = metrics["accuracy"]

            # Top metrics
            metrics_row([
                {"title": "Accuracy",   "value": f"{acc*100:.1f}%",
                 "subtitle": f"on {int(sum(r.get('support',0) for r in metrics['report'].values() if isinstance(r,dict) and 'support' in r))} test samples",
                 "accent": C["success"] if acc >= 0.8 else C["warning"] if acc >= 0.6 else C["error"]},
                {"title": "Classes",    "value": str(len(metrics["classes"])),    "accent": C["primary"]},
                {"title": "Features",   "value": str(len(result["feature_cols"])), "accent": C["secondary"]},
                {"title": "Test split", "value": f"{cfg()['test_pct']}%",          "accent": C["violet"]},
            ])

            divider()
            section_title("Per-class Performance")
            report = metrics["report"]
            tbl = [{"Class": cls,
                    "Precision": round(report.get(str(cls),{}).get("precision",0),3),
                    "Recall":    round(report.get(str(cls),{}).get("recall",   0),3),
                    "F1":        round(report.get(str(cls),{}).get("f1-score", 0),3),
                    "Support":   int(report.get(str(cls),{}).get("support",    0))}
                   for cls in metrics["classes"]]
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

            divider()
            section_title("Confusion Matrix")
            cm      = np.array(metrics["confusion_matrix"])
            classes = [str(c) for c in metrics["classes"]]
            fig_cm  = px.imshow(cm, x=classes, y=classes, text_auto=True,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                color_continuous_scale=[[0,C["surface2"]],[1,C["primary"]]],
                                title="Confusion Matrix")
            st.plotly_chart(styled_fig(fig_cm), use_container_width=True)

        else:
            metrics_row([
                {"title": "R² Score",    "value": str(metrics["r2"]),
                 "subtitle": "1.0 = perfect fit",
                 "accent": C["success"] if metrics["r2"] >= 0.8 else C["warning"]},
                {"title": "MAE",         "value": str(metrics["mae"]),  "accent": C["secondary"]},
                {"title": "Features",    "value": str(len(result["feature_cols"])), "accent": C["primary"]},
                {"title": "Test split",  "value": f"{cfg()['test_pct']}%",          "accent": C["violet"]},
            ])
            divider()
            section_title("Actual vs Predicted")
            y_test, y_pred = result["y_test"], result["y_pred"]
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=list(y_test), y=list(y_pred), mode="markers",
                                       marker=dict(color=C["primary"], size=5, opacity=0.6),
                                       name="Predictions"))
            lo, hi = float(min(y_test)), float(max(y_test))
            fig_s.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                       line=dict(color=C["error"], dash="dash", width=1.5),
                                       name="Perfect fit"))
            fig_s.update_layout(title="Actual vs Predicted",
                                xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(styled_fig(fig_s), use_container_width=True)

        divider()
        section_title("Feature Importances", "Higher = more influence on predictions")
        fi = get_feature_importances(result).head(int(cfg()["bar_chart_limit"]))
        if fi["importance"].sum() > 0:
            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                            color="importance",
                            color_continuous_scale=[[0,"rgba(99,102,241,0.33)"],[1,C["violet"]]],
                            title=f"Top {len(fi)} Feature Importances")
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(styled_fig(fig_fi), use_container_width=True)
        else:
            st.info("Feature importances not available for this model.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if not st.session_state.trained:
        empty_state("🔮", "No model trained yet",
                    "Train a model first to start making predictions.")
    else:
        result       = st.session_state.result
        df           = st.session_state.df
        feature_cols = result["feature_cols"]
        col_audit    = result.get("col_audit", {})
        cpr          = int(cfg()["predict_cols_per_row"])

        page_header(
            f"Predict · {result['target_col']}",
            f"Model: {result['model_name']}  ·  {len(feature_cols)} input features"
        )

        st.markdown(f'<p style="color:{C["muted"]};font-size:13px;margin-bottom:20px">Input values for each feature below. Widget types adapt automatically to the column data type.</p>', unsafe_allow_html=True)

        input_data = {}
        chunks = [feature_cols[i:i+cpr] for i in range(0, len(feature_cols), cpr)]

        for chunk in chunks:
            row_widgets = st.columns(len(chunk))
            for wcol, feat in zip(row_widgets, chunk):
                series   = df[feat] if feat in df.columns else pd.Series(dtype=object)
                role     = col_audit.get(feat, {}).get("role", "")
                override = st.session_state.col_overrides.get(feat)
                wtype    = infer_widget_type(series, role, override, cfg())

                with wcol:
                    if wtype == "bool":
                        input_data[feat] = st.selectbox(feat, [True, False], key=f"p_{feat}")
                    elif wtype == "selectbox":
                        opts = sorted(series.dropna().astype(str).unique().tolist())
                        input_data[feat] = st.selectbox(feat, opts if opts else [""], key=f"p_{feat}")
                    elif wtype == "text_input":
                        ex = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
                        input_data[feat] = st.text_input(feat, value=ex, key=f"p_{feat}",
                                                          help=f"Role: {role}")
                    elif wtype == "integer":
                        vals = series.dropna()
                        mn, mx, med = (int(vals.min()), int(vals.max()), int(vals.median())) if len(vals) else (0,100,0)
                        input_data[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                            value=med, step=1, key=f"p_{feat}")
                    else:
                        vals = pd.to_numeric(series, errors="coerce").dropna()
                        mn, mx, med = (float(vals.min()), float(vals.max()), float(vals.median())) if len(vals) else (0.,1.,.5)
                        input_data[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                            value=med, key=f"p_{feat}")

        divider()
        if st.button("🔮  Run Prediction", type="primary", use_container_width=True):
            try:
                prediction = predict_single(result, input_data)
                task       = result["task"]
                target     = result["target_col"]

                if task == "classification":
                    all_cls = result["metrics"].get("all_classes", result["metrics"]["classes"])
                    icon    = class_icon(str(prediction), [str(c) for c in all_cls])

                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {C['primary']}15, {C['violet']}10);
                        border: 1px solid {C['primary']}40;
                        border-radius: 16px;
                        padding: 28px 32px;
                        text-align: center;
                        margin: 16px 0;
                    ">
                        <div style="font-size:13px;color:{C['muted']};font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
                            Predicted {target}
                        </div>
                        <div style="font-size:52px;font-weight:800;
                                    background:linear-gradient(135deg,{C['primary']},{C['violet']});
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                                    letter-spacing:-2px;line-height:1">
                            {icon} {prediction}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    proba = predict_proba(result, input_data)
                    if proba:
                        divider()
                        section_title("Confidence per Class")
                        prob_df = (pd.DataFrame(list(proba.items()), columns=["Class","Probability"])
                                   .sort_values("Probability", ascending=False))
                        fig_prob = px.bar(prob_df, x="Class", y="Probability",
                                         color="Probability",
                                         color_continuous_scale=[[0,"rgba(99,102,241,0.27)"],[1,C["success"]]],
                                         range_y=[0,1], title="Prediction Confidence")
                        for trace in fig_prob.data:
                            trace.text = [f"{v:.1%}" for v in trace.y]
                            trace.textposition = "outside"
                        st.plotly_chart(styled_fig(fig_prob), use_container_width=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {C['secondary']}15, {C['primary']}10);
                        border: 1px solid {C['secondary']}40;
                        border-radius: 16px;
                        padding: 28px 32px;
                        text-align: center;
                        margin: 16px 0;
                    ">
                        <div style="font-size:13px;color:{C['muted']};font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
                            Predicted {target}
                        </div>
                        <div style="font-size:52px;font-weight:800;color:{C['secondary']};
                                    letter-spacing:-2px;line-height:1">
                            {prediction}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY  (Supabase)
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    page_header("Experiment History", "Saved to Supabase")

    if not sb.is_configured():
        empty_state(
            "☁️", "Supabase not configured",
            "Set SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY in your .env file.",
        )
    else:
        if st.button("🔄  Refresh", key="history_refresh"):
            st.cache_data.clear()

        with st.spinner("Loading experiments…"):
            experiments = sb.load_experiments()

        if not experiments:
            empty_state(
                "📭", "No experiments saved yet",
                "Train a model and click '☁️ Save to Supabase' in the sidebar.",
            )
        else:
            rows = []
            for e in experiments:
                raw_m = e.get("experiment_metrics") or []
                m     = raw_m[0] if isinstance(raw_m, list) and raw_m else (raw_m if isinstance(raw_m, dict) else {})
                if e["task"] == "classification":
                    perf = f"{(m.get('accuracy') or 0)*100:.1f}%  acc"
                else:
                    r2 = m.get("r2_score") or 0
                    mae = m.get("mae") or 0
                    perf = f"R²={r2:.3f}  MAE={mae:.3f}"
                rows.append({
                    "Name":        e["name"],
                    "Model":       e["model_name"],
                    "Task":        e["task"],
                    "Target":      e["target_col"],
                    "Performance": perf,
                    "Saved":       e["created_at"][:16].replace("T", " ") + " UTC",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=400)
