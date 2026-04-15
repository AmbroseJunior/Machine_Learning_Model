"""
ML pipeline — fully dynamic, zero hardcoded assumptions.
All thresholds, hyperparameters, and task detection are passed in at call time.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib


# ── Model catalogue ───────────────────────────────────────────────────────────
# Each entry: display_name → {clf_class, reg_class, param_specs}
# param_specs: {param_name: {type, min, max, default, step(opt), label}}

MODEL_CATALOGUE = {
    "Gradient Boosting": {
        "clf": GradientBoostingClassifier,
        "reg": GradientBoostingRegressor,
        "params": {
            "n_estimators":  {"type": "int",   "min": 10,   "max": 1000, "default": 200, "step": 10,   "label": "Trees"},
            "learning_rate": {"type": "float", "min": 0.001,"max": 1.0,  "default": 0.1,              "label": "Learning rate"},
            "max_depth":     {"type": "int",   "min": 1,    "max": 20,   "default": 4,   "step": 1,   "label": "Max depth"},
        },
        "shared": {"random_state": 42},
    },
    "Random Forest": {
        "clf": RandomForestClassifier,
        "reg": RandomForestRegressor,
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 200, "step": 10,  "label": "Trees"},
            "max_depth":    {"type": "int", "min": 1,  "max": 50,   "default": 10,  "step": 1,   "label": "Max depth"},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "step": 1,  "label": "Min samples split"},
        },
        "shared": {"random_state": 42, "n_jobs": -1},
    },
    "Decision Tree": {
        "clf": DecisionTreeClassifier,
        "reg": DecisionTreeRegressor,
        "params": {
            "max_depth":         {"type": "int", "min": 1, "max": 50,  "default": 8, "step": 1, "label": "Max depth"},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "step": 1, "label": "Min samples split"},
            "min_samples_leaf":  {"type": "int", "min": 1, "max": 20, "default": 1, "step": 1, "label": "Min samples leaf"},
        },
        "shared": {"random_state": 42},
    },
    "Logistic / Linear Regression": {
        "clf": LogisticRegression,
        "reg": Ridge,
        "params": {
            "max_iter":  {"type": "int",   "min": 100, "max": 10000, "default": 1000, "step": 100, "label": "Max iterations (clf)"},
            "reg_alpha": {"type": "float", "min": 0.0, "max": 100.0, "default": 1.0,              "label": "Alpha / regularisation"},
        },
        "shared": {},
    },
    "K-Nearest Neighbors": {
        "clf": KNeighborsClassifier,
        "reg": KNeighborsRegressor,
        "params": {
            "n_neighbors": {"type": "int", "min": 1, "max": 100, "default": 5, "step": 1, "label": "Neighbours (k)"},
            "leaf_size":   {"type": "int", "min": 5, "max": 100, "default": 30, "step": 5, "label": "Leaf size"},
        },
        "shared": {"n_jobs": -1},
    },
}


def build_model(model_name: str, task: str, user_params: dict):
    """Instantiate the chosen model with user-supplied hyperparameters."""
    spec   = MODEL_CATALOGUE[model_name]
    shared = spec["shared"].copy()
    klass  = spec["clf"] if task == "classification" else spec["reg"]

    # Map user_params keys to constructor kwargs
    kwargs = {**shared}
    for pname, val in user_params.items():
        if pname == "reg_alpha":
            if task == "classification":
                kwargs["C"] = 1.0 / max(float(val), 1e-6)   # LogisticRegression uses C
            else:
                kwargs["alpha"] = float(val)
        else:
            kwargs[pname] = val

    # Remove kwargs the class doesn't accept
    import inspect
    valid = inspect.signature(klass.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return klass(**kwargs)


# ── Column normalisation ──────────────────────────────────────────────────────

def normalise_dataframe(
    df: pd.DataFrame,
    cols: list[str],
    col_roles: dict | None = None,
) -> pd.DataFrame:
    """
    Convert column types that sklearn cannot handle natively.
    col_roles: {col: role} saved at train time so predict uses identical logic.
      - datetime / role=='datetime'  → unix timestamp (int seconds)
      - bool     / role=='bool'      → Int64 (nullable)
    """
    out = df[cols].copy()
    for col in [c for c in cols if c in out.columns]:
        s    = out[col]
        role = (col_roles or {}).get(col, "")
        if pd.api.types.is_datetime64_any_dtype(s) or role == "datetime":
            out[col] = pd.to_datetime(s, errors="coerce").astype("int64") // 10 ** 9
        elif pd.api.types.is_bool_dtype(s) or role == "bool":
            out[col] = s.astype("Int64")
    return out


# ── Column audit ──────────────────────────────────────────────────────────────

def audit_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    ohe_max_cardinality: int = 20,
    high_card_threshold: int = 50,
) -> dict:
    """
    Return per-column metadata. Thresholds are passed in — nothing hardcoded.
    Roles: numeric | bool | datetime | low-card-cat | high-card-cat | very-high-card | constant
    """
    info = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        s        = df[col]
        null_pct = round(s.isnull().mean() * 100, 1)
        n_unique = s.nunique()

        if n_unique <= 1:
            role = "constant"
        elif pd.api.types.is_bool_dtype(s):
            role = "bool"
        elif pd.api.types.is_datetime64_any_dtype(s):
            role = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            role = "numeric"
        elif n_unique <= ohe_max_cardinality:
            role = "low-card-cat"
        elif n_unique <= high_card_threshold:
            role = "high-card-cat"
        else:
            role = "very-high-card"

        info[col] = {
            "dtype":    str(s.dtype),
            "null_pct": null_pct,
            "n_unique": n_unique,
            "role":     role,
        }
    return info


# ── Task detection ────────────────────────────────────────────────────────────

def detect_task(y: pd.Series, task_override: str | None, classification_threshold: int) -> str:
    """
    Determine 'classification' or 'regression'.
    task_override: 'classification' | 'regression' | None (auto)
    classification_threshold: if target is numeric with <= this many unique values → classification
    """
    if task_override in ("classification", "regression"):
        return task_override
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if not pd.api.types.is_numeric_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if y.nunique() <= classification_threshold:
        return "classification"
    return "regression"


# ── Pipeline builder ──────────────────────────────────────────────────────────

def _classify_columns(X: pd.DataFrame, ohe_max: int, high_card: int):
    num_cols, low_cat, high_cat = [], [], []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(col)
        elif s.nunique() <= ohe_max:
            low_cat.append(col)
        else:
            high_cat.append(col)
    return num_cols, low_cat, high_cat


def build_pipeline(
    X: pd.DataFrame,
    task: str,
    model_name: str,
    user_params: dict,
    ohe_max_cardinality: int,
    high_card_threshold: int,
) -> Pipeline:
    num_cols, low_cat, high_cat = _classify_columns(X, ohe_max_cardinality, high_card_threshold)

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ]), num_cols))
    if low_cat:
        transformers.append(("cat_low", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore")),
        ]), low_cat))
    if high_cat:
        transformers.append(("cat_high", Pipeline([
            ("impute",  SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scale",   StandardScaler()),
        ]), high_cat))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model        = build_model(model_name, task, user_params)
    return Pipeline([("prep", preprocessor), ("model", model)])


# ── Main train ────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float,
    model_name: str,
    model_params: dict,
    task_override: str | None,
    classification_threshold: int,
    ohe_max_cardinality: int,
    high_card_threshold: int,
    col_overrides: dict | None = None,
) -> dict:
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN — sklearn cannot train on NaN targets
    valid_mask = y.notna()
    if not valid_mask.all():
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    if len(y) == 0:
        raise ValueError("Target column has no valid (non-NaN) rows after filtering.")

    # User-forced column type overrides
    if col_overrides:
        for col, override in col_overrides.items():
            if col not in X.columns:
                continue
            if override == "categorical":
                X[col] = X[col].astype(str)
            elif override == "numeric":
                X[col] = pd.to_numeric(X[col], errors="coerce")

    # Save raw roles before any normalisation
    col_roles_raw = {
        col: audit_columns(X, [col], ohe_max_cardinality, high_card_threshold)[col]["role"]
        for col in X.columns
    }

    # Normalise datetimes + bools
    X = normalise_dataframe(X, X.columns.tolist(), col_roles_raw)

    # Drop constant columns
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    feature_cols = X.columns.tolist()
    col_audit    = audit_columns(X, feature_cols, ohe_max_cardinality, high_card_threshold)
    task         = detect_task(y, task_override, classification_threshold)

    le = None
    if task == "classification":
        le    = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
    else:
        y_enc = pd.to_numeric(y, errors="coerce").values
        nan_mask = ~np.isnan(y_enc)
        if not nan_mask.all():
            X     = X[nan_mask]
            y_enc = y_enc[nan_mask]

    class_counts = pd.Series(y_enc).value_counts()
    can_stratify = task == "classification" and (class_counts >= 2).all()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42,
        stratify=(y_enc if can_stratify else None),
    )

    pipeline = build_pipeline(X_train, task, model_name, model_params,
                               ohe_max_cardinality, high_card_threshold)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task == "classification":
        classes       = le.classes_.tolist()
        present       = sorted(set(y_test))
        present_names = [le.inverse_transform([i])[0] for i in present]
        metrics = {
            "task":             "classification",
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "report":           classification_report(
                                    y_test, y_pred, labels=present,
                                    target_names=present_names,
                                    output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=present).tolist(),
            "classes":          present_names,
            "all_classes":      classes,
        }
    else:
        metrics = {
            "task": "regression",
            "mae":  round(mean_absolute_error(y_test, y_pred), 4),
            "r2":   round(r2_score(y_test, y_pred), 4),
        }

    return {
        "pipeline":      pipeline,
        "label_encoder": le,
        "metrics":       metrics,
        "feature_cols":  feature_cols,
        "target_col":    target_col,
        "task":          task,
        "model_name":    model_name,
        "model_params":  model_params,
        "col_audit":     col_audit,
        "col_roles":     col_roles_raw,
        "constant_cols": constant_cols,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_pred":        y_pred,
    }


# ── Prediction helpers ────────────────────────────────────────────────────────

def _prepare_row(result: dict, input_data: dict) -> pd.DataFrame:
    feature_cols = result["feature_cols"]
    col_roles    = result.get("col_roles", {})
    row = pd.DataFrame([{col: input_data.get(col) for col in feature_cols}])
    return normalise_dataframe(row, row.columns.tolist(), col_roles)


def predict_single(result: dict, input_data: dict):
    row  = _prepare_row(result, input_data)
    pred = result["pipeline"].predict(row)[0]
    le   = result["label_encoder"]
    if result["task"] == "classification" and le is not None:
        return le.inverse_transform([int(pred)])[0]
    return round(float(pred), 3)


def predict_proba(result: dict, input_data: dict) -> dict | None:
    pipeline = result["pipeline"]
    le       = result["label_encoder"]
    if not hasattr(pipeline.named_steps["model"], "predict_proba"):
        return None
    row   = _prepare_row(result, input_data)
    proba = pipeline.predict_proba(row)[0]
    return {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}


def get_feature_importances(result: dict) -> pd.DataFrame:
    pipeline     = result["pipeline"]
    feature_cols = result["feature_cols"]
    prep         = pipeline.named_steps["prep"]
    model        = pipeline.named_steps["model"]

    try:
        feature_names = list(prep.get_feature_names_out())
    except Exception:
        feature_names = feature_cols

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()[:len(feature_names)]
    else:
        return pd.DataFrame({"feature": feature_cols, "importance": [0.0] * len(feature_cols)})

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(result: dict, path: str = "model.joblib"):
    joblib.dump({k: result[k] for k in
                 ("pipeline", "label_encoder", "feature_cols",
                  "target_col", "task", "model_name", "model_params")}, path)


def load_model(path: str = "model.joblib") -> dict:
    return joblib.load(path)
