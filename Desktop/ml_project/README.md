# ML Studio

A browser-based machine-learning workbench built entirely in Python.
Upload a dataset, train a model, inspect its performance, make predictions,
and persist your experiments to Supabase — all without leaving a single tab.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI / Frontend | [Streamlit](https://streamlit.io) |
| ML | scikit-learn, pandas, NumPy |
| Persistence | [Supabase](https://supabase.com) (PostgreSQL + Auth) |
| Language | **Python only** — no Node.js, no Docker |

---

## Project Structure

```
ml_project/
├── app.py               # Streamlit application — all UI and page logic
├── ml_pipeline.py       # Training, prediction, and feature-importance helpers
├── data_loader.py       # Universal file loader (CSV → Parquet → ODS …)
├── sample_data.py       # Synthetic survey dataset generator
├── theme.py             # Shared colour palette and UI component helpers
├── supabase_client.py   # Supabase persistence (save / load experiments)
│
├── supabase/
│   ├── migrations/
│   │   └── 001_initial_schema.sql   # Full DB schema with RLS policies
│   └── storage_policies.sql         # Supabase Storage bucket policies
│
├── requirements.txt     # All Python dependencies
├── .env.example         # Environment variable template
└── .env                 # Your credentials (git-ignored)
```

---

## Quick Start

### 1 — Prerequisites

- Python 3.11 or newer
- A [Supabase](https://supabase.com) project (free tier is fine)

### 2 — Clone and install

```bash
git clone <repo-url>
cd ml_project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 — Configure Supabase

```bash
cp .env.example .env
```

Open `.env` and fill in the three values from your Supabase dashboard
(**Settings → API**):

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### 4 — Apply the database schema

In the Supabase dashboard open the **SQL Editor** and run:

```
supabase/migrations/001_initial_schema.sql
```

This creates the `experiments`, `experiment_metrics`, `datasets`, `models`,
`predictions`, and `audit_logs` tables with Row Level Security enabled.

### 5 — Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Features

### Sidebar — step by step

| Step | What happens |
|---|---|
| **01 · Load Data** | Generate a synthetic survey dataset or upload your own file |
| **02 · Configure** | Pick the target column, feature columns, model, and task type |
| **03 · Train** | One click trains the pipeline and shows a summary metric |
| **04 · Save** | Name the experiment and save metrics to Supabase |

The export button (💾) downloads the trained pipeline as a `.joblib` file
that can be loaded back with `joblib.load()` for offline inference.

---

### Tab — Data

- **Dataset preview** — first 100 rows of the loaded DataFrame
- **Column distribution** — bar chart (categorical) or histogram (numeric)
- **Correlation matrix** — Pearson heatmap for all numeric columns
- Summary metrics: row count, column count, missing cells, duplicate rows

### Tab — Quality

Automated column-level data quality report:

| Role | Meaning |
|---|---|
| `numeric` | Continuous number — median-imputed, standard-scaled |
| `bool` | Boolean — cast to Int64 |
| `datetime` | Date/time — converted to Unix timestamp |
| `low-card-cat` | ≤ N unique values — one-hot encoded |
| `high-card-cat` | N+1 … M unique values — ordinal encoded |
| `very-high-card` | > M unique values — ordinal encoded + scaled |
| `constant` | Only 1 unique value — automatically dropped |

Thresholds for N and M are set with the **Advanced settings** sliders.
The tab also surfaces per-column warnings (high missingness, datetime
conversion, very-high cardinality) and a missing-value bar chart.

### Tab — Metrics

Automatically adapts to the detected task type:

**Classification**
- Accuracy, per-class Precision / Recall / F1, Support
- Confusion matrix heatmap
- Feature importances bar chart

**Regression**
- R² score, Mean Absolute Error
- Actual vs Predicted scatter plot with a perfect-fit reference line
- Feature importances bar chart

### Tab — Predict

Interactive prediction form — widget types adapt automatically to the
column data type detected at training time:

| Data type | Widget |
|---|---|
| Boolean | Selectbox (True / False) |
| Low-cardinality categorical | Selectbox with observed values |
| High-cardinality categorical | Free-text input |
| Integer | Number input (min / max / median from training data) |
| Float | Number input (min / max / median from training data) |

For classification models with `predict_proba` support, a confidence
bar chart is shown alongside the predicted class.

### Tab — History

Loads all saved experiments from Supabase and displays them in a table:

| Column | Description |
|---|---|
| Name | User-supplied experiment label |
| Model | Algorithm used |
| Task | `classification` or `regression` |
| Target | Column that was predicted |
| Performance | Accuracy % (classification) or R² + MAE (regression) |
| Saved | UTC timestamp |

---

## Supported File Formats

| Extension | Format |
|---|---|
| `.csv`, `.txt` | Comma-separated values |
| `.tsv` | Tab-separated values |
| `.xlsx`, `.xls` | Microsoft Excel |
| `.json`, `.jsonl`, `.ndjson` | JSON / newline-delimited JSON |
| `.parquet` | Apache Parquet |
| `.feather` | Apache Arrow Feather |
| `.ods` | OpenDocument Spreadsheet |

Multi-sheet Excel / ODS files prompt for sheet selection before loading.

---

## Available Models

| Name | Classification | Regression | Key hyperparameters |
|---|---|---|---|
| Gradient Boosting | `GradientBoostingClassifier` | `GradientBoostingRegressor` | Trees, Learning rate, Max depth |
| Random Forest | `RandomForestClassifier` | `RandomForestRegressor` | Trees, Max depth, Min samples split |
| Decision Tree | `DecisionTreeClassifier` | `DecisionTreeRegressor` | Max depth, Min samples split/leaf |
| Logistic / Linear Regression | `LogisticRegression` | `Ridge` | Max iterations, Alpha / C |
| K-Nearest Neighbors | `KNeighborsClassifier` | `KNeighborsRegressor` | k (neighbours), Leaf size |

Task type is **auto-detected**: a numeric target with ≤ N unique values is
treated as classification; above that threshold it becomes regression.
The threshold and the override toggle are both exposed in the sidebar.

---

## Preprocessing Pipeline

Every training run builds a `scikit-learn` `Pipeline` automatically:

```
Raw DataFrame
    │
    ├── Numeric columns   → SimpleImputer(median) → StandardScaler
    ├── Low-card cats     → SimpleImputer(most_frequent) → OneHotEncoder
    └── High-card cats    → SimpleImputer(most_frequent) → OrdinalEncoder → StandardScaler
    │
    └── ColumnTransformer → Model
```

- **Missing values** are imputed — no rows are dropped.
- **Constant columns** (only 1 unique value) are dropped before training.
- **Datetime columns** are converted to integer Unix timestamps.
- **Boolean columns** are cast to nullable `Int64`.
- **Label encoding** is applied to the target for classification so class
  names are preserved in predictions and the confusion matrix.
- Train/test split defaults to 80 / 20 with stratification for classification
  (falls back to random split when any class has fewer than 2 samples).

---

## Supabase Integration

`supabase_client.py` manages the full Supabase lifecycle automatically:

1. **First run** — calls the Admin API (`service_role` key) to create a
   dedicated local app user. The assigned UUID is written to
   `.mlstudio_config` (git-ignored) so it persists across sessions.
2. **Subsequent runs** — reads the UUID from `.mlstudio_config` and skips
   user creation.
3. **Save experiment** — inserts a row into `experiments` and a linked row
   into `experiment_metrics`.
4. **Load history** — queries `experiments` joined with `experiment_metrics`
   for the local user, ordered by creation time descending.

Row Level Security is active on all tables. The service-role key bypasses
RLS for the automated user-setup step only; all other operations run as the
local user identity.

### Supabase schema summary

```
profiles            — user identity (auto-created on first sign-in)
datasets            — dataset metadata + column audit
experiments         — one row per training run
experiment_metrics  — accuracy / R² / MAE / confusion matrix / feature importances
models              — stored model artifacts (registry)
predictions         — logged inference results
audit_logs          — append-only action log
```

Full DDL: [`supabase/migrations/001_initial_schema.sql`](supabase/migrations/001_initial_schema.sql)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Project URL from Supabase dashboard |
| `SUPABASE_ANON_KEY` | Yes | Public anon key (used for auth sign-in) |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Private service key (used for admin user setup and DB writes) |

Copy [`.env.example`](.env.example) to `.env` and fill in these three values.
The `.env` file is read automatically by `python-dotenv` at startup.

---

## Exporting a Trained Model

After training, click **💾 Export model (.joblib)** in the sidebar to
download a self-contained bundle. Load it anywhere:

```python
import joblib
bundle = joblib.load("model.joblib")

# bundle keys:
#   pipeline        — fitted sklearn Pipeline (preprocessor + model)
#   label_encoder   — LabelEncoder for classification targets (None for regression)
#   feature_cols    — list of feature column names in training order
#   target_col      — name of the target column
#   task            — "classification" or "regression"
#   model_name      — display name of the algorithm
#   model_params    — hyperparameter dict used during training

import pandas as pd
df_new = pd.DataFrame([{"age": 25, "focus_score": 80, ...}])
predictions = bundle["pipeline"].predict(df_new[bundle["feature_cols"]])
```

---

## Development Notes

- All thresholds (OHE cardinality, classification boundary, test split %)
  are runtime parameters — nothing is hardcoded in `ml_pipeline.py`.
- The Streamlit session state dictionary (`st.session_state`) is the sole
  in-memory store; there is no external state server.
- Plotly figures use a shared dark theme defined in `theme.py` (`C` dict).
- The `sample_data.py` generator produces a synthetic survey dataset
  (performance category classification) and can be run standalone to
  regenerate `sample_data.csv`.

---

## Requirements

See [`requirements.txt`](requirements.txt) for pinned versions. Core packages:

```
streamlit        # UI framework
scikit-learn     # ML algorithms and preprocessing
pandas           # DataFrames
numpy            # Numerical operations
plotly           # Interactive charts
joblib           # Model serialisation
supabase         # Supabase Python client
python-dotenv    # .env file loading
openpyxl / xlrd / pyarrow / odfpy   # File format support
```
