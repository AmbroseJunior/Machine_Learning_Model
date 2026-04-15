"""
Universal data loader — reads any common dataset format into a DataFrame.
Supported: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, Feather, ODS
"""
import io
import pandas as pd

# Maps file extension → loader function
# Each loader receives the raw file bytes and returns a DataFrame

def _load_csv(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False, **kw)


def _load_tsv(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), sep="\t", low_memory=False, **kw)


def _load_excel(data: bytes, **kw) -> pd.DataFrame:
    xf = pd.ExcelFile(io.BytesIO(data))
    if len(xf.sheet_names) == 1:
        return pd.read_excel(io.BytesIO(data), **kw)
    # Multiple sheets → let caller handle sheet selection; return first by default
    # but also expose sheet names
    return pd.read_excel(io.BytesIO(data), sheet_name=xf.sheet_names[0], **kw), xf.sheet_names


def _load_json(data: bytes, **kw) -> pd.DataFrame:
    try:
        df = pd.read_json(io.BytesIO(data), **kw)
    except ValueError:
        # Try line-delimited JSON (JSONL / ndjson)
        df = pd.read_json(io.BytesIO(data), lines=True, **kw)
    # Flatten one level of nesting for any object columns
    object_cols = [c for c in df.columns if df[c].dtype == object
                   and df[c].dropna().apply(lambda x: isinstance(x, dict)).any()]
    if object_cols:
        parts = [df.drop(columns=object_cols)]
        for col in object_cols:
            try:
                parts.append(pd.json_normalize(df[col].dropna()).add_prefix(f"{col}."))
            except Exception:
                parts.append(df[[col]])
        df = pd.concat(parts, axis=1)
    return df


def _load_parquet(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data), **kw)


def _load_feather(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_feather(io.BytesIO(data), **kw)


def _load_ods(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), engine="odf", **kw)  # requires odfpy


# Extension → (loader_fn, human_label)
SUPPORTED_FORMATS = {
    "csv":     (_load_csv,     "CSV"),
    "tsv":     (_load_tsv,     "TSV (tab-separated)"),
    "txt":     (_load_csv,     "TXT (comma-separated)"),
    "xlsx":    (_load_excel,   "Excel (.xlsx)"),
    "xls":     (_load_excel,   "Excel (.xls)"),
    "json":    (_load_json,    "JSON"),
    "jsonl":   (_load_json,    "JSONL (newline-delimited JSON)"),
    "ndjson":  (_load_json,    "NDJSON"),
    "parquet": (_load_parquet, "Parquet"),
    "feather": (_load_feather, "Feather"),
    "ods":     (_load_ods,     "ODS (OpenDocument)"),
}

ACCEPTED_EXTENSIONS = list(SUPPORTED_FORMATS.keys())


def load_file(uploaded_file, sheet_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Load an uploaded Streamlit file object into a DataFrame.
    Returns (df, sheet_names).
    sheet_names is non-empty only for multi-sheet Excel/ODS files.
    """
    name = uploaded_file.name.lower()
    ext  = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Accepted: {', '.join(ACCEPTED_EXTENSIONS)}"
        )

    data   = uploaded_file.read()
    loader = SUPPORTED_FORMATS[ext][0]

    if ext in ("xlsx", "xls", "ods"):
        result = loader(data)
        if isinstance(result, tuple):
            # Multi-sheet: result = (df_first_sheet, [sheet_names])
            df, sheets = result
            if sheet_name and sheet_name != sheets[0]:
                df = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)
            return df, sheets
        return result, []
    else:
        return loader(data), []
