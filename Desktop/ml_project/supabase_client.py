"""
Supabase persistence for ML Studio.

Tech stack: Python · Supabase · Streamlit  (no Docker, no Node.js)

On first use this module auto-creates a local app user via the Supabase
Admin API (service-role key) so experiments can be stored without a
manual sign-up flow.  The assigned user-id is cached in .mlstudio_config
so subsequent runs reuse the same identity.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_URL  = os.getenv("SUPABASE_URL", "")
_ANON = os.getenv("SUPABASE_ANON_KEY", "")
_SVC  = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Credentials for the auto-created local user
_LOCAL_EMAIL = "local@mlstudio.app"
_LOCAL_PASS  = "MLStudio!Local#2024"

# Persist the user-id across Streamlit reruns / sessions
_CONFIG_PATH = Path(__file__).parent / ".mlstudio_config"


# ── helpers ───────────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Return True when the minimum env vars are present."""
    return bool(_URL and _ANON and _SVC)


def _load_user_id() -> str | None:
    try:
        if _CONFIG_PATH.exists():
            return json.loads(_CONFIG_PATH.read_text()).get("user_id")
    except Exception:
        pass
    return None


def _save_user_id(uid: str) -> None:
    _CONFIG_PATH.write_text(json.dumps({"user_id": uid}))


def _get_or_create_user() -> str | None:
    """
    Return a valid Supabase user-id for local use.
    Creates the user via the Admin API on the very first call;
    subsequent calls return the cached id.
    """
    cached = _load_user_id()
    if cached:
        return cached

    from supabase import create_client

    admin = create_client(_URL, _SVC)
    anon  = create_client(_URL, _ANON)

    uid: str | None = None

    # Try to create a fresh user
    try:
        resp = admin.auth.admin.create_user({
            "email":         _LOCAL_EMAIL,
            "password":      _LOCAL_PASS,
            "email_confirm": True,
        })
        uid = resp.user.id
    except Exception:
        # User already exists — sign in to retrieve the id
        try:
            resp = anon.auth.sign_in_with_password({
                "email":    _LOCAL_EMAIL,
                "password": _LOCAL_PASS,
            })
            uid = resp.user.id
        except Exception:
            pass

    if uid:
        _save_user_id(uid)
    return uid


def get_client():
    """
    Return ``(supabase_client, user_id)`` ready for DB operations,
    or ``(None, None)`` when Supabase is not configured.
    """
    if not is_configured():
        return None, None
    uid = _get_or_create_user()
    if not uid:
        return None, None
    from supabase import create_client
    return create_client(_URL, _SVC), uid


# ── public API ────────────────────────────────────────────────────────────────

def save_experiment(result: dict, name: str) -> str | None:
    """
    Persist a trained-model result to Supabase.
    Returns the new experiment UUID, or None on failure.
    """
    client, uid = get_client()
    if not client:
        return None

    task = result["task"]
    m    = result["metrics"]
    now  = datetime.now(timezone.utc).isoformat()

    exp = client.table("experiments").insert({
        "user_id":         uid,
        "name":            name,
        "model_name":      result["model_name"],
        "task":            task,
        "target_col":      result["target_col"],
        "feature_cols":    result["feature_cols"],
        "model_params":    result.get("model_params", {}),
        "pipeline_config": {},
        "status":          "completed",
        "started_at":      now,
        "completed_at":    now,
    }).execute()

    exp_id = exp.data[0]["id"]

    metrics: dict = {"experiment_id": exp_id}

    if task == "classification":
        report = m.get("report", {})
        macro  = report.get("macro avg", {})
        metrics.update({
            "accuracy":              float(m.get("accuracy", 0)),
            "precision_macro":       float(macro.get("precision", 0)),
            "recall_macro":          float(macro.get("recall", 0)),
            "f1_macro":              float(macro.get("f1-score", 0)),
            "classification_report": report,
            "confusion_matrix":      m.get("confusion_matrix"),
        })
    else:
        metrics.update({
            "r2_score": float(m.get("r2", 0)),
            "mae":      float(m.get("mae", 0)),
        })

    client.table("experiment_metrics").insert(metrics).execute()
    return exp_id


def load_experiments() -> list[dict]:
    """
    Return up to 50 most-recent experiments for the local user,
    with their metrics joined in.
    """
    client, uid = get_client()
    if not client:
        return []

    resp = (
        client.table("experiments")
        .select(
            "id, name, model_name, task, target_col, feature_cols, "
            "model_params, status, created_at, "
            "experiment_metrics(accuracy, r2_score, mae, f1_macro)"
        )
        .eq("user_id", uid)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return resp.data or []
