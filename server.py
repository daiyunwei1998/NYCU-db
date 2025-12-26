# server.py
# FastAPI mock server backed by Kaggle CSV (loaded once into memory).
# Implements the API endpoints and persists modifications back to CSV.
#
# Frontend alignment (IMPORTANT):
# - name      = tool name (stored in CSV column: company_name)
# - dev_name  = developer/team name (stored in CSV column: dev_name)  <-- added by us
#
# Endpoints:
# - GET    /api/v1/tools (supports page, q, task, tag, dev_name, sort=rating)
# - GET    /api/v1/tools/{tool_id}
# - POST   /api/v1/tools
# - PUT    /api/v1/tools/{tool_id}
# - DELETE /api/v1/tools/{tool_id}
# - GET    /api/v1/tags
# - GET    /ping
#
# Computed fields:
# - rating: deterministic 1..5 from tool_id (stored in DF column "rating")
# - has_free_ver: True if pricing contains "free" anywhere (case-insensitive)

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import re
import threading
import hashlib

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

# =============================================================================
# 0) CONFIG
# =============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tools.csv")
PAGE_SIZE = 20

KAGGLE_REQUIRED_COLUMNS = [
    "detail_url",
    "logo_url",
    "company_name",         # tool name
    "short_description",
    "primary_task",
    "applicable_tasks",
    "full_description",
    "pros",
    "cons",
    "pricing",
    "visit_website_url",
]

_Q_RE = re.compile(r"^Q(\d+)$")

# =============================================================================
# 1) IN MEMORY STORE
# =============================================================================

_DATA_LOCK = threading.RLock()
_DF: Optional[pd.DataFrame] = None


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _norm_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _parse_tags(applicable_tasks: str) -> List[str]:
    s = _norm_str(applicable_tasks)
    if s == "":
        return []
    parts = re.split(r"[|,;]", s)
    out: List[str] = []
    for p in parts:
        t = p.strip()
        if t != "":
            out.append(t)
    return out


def _is_free_from_pricing(pricing: str) -> bool:
    s = _norm_str(pricing).lower()
    if s == "":
        return False
    return "free" in s


def _has_free_ver_from_pricing(pricing: str) -> bool:
    return "free" in _norm_str(pricing).lower()


def _tags_to_applicable_tasks(tags: List[str]) -> str:
    cleaned: List[str] = []
    for t in tags:
        s = str(t).strip()
        if s != "":
            cleaned.append(s)
    return ", ".join(cleaned)


def _pricing_from_is_free(is_free: bool, pricing_existing: str) -> str:
    if pricing_existing.strip() != "":
        return pricing_existing
    if is_free:
        return "Free"
    return ""


def _pseudo_rating_from_tool_id(tool_id: int) -> int:
    h = hashlib.md5(str(int(tool_id)).encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return (v % 5) + 1


def _ensure_tool_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "tool_id" in df.columns:
        df2 = df.copy()
        df2["tool_id"] = df2["tool_id"].astype(int)
        return df2

    df2 = df.copy()
    df2.insert(0, "tool_id", range(1, len(df2) + 1))
    return df2


def _ensure_dev_name_column(df: pd.DataFrame) -> pd.DataFrame:
    # Add dev_name (developer/team) column if missing
    if "dev_name" in df.columns:
        df2 = df.copy()
        df2["dev_name"] = df2["dev_name"].fillna("").astype(str)
        return df2

    df2 = df.copy()
    # Put dev_name right after company_name for readability
    insert_at = list(df2.columns).index("company_name") + 1 if "company_name" in df2.columns else 0
    df2.insert(insert_at, "dev_name", "")
    return df2


def _extract_faqs_from_row(df: pd.DataFrame, row: pd.Series) -> List[Dict[str, str]]:
    q_nums: List[int] = []
    for col in df.columns:
        m = _Q_RE.match(col)
        if m:
            q_nums.append(int(m.group(1)))
    q_nums.sort()

    faqs: List[Dict[str, str]] = []
    for n in q_nums:
        q_col = f"Q{n}"
        a_col = f"A{n}"
        if q_col not in df.columns or a_col not in df.columns:
            continue

        q = _norm_str(row.get(q_col, ""))
        a = _norm_str(row.get(a_col, ""))
        if q == "" or a == "":
            continue
        faqs.append({"question": q, "answer": a})

    return faqs


def _kaggle_row_to_tool_response(df: pd.DataFrame, row: pd.Series) -> Dict[str, Any]:
    tool_id = int(row["tool_id"])

    pricing = _norm_str(row.get("pricing", ""))
    is_free = _is_free_from_pricing(pricing)
    has_free_ver = _has_free_ver_from_pricing(pricing)
    tags = _parse_tags(_norm_str(row.get("applicable_tasks", "")))

    rating_val = row.get("rating", "")
    if str(rating_val).strip() != "":
        try:
            rating = int(rating_val)
        except Exception:
            rating = _pseudo_rating_from_tool_id(tool_id)
    else:
        rating = _pseudo_rating_from_tool_id(tool_id)

    # Correct mapping
    tool_name = _norm_str(row.get("company_name", ""))
    dev_name = _norm_str(row.get("dev_name", ""))

    return {
        "tool_id": tool_id,

        # Frontend expected semantics
        "name": tool_name,
        "dev_name": dev_name,

        "visit_website_url": _norm_str(row.get("visit_website_url", "")),
        "short_description": _norm_str(row.get("short_description", "")),
        "full_description": _norm_str(row.get("full_description", "")),
        "primary_task_name": _norm_str(row.get("primary_task", "")),
        "is_free": is_free,
        "has_free_ver": has_free_ver,
        "tags": tags,

        "logo_url": _norm_str(row.get("logo_url", "")),
        "detail_url": _norm_str(row.get("detail_url", "")),
        "pros": _norm_str(row.get("pros", "")),
        "cons": _norm_str(row.get("cons", "")),
        "pricing": pricing,
        "faqs": _extract_faqs_from_row(df, row),

        "rating": rating,
    }


def _apply_filters(
    df: pd.DataFrame,
    task: Optional[str],
    tag: Optional[str],
    q: Optional[str],
    dev_name: Optional[str],
) -> pd.DataFrame:
    out = df

    if dev_name is not None and dev_name.strip() != "":
        dn = dev_name.strip().lower()
        out = out[out["dev_name"].astype(str).str.strip().str.lower().str.contains(dn, na=False)]

    if task is not None and task.strip() != "":
        task_norm = task.strip().lower()
        out = out[out["primary_task"].astype(str).str.strip().str.lower() == task_norm]

    if tag is not None and tag.strip() != "":
        tag_norm = tag.strip().lower()

        def has_tag(applicable_tasks: Any) -> bool:
            tags = _parse_tags(_norm_str(applicable_tasks))
            return any(t.lower() == tag_norm for t in tags)

        out = out[out["applicable_tasks"].apply(has_tag)]

    if q is not None and q.strip() != "":
        q_norm = q.strip().lower()
        name = out["company_name"].astype(str).str.lower()
        short = out["short_description"].astype(str).str.lower()
        out = out[name.str.contains(q_norm, na=False) | short.str.contains(q_norm, na=False)]

    return out


def _load_into_memory() -> None:
    global _DF

    if not os.path.exists(DATA_PATH):
        _fail(f"Missing CSV file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in KAGGLE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _fail(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    df = _ensure_tool_id_column(df)
    df = _ensure_dev_name_column(df)

    # Add deterministic pseudo ratings (no CSV column required)
    if "rating" not in df.columns:
        df["rating"] = df["tool_id"].apply(_pseudo_rating_from_tool_id).astype(int)
    else:
        df["rating"] = df["rating"].fillna(0).astype(int)
        df.loc[(df["rating"] < 1) | (df["rating"] > 5), "rating"] = df["tool_id"].apply(
            _pseudo_rating_from_tool_id
        ).astype(int)

    # Persist schema upgrades once (tool_id/dev_name/rating)
    current_cols = list(pd.read_csv(DATA_PATH).columns)
    need_persist = False
    if "tool_id" not in current_cols:
        need_persist = True
    if "dev_name" not in current_cols:
        need_persist = True
    if "rating" not in current_cols:
        need_persist = True
    if need_persist:
        df.to_csv(DATA_PATH, index=False)

    _DF = df


def _persist_from_memory() -> None:
    if _DF is None:
        _fail("Internal error: DF not loaded")
    _DF.to_csv(DATA_PATH, index=False)


def _get_df() -> pd.DataFrame:
    if _DF is None:
        _fail("Internal error: DF not loaded")
    return _DF


def _write_faqs_into_row_columns(df: pd.DataFrame, row_index: int, faqs: List[Dict[str, str]]) -> pd.DataFrame:
    df2 = df.copy()

    # Clear existing Q/A columns
    q_nums: List[int] = []
    for col in df2.columns:
        m = _Q_RE.match(col)
        if m:
            q_nums.append(int(m.group(1)))
    q_nums.sort()
    for n in q_nums:
        q_col = f"Q{n}"
        a_col = f"A{n}"
        if q_col in df2.columns:
            df2.at[row_index, q_col] = ""
        if a_col in df2.columns:
            df2.at[row_index, a_col] = ""

    # Write new ones
    for idx, qa in enumerate(faqs, start=1):
        q_col = f"Q{idx}"
        a_col = f"A{idx}"
        if q_col not in df2.columns:
            df2[q_col] = ""
        if a_col not in df2.columns:
            df2[a_col] = ""
        df2.at[row_index, q_col] = _norm_str(qa.get("question", ""))
        df2.at[row_index, a_col] = _norm_str(qa.get("answer", ""))

    return df2


# =============================================================================
# 2) REQUEST MODELS
# =============================================================================

class ToolCreate(BaseModel):
    # frontend semantics
    name: str = Field(min_length=1)          # tool name
    dev_name: Optional[str] = None           # developer/team name

    visit_website_url: HttpUrl
    logo_url: Optional[HttpUrl] = None
    detail_url: Optional[HttpUrl] = None

    short_description: str = Field(min_length=1)
    full_description: str = Field(min_length=1)
    primary_task_name: str = Field(min_length=1)

    is_free: bool
    tags: List[str] = Field(default_factory=list)

    pros: Optional[str] = None
    cons: Optional[str] = None
    pricing: Optional[str] = None

    faqs: Optional[List[Dict[str, str]]] = None


class ToolUpdate(BaseModel):
    # frontend semantics
    name: str = Field(min_length=1)          # tool name
    dev_name: Optional[str] = None           # developer/team name

    visit_website_url: HttpUrl
    logo_url: Optional[HttpUrl] = None
    detail_url: Optional[HttpUrl] = None

    short_description: str = Field(min_length=1)
    full_description: str = Field(min_length=1)
    primary_task_name: str = Field(min_length=1)

    is_free: bool
    tags: List[str] = Field(default_factory=list)

    pros: Optional[str] = None
    cons: Optional[str] = None
    pricing: Optional[str] = None

    faqs: Optional[List[Dict[str, str]]] = None


# =============================================================================
# 3) APP
# =============================================================================

app = FastAPI(title="Tool Management Mock API (In Memory CSV Cache)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://3.26.252.24:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   


@app.on_event("startup")
def on_startup() -> None:
    with _DATA_LOCK:
        _load_into_memory()


@app.get("/ping")
def ping() -> Dict[str, Any]:
    return {"ok": True}


# =============================================================================
# 4) ENDPOINTS
# =============================================================================

@app.get("/api/v1/tools")
def list_tools(
    task: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    dev_name: Optional[str] = Query(default=None),
    sort: Optional[str] = Query(default=None),  # supports "rating"
    page: int = Query(default=1, ge=1),
) -> Dict[str, Any]:
    with _DATA_LOCK:
        df = _get_df()
        filtered = _apply_filters(df, task=task, tag=tag, q=q, dev_name=dev_name)

        total = int(filtered.shape[0])
        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE

        sort_key = (sort or "").strip().lower()
        if sort_key == "rating":
            sorted_df = filtered.sort_values(["rating", "tool_id"], ascending=[False, True])
        else:
            sorted_df = filtered.sort_values("tool_id", ascending=True)

        page_df = sorted_df.iloc[start:end]
        items = [_kaggle_row_to_tool_response(df, page_df.iloc[i]) for i in range(page_df.shape[0])]
        return {"items": items, "page": page, "total": total}


@app.post("/api/v1/tools")
def create_tool(payload: ToolCreate) -> Dict[str, Any]:
    global _DF
    with _DATA_LOCK:
        df = _get_df()

        next_id = int(df["tool_id"].max()) + 1 if df.shape[0] > 0 else 1

        applicable_tasks = _tags_to_applicable_tasks(payload.tags)
        pricing = _pricing_from_is_free(payload.is_free, _norm_str(payload.pricing or ""))

        new_row = {
            "tool_id": next_id,
            "rating": _pseudo_rating_from_tool_id(next_id),

            "detail_url": str(payload.detail_url) if payload.detail_url is not None else "",
            "logo_url": str(payload.logo_url) if payload.logo_url is not None else "",

            # Correct semantics
            "company_name": payload.name,                      # tool name
            "dev_name": _norm_str(payload.dev_name or ""),     # developer/team

            "short_description": payload.short_description,
            "primary_task": payload.primary_task_name,
            "applicable_tasks": applicable_tasks,
            "full_description": payload.full_description,
            "pros": _norm_str(payload.pros or ""),
            "cons": _norm_str(payload.cons or ""),
            "pricing": pricing,
            "visit_website_url": str(payload.visit_website_url),
        }

        df2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if payload.faqs is not None:
            row_i = df2.index[df2["tool_id"] == next_id].tolist()[0]
            df2 = _write_faqs_into_row_columns(df2, row_i, payload.faqs)

        _DF = df2
        _persist_from_memory()

        created = _DF[_DF["tool_id"] == next_id].iloc[0]
        return _kaggle_row_to_tool_response(_DF, created)


@app.get("/api/v1/tools/{tool_id}")
def get_tool(tool_id: int) -> Dict[str, Any]:
    with _DATA_LOCK:
        df = _get_df()
        hit = df[df["tool_id"] == int(tool_id)]
        if hit.shape[0] == 0:
            raise HTTPException(status_code=404, detail="Tool not found")
        return _kaggle_row_to_tool_response(df, hit.iloc[0])


@app.put("/api/v1/tools/{tool_id}")
def update_tool(tool_id: int, payload: ToolUpdate) -> Dict[str, Any]:
    global _DF
    with _DATA_LOCK:
        df = _get_df()

        idxs = df.index[df["tool_id"] == int(tool_id)].tolist()
        if len(idxs) == 0:
            raise HTTPException(status_code=404, detail="Tool not found")
        i = idxs[0]

        applicable_tasks = _tags_to_applicable_tasks(payload.tags)
        pricing = _pricing_from_is_free(payload.is_free, _norm_str(payload.pricing or ""))

        df2 = df.copy()

        # Correct semantics
        df2.at[i, "company_name"] = payload.name
        df2.at[i, "dev_name"] = _norm_str(payload.dev_name or "")

        df2.at[i, "visit_website_url"] = str(payload.visit_website_url)
        df2.at[i, "logo_url"] = str(payload.logo_url) if payload.logo_url is not None else ""
        df2.at[i, "detail_url"] = str(payload.detail_url) if payload.detail_url is not None else ""

        df2.at[i, "short_description"] = payload.short_description
        df2.at[i, "full_description"] = payload.full_description
        df2.at[i, "primary_task"] = payload.primary_task_name

        df2.at[i, "applicable_tasks"] = applicable_tasks
        df2.at[i, "pricing"] = pricing

        df2.at[i, "pros"] = _norm_str(payload.pros or "")
        df2.at[i, "cons"] = _norm_str(payload.cons or "")

        if payload.faqs is not None:
            df2 = _write_faqs_into_row_columns(df2, i, payload.faqs)

        _DF = df2
        _persist_from_memory()

        row = _DF[_DF["tool_id"] == int(tool_id)].iloc[0]
        return _kaggle_row_to_tool_response(_DF, row)


@app.delete("/api/v1/tools/{tool_id}")
def delete_tool(tool_id: int) -> Dict[str, Any]:
    global _DF
    with _DATA_LOCK:
        df = _get_df()
        before = df.shape[0]
        df2 = df[df["tool_id"] != int(tool_id)].copy()
        after = df2.shape[0]

        if before == after:
            raise HTTPException(status_code=404, detail="Tool not found")

        _DF = df2
        _persist_from_memory()
        return {"ok": True}


@app.get("/api/v1/tags")
def list_tags() -> Dict[str, Any]:
    # Tags come from applicable_tasks
    with _DATA_LOCK:
        df = _get_df()

        counts: Dict[str, int] = {}
        for v in df["applicable_tasks"].tolist():
            for t in _parse_tags(_norm_str(v)):
                counts[t] = counts.get(t, 0) + 1

        sorted_keys = sorted(counts.keys(), key=lambda k: (-counts[k], k.lower()))
        items = [{"tag": k, "count": counts[k]} for k in sorted_keys]
        return {"items": items, "total": len(items)}
