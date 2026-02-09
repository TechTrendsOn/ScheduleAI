# pyright: ignore
# ingestion/common_utils.py
# Unified, production‑ready utilities for all ingestors

import os
import re
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# REGEX HELPERS
# ---------------------------------------------------------

HTML_TOKEN_RE = re.compile(r"<[^>]+>")
COLAB_TOKEN_RE = re.compile(r"\.colab-[\w\-]+|colab-df-[\w\-]+")


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(s: str) -> str:
    """Normalize unicode spaces and strip whitespace."""
    if not isinstance(s, str):
        return s
    return (
        s.replace("\xa0", " ")
         .replace("\u200b", "")
         .replace("\u200c", "")
         .replace("\u200d", "")
         .replace("\u202f", " ")
         .replace("\u205f", " ")
         .strip()
    )


def strip_html_tokens(x) -> str:
    """Remove HTML/Colab tokens and normalize whitespace."""
    if x is None:
        return ""
    s = str(x)
    s = HTML_TOKEN_RE.sub(" ", s)
    s = COLAB_TOKEN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return normalize_text(s)


# ---------------------------------------------------------
# VALUE NORMALIZATION
# ---------------------------------------------------------

def clean_value(v):
    """Normalize a single cell value."""
    if v is None:
        return None
    s = normalize_text(str(v))

    if s.lower() in ("", "nan", "none", "-", "null"):
        return None

    return s


def is_empty_row(row) -> bool:
    """Return True if all values in the row are empty/None."""
    vals = [clean_value(v) for v in row.tolist()]
    return all(v is None for v in vals)


# ---------------------------------------------------------
# HEADER NORMALIZATION
# ---------------------------------------------------------

def sanitize_headers(headers: List[Any]) -> List[str]:
    """Clean header labels, deduplicate, and normalize."""
    cleaned = []
    for h in headers:
        s = strip_html_tokens(h).replace("\n", " ").strip()
        s = normalize_text(s)
        s = re.sub(r"\s+", " ", s)
        cleaned.append(s if s else "unnamed")

    # Deduplicate
    seen = {}
    out = []
    for c in cleaned:
        idx = seen.get(c, 0)
        name = c if idx == 0 else f"{c}_{idx}"
        seen[c] = idx + 1
        out.append(name)
    return out


# ---------------------------------------------------------
# TABLE CLEANING
# ---------------------------------------------------------

def clean_table(df, min_cols: int = 1) -> pd.DataFrame:
    """Drop empty rows/columns and enforce minimum non‑empty columns."""
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return df.reset_index(drop=True)
    df = pd.DataFrame(df[df.count(axis=1) >= min_cols]).reset_index(drop=True)
    return df


def drop_junk_rows(df) -> pd.DataFrame:
    """Remove header echoes, whitespace-only rows, and fully empty rows."""
    if df is None or df.empty:
        return df

    df = df.copy()

    # Normalize all cells
    df = df.map(clean_value)

    # Drop rows where all fields are empty/None
    df = pd.DataFrame(df[~df.apply(is_empty_row, axis=1)])

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# NOTES NORMALIZATION
# ---------------------------------------------------------

def normalize_notes(notes: List[str]) -> List[str]:
    """Clean extracted notes."""
    cleaned = []
    for n in notes:
        n = normalize_text(n)
        if n and n.lower() != "nan":
            cleaned.append(n)
    return cleaned


# ---------------------------------------------------------
# HEADER DETECTION
# ---------------------------------------------------------

def detect_header_row(raw: pd.DataFrame,
                      expected_tokens=None,
                      min_matches: int = 2) -> Optional[int]:
    """Heuristic header detection based on token matches."""
    if expected_tokens is None:
        expected_tokens = [
            "id", "employee", "name", "type", "station", "shift",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun",
            "date", "code", "time", "parameter", "value"
        ]

    raw = raw.fillna("").astype(str)
    best_idx, best_score = None, -1.0

    for i in range(min(len(raw), 20)):
        row = [strip_html_tokens(x).strip().lower()
               for x in raw.iloc[i].tolist()]
        score = sum(1 for t in expected_tokens
                    if any(t in c for c in row if c))
        score += 0.08 * sum(1 for c in row if c)

        if score > best_score:
            best_score, best_idx = score, i

    if best_idx is None or best_score < min_matches:
        return None
    return int(best_idx)


def collapse_multirow_header(raw: pd.DataFrame,
                             max_rows: int = 4) -> Tuple[Optional[int], Optional[List[str]]]:
    """Detect two‑row headers and collapse them."""
    n = min(max_rows, len(raw))
    nonempty_counts = [raw.iloc[r].notna().sum() for r in range(n)]

    for r in range(n - 1):
        if nonempty_counts[r] >= 2 and nonempty_counts[r + 1] >= 2:
            hdr1 = raw.iloc[r].fillna("").astype(str).tolist()
            hdr2 = raw.iloc[r + 1].fillna("").astype(str).tolist()
            combined = [
                f"{strip_html_tokens(a)} {strip_html_tokens(b)}".strip()
                for a, b in zip(hdr1, hdr2)
            ]
            return r + 1, combined

    return None, None


# ---------------------------------------------------------
# NOTES EXTRACTION
# ---------------------------------------------------------

def extract_notes_below_table(raw: pd.DataFrame) -> List[str]:
    """Extract notes below the last non‑empty table row."""
    raw = raw.fillna("").astype(str)
    nonempty = pd.Series(
        raw.apply(lambda r: sum(1 for v in r if str(v).strip() != ""), axis=1)
    )

    if nonempty.empty:
        return []

    idxs = [i for i, v in enumerate(nonempty.tolist()) if v >= 2]
    last_table_idx = idxs[-1] if idxs else -1

    start_idx = last_table_idx + 1
    notes_rows = raw.iloc[start_idx:] if start_idx < len(raw) else pd.DataFrame()

    notes = []
    for _, r in notes_rows.iterrows():
        line = " ".join([str(v).strip() for v in r if str(v).strip() != ""]).strip()
        if line:
            notes.append(strip_html_tokens(line))

    return notes


# ---------------------------------------------------------
# SCHEMA VALIDATION
# ---------------------------------------------------------

def validate_schema(df: pd.DataFrame, required_cols: List[str], sheet: str) -> List[str]:
    """Check required columns exist."""
    warnings = []
    if df is None or df.empty:
        warnings.append(f"{sheet}: table is empty")
        return warnings

    cols = [c.lower() for c in df.columns]
    for rc in required_cols:
        if rc.lower() not in cols:
            warnings.append(f"{sheet}: missing required column '{rc}'")

    return warnings


# ---------------------------------------------------------
# SAFE NUMERIC PARSING
# ---------------------------------------------------------

def safe_float(x, default=None):
    """Convert to float safely, returning default on failure."""
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s.lower() in ("", "nan", "none", "-", "null"):
            return default
        return float(s)
    except Exception:
        return default


# ---------------------------------------------------------
# SAFE TIME PARSING
# ---------------------------------------------------------

def safe_time(t: str):
    """Parse HH:MM into (hour, minute) or return None."""
    if not isinstance(t, str):
        return None
    t = t.strip()
    if not re.match(r"^\d{1,2}:\d{2}$", t):
        return None
    try:
        h, m = map(int, t.split(":"))
        return h, m
    except Exception:
        return None


def time_to_minutes(t: str):
    """Convert HH:MM to total minutes or return None."""
    parsed = safe_time(t)
    if not parsed:
        return None
    h, m = parsed
    return h * 60 + m


def validate_time_range(start: str, end: str) -> bool:
    """Return True if end > start and both are valid times."""
    s = time_to_minutes(start)
    e = time_to_minutes(end)
    if s is None or e is None:
        return False
    return e > s


# ---------------------------------------------------------
# DATE COLUMN DETECTION
# ---------------------------------------------------------

def is_date_column(col: str) -> bool:
    """Detect if a column label represents a date or weekday."""
    c = col.lower()
    weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    return bool(any(w in c for w in weekdays) or re.search(r"\b\d{1,2}\b", c))


# ---------------------------------------------------------
# STORE ID NORMALIZATION
# ---------------------------------------------------------

def normalize_store_id(s: str) -> str:
    """Normalize store identifiers (Store 1 → Store_1)."""
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------
# SHIFT TOKEN VALIDATION
# ---------------------------------------------------------

def is_shift_token(tok: str) -> bool:
    """Return True if token looks like a valid shift code."""
    if not tok:
        return False
    t = tok.strip().upper()
    # Known patterns
    if t in {"1F", "2F", "3F", "SC", "S", "M", "/", "NA"}:
        return True
    # Generic alphanumeric
    return t.replace("/", "").isalnum()


# ---------------------------------------------------------
# SHEET NAME NORMALIZATION
# ---------------------------------------------------------

def normalize_sheet_name(name: str) -> str:
    """Normalize sheet names for consistent matching."""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------
# ROSTER PERIOD EXTRACTION
# ---------------------------------------------------------

def extract_roster_period(raw_df: pd.DataFrame):
    """
    Extract metadata like:
    - 'Roster Period: December 2024'
    - 'Date Range: Nov 25 - Dec 31'
    """
    text = " ".join(
        " ".join(str(v) for v in row if str(v).strip())
        for _, row in raw_df.iterrows()
    ).lower()

    period = None
    date_range = None

    m1 = re.search(r"roster period[:\s]+([a-z]+\s+\d{4})", text)
    if m1:
        period = m1.group(1).title()

    m2 = re.search(r"date range[:\s]+([a-z0-9\s\-]+)", text)
    if m2:
        date_range = m2.group(1).title()

    return {"roster_period": period, "date_range": date_range}


def parse_date_label(label: str, roster_period: Optional[str] = None) -> Optional[str]:
    """Parse date labels like 'Tue Dec 10' into ISO date string."""
    if not label:
        return None
    s = normalize_text(str(label))
    if not s:
        return None

    year: Optional[int] = None
    if roster_period:
        try:
            year = int(str(roster_period).strip().split()[-1])
        except Exception:
            year = None

    for fmt in ("%a %b %d", "%a %B %d", "%b %d", "%B %d"):
        try:
            dt = datetime.strptime(s, fmt)
            year_val = year if year is not None else 2024
            dt = dt.replace(year=year_val)
            return dt.date().isoformat()
        except Exception:
            continue

    try:
        dt = datetime.fromisoformat(s)
        return dt.date().isoformat()
    except Exception:
        return None


# ---------------------------------------------------------
# NOTES BLOCK CLEANING
# ---------------------------------------------------------

def clean_notes_block(notes: List[str]) -> List[str]:
    """Split multi-line notes into clean bullet points."""
    out = []
    for n in notes:
        n = normalize_text(n)
        if not n:
            continue
        parts = re.split(r"[•\-–]\s*", n)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out


# ---------------------------------------------------------
# EXCEL INGESTION
# ---------------------------------------------------------

def ingest_excel_file(path: str) -> Dict[str, Any]:
    """Ingest an Excel file into structured sheet objects."""
    xls = pd.ExcelFile(path)
    sheets: Dict[str, Any] = {}

    for sheet in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
        except Exception as e:
            sheets[sheet] = {
                "table": pd.DataFrame(),
                "detected_header_row": None,
                "notes": [f"read_error: {e}"],
                "raw_preview": pd.DataFrame(),
            }
            continue

        if raw is None or raw.empty:
            sheets[sheet] = {
                "table": pd.DataFrame(),
                "detected_header_row": None,
                "notes": [],
                "raw_preview": pd.DataFrame(),
            }
            continue

        # SPECIAL CASE: SERVICE PERIODS
        if "service period" in sheet.lower():
            expected_cols = ["service period", "start time", "end time", "description"]
            raw = raw.fillna("").astype(str)

            best_idx = None
            for i in range(min(len(raw), 20)):
                row = [strip_html_tokens(x).strip().lower() for x in raw.iloc[i].tolist()]
                match_count = sum(1 for col in expected_cols if any(col in cell for cell in row))
                if match_count >= 3:
                    best_idx = i
                    break

            if best_idx is not None:
                headers = sanitize_headers(raw.iloc[best_idx].tolist())
                df = raw.copy()
                df.columns = headers
                df = df[best_idx + 1:].reset_index(drop=True)
                df = clean_table(df)
                df = drop_junk_rows(df)

                sheets[sheet] = {
                    "table": df,
                    "detected_header_row": best_idx,
                    "notes": normalize_notes(extract_notes_below_table(raw)),
                    "raw_preview": raw.head(12),
                }
                continue

        # NORMAL INGESTION
        raw_preview = raw.head(12).apply(lambda col: col.map(strip_html_tokens))
        raw_preview = raw_preview.iloc[:, :20].dropna(axis=1, how="all")

        hdr_idx, combined = collapse_multirow_header(raw)

        if hdr_idx is not None and combined is not None:
            headers = sanitize_headers(combined)
            df = raw.copy()
            df.columns = headers
            df = df[hdr_idx + 1:].reset_index(drop=True)
            detected_header_row = hdr_idx

        else:
            detected_header_row = detect_header_row(raw)
            if detected_header_row is not None:
                headers = sanitize_headers(raw.iloc[detected_header_row].tolist())
                df = raw.copy()
                df.columns = headers
                df = df[detected_header_row + 1:].reset_index(drop=True)
            else:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet, header=0, dtype=str)
                    df.columns = sanitize_headers(df.columns.tolist())
                    detected_header_row = 0
                except Exception:
                    df = raw.copy()
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                    detected_header_row = None

        df = clean_table(df)
        df = drop_junk_rows(df)
        notes = normalize_notes(extract_notes_below_table(raw))

        sheets[sheet] = {
            "table": df,
            "detected_header_row": detected_header_row,
            "notes": notes,
            "raw_preview": raw_preview,
        }

    # INGESTION HEALTH CHECKS
    sheets["_warnings"] = []

    expected_patterns = {
        "basic parameters",
        "service periods",
        "compliance notes",
        "store configurations",
        "fixed hours template",
        "monthly roster",
        "shift codes",
        "employee availability",
    }

    for s in sheets:
        if s.startswith("_"):
            continue
        s_lower = s.lower()
        if not any(p in s_lower for p in expected_patterns):
            sheets["_warnings"].append(f"Unexpected sheet: {s}")

    # Schema validation
    for key in sheets:
        if key.lower() == "basic parameters":
            sheets["_warnings"] += validate_schema(
                sheets[key]["table"],
                ["Parameter Name", "Value", "Unit"],
                "Basic Parameters"
            )

        if key.lower() == "service periods":
            sheets["_warnings"] += validate_schema(
                sheets[key]["table"],
                ["Service Period", "Start Time", "End Time"],
                "Service Periods"
            )

    return sheets


# ---------------------------------------------------------
# CSV INGESTION
# ---------------------------------------------------------

def ingest_csv_file(path: str) -> Dict[str, Any]:
    """Ingest CSV into a sheet‑like structure."""
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = sanitize_headers(df.columns.tolist())
        df = clean_table(df)
        df = drop_junk_rows(df)

        raw_preview = df.head(12).apply(lambda col: col.map(strip_html_tokens))
        raw_preview = raw_preview.iloc[:, :20].dropna(axis=1, how="all")

        return {
            "__csv__": {
                "table": df,
                "detected_header_row": 0,
                "notes": [],
                "raw_preview": raw_preview,
            }
        }

    except Exception as e:
        return {
            "__csv__": {
                "table": pd.DataFrame(),
                "detected_header_row": None,
                "notes": [str(e)],
                "raw_preview": pd.DataFrame(),
            }
        }


# ---------------------------------------------------------
# DIRECTORY INGESTION
# ---------------------------------------------------------

def ingest_directory(directory: str = ".") -> Dict[str, Any]:
    """Ingest all Excel/CSV files in a directory."""
    files_in_dir = [
        f for f in os.listdir(directory)
        if f.lower().endswith((".xlsx", ".xls", ".csv"))
    ]

    ingested: Dict[str, Any] = {}

    for fname in sorted(files_in_dir):
        if fname.startswith("~$"):
            continue

        path = os.path.join(directory, fname)

        try:
            if fname.lower().endswith((".xlsx", ".xls")):
                sheets = ingest_excel_file(path)
            else:
                sheets = ingest_csv_file(path)

            ingested[fname] = sheets

        except Exception as e:
            ingested[fname] = {"__error__": str(e)}

    return ingested

# ---------------------------------------------------------
# FILE OUTPUT HELPERS (required by IngestionAgent)
# ---------------------------------------------------------

def save_json(obj, path: str):
    """Write a Python object to JSON with UTF-8 encoding."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_pickle(obj, path: str):
    """Write a Python object to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

