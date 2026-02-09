# pyright: ignore
# ingestion/availability_ingestor.py

import os
from typing import Dict, Any, List, Set

import pandas as pd

from .common_utils import (
    sanitize_headers,
    drop_junk_rows,
    is_date_column,
    is_shift_token,
    time_to_minutes,
    extract_roster_period,
    parse_date_label,
)


class AvailabilityIngestor:
    def __init__(self,
                 availability_file_pattern: str = "employee_availability",
                 out_dir: str = "data/artifacts"):
        self.availability_file_pattern = availability_file_pattern
        self.out_dir = out_dir

    # ---------------------------------------------------------
    # FIND THE AVAILABILITY TABLE
    # ---------------------------------------------------------
    def find_availability_table(self,
                                ingested: Dict[str, Any]) -> pd.DataFrame:
        """
        Locate the availability table by scanning for:
        - filename containing availability_file_pattern
        - sheet containing employee columns + weekday/date columns
        """
        for fname, sheets in ingested.items():
            if self.availability_file_pattern.lower() not in fname.lower():
                continue
            if not isinstance(sheets, dict):
                continue

            for sname, obj in sheets.items():
                tbl = obj.get("table")
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    cols = [str(c).lower() for c in tbl.columns]

                    has_employee = any("employee" in c for c in cols)
                    has_dates = any(is_date_column(c) for c in cols)

                    if has_employee and has_dates:
                        return drop_junk_rows(tbl.copy())

        raise RuntimeError("Availability table not found in ingested data")

    # ---------------------------------------------------------
    # SPLIT EMPLOYEE ROWS VS METADATA
    # ---------------------------------------------------------
    def split_employee_vs_metadata(self, df: pd.DataFrame):
        """
        Employee rows always have:
        - numeric ID
        - non-empty employee name
        - valid Type (Full-Time, Part-Time, Casual)
        """
        df = df.copy()

        def is_employee_row(row):
            # ID must be numeric
            try:
                int(str(row.iloc[0]).strip())
            except Exception:
                return False

            # Employee name must be real
            name = str(row.iloc[1]).strip()
            if name == "" or name.lower() in (
                "legend", "shift codes", "weekly overview", "employee summary"
            ):
                return False

            # Type must be valid
            typ = str(row.iloc[2]).lower()
            if not any(t in typ for t in ("full", "part", "casual")):
                return False

            return True

        employee_mask = df.apply(is_employee_row, axis=1)

        employee_df = df[employee_mask].reset_index(drop=True)
        metadata_df = df[~employee_mask].reset_index(drop=True)

        return employee_df, metadata_df

    # ---------------------------------------------------------
    # CLASSIFY AVAILABILITY TOKEN
    # ---------------------------------------------------------
    def classify_token(self, token: str) -> str:
        """
        Explicitly classify availability tokens:
        - shift: valid shift code (handled later)
        - not_available: '/'
        - leave: 'NA'
        - empty: blank or formatting
        - unknown: unmapped token
        """
        t = token.upper().strip()

        if t in ("", "NAN", "N/A"):
            return "empty"
        if t == "/":
            return "not_available"
        if t == "NA":
            return "leave"
        return "unknown"   # may still be a shift code

    # ---------------------------------------------------------
    # NORMALIZE AVAILABILITY INTO TIDY FORMAT
    # ---------------------------------------------------------
    def normalize(self,
                  raw_avail: pd.DataFrame,
                  shift_map: pd.DataFrame,
                  out_name: str = "availability_tidy.csv",
                  meta_name: str = "availability_metadata.csv") -> Dict[str, Any]:

        # --- Clean and index shift map ---
        shift_map = shift_map.copy()
        shift_map["code"] = shift_map["code"].astype(str).str.strip()
        shift_map = shift_map.set_index("code")

        # --- Clean availability table ---
        avail = raw_avail.copy()
        roster_period = extract_roster_period(avail).get("roster_period")
        avail.columns = sanitize_headers(avail.columns.tolist())
        avail = drop_junk_rows(avail)

        # --- Split into employee rows + metadata rows ---
        employee_df, metadata_df = self.split_employee_vs_metadata(avail)

        # --- Save metadata table ---
        meta_path = os.path.join(self.out_dir, meta_name)
        metadata_df.to_csv(meta_path, index=False)

        # --- Rename columns for solver compatibility ---
        col_list = list(employee_df.columns)
        if len(col_list) >= 4:
            new_cols = ["employee_id", "employee", "type", "station"] + col_list[4:]
            employee_df.columns = new_cols

        meta_cols = ["employee_id", "employee", "type", "station"]
        date_cols = [c for c in employee_df.columns if c not in meta_cols]

        tidy_rows: List[Dict[str, Any]] = []
        unmapped: Set[str] = set()

        # --- Compute hours helper using new time helpers ---
        def _compute_hours(start: str, end: str):
            s = time_to_minutes(start)
            e = time_to_minutes(end)
            if s is None or e is None:
                return None
            return round((e - s) / 60, 2)

        # --- Build tidy rows ---
        for _, row in employee_df.iterrows():
            meta = {k: row[k] for k in meta_cols}

            for date in date_cols:
                raw_token = row[date]
                token = str(raw_token).strip()

                # Skip formatting/junk tokens
                if (
                    token == "" or
                    token.lower() in ("nan", "n/a") or
                    token.endswith(":") or
                    " " in token or
                    token.lower().startswith(("gray", "green", "light", "orange", "yellow"))
                ):
                    continue

                status = self.classify_token(token)

                entry = {
                    **meta,
                    "date": date,
                    "token": token,
                    "availability_status": status,
                    "date_parsed": parse_date_label(str(date), roster_period=roster_period) or "",
                }

                # If not a shift, record minimal info and continue
                if status in ("not_available", "leave", "empty"):
                    entry["start_time"] = ""
                    entry["end_time"] = ""
                    entry["time_kind"] = status
                    entry["hours"] = None
                    tidy_rows.append(entry)
                    continue

                # If token is a valid shift code
                if token in shift_map.index:
                    shift = shift_map.loc[token]
                    start = shift.get("start_time", "")
                    end = shift.get("end_time", "")

                    entry["start_time"] = start
                    entry["end_time"] = end
                    entry["time_kind"] = shift.get("time_kind", "")
                    entry["hours"] = _compute_hours(start, end)

                else:
                    # Unknown token → unmapped
                    entry["start_time"] = ""
                    entry["end_time"] = ""
                    entry["time_kind"] = "unmapped"
                    entry["hours"] = None
                    unmapped.add(token)

                tidy_rows.append(entry)

        # --- Save tidy output ---
        tidy_df = pd.DataFrame(tidy_rows)
        out_path = os.path.join(self.out_dir, out_name)
        tidy_df.to_csv(out_path, index=False)

        return {
            "path": out_path,
            "metadata_path": meta_path,
            "unmapped_tokens": sorted(unmapped),
            "warnings": (
                [f"Unmapped availability tokens: {sorted(unmapped)}"]
                if unmapped else []
            ),
        }
