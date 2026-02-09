# ingestion/management_ingestor.py
# Production‑grade management roster ingestion with semantic validation

from typing import Dict, Any, List
import pandas as pd
import re

from .common_utils import (
    drop_junk_rows,
    sanitize_headers,
    is_date_column,
    is_shift_token,
)


class ManagementIngestor:
    def extract(self, ingested: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts:
        - sheets: raw management roster sheets (cleaned)
        - structured: parsed monthly roster (manager → daily shift tokens)
        - metadata: preserved raw tables for auditability
        - warnings: semantic validation issues
        """

        mgmt: Dict[str, Any] = {
            "sheets": {},
            "structured": [],
            "metadata": {},
            "warnings": [],
        }

        # ---------------------------------------------------------
        # Locate the correct file
        # ---------------------------------------------------------
        for fname, sheets in ingested.items():
            if "management_roster" not in fname.lower():
                continue
            if not isinstance(sheets, dict):
                continue

            # ---------------------------------------------------------
            # Process each sheet
            # ---------------------------------------------------------
            for sname, obj in sheets.items():
                if not isinstance(obj, dict):
                    continue

                tbl = obj.get("table")
                if not isinstance(tbl, pd.DataFrame) or tbl.empty:
                    continue

                # Clean table
                tbl = drop_junk_rows(tbl)

                # Save raw sheet (after basic cleaning)
                mgmt["sheets"][sname] = tbl.to_dict(orient="records")
                mgmt["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")

                # ---------------------------------------------------------
                # STRUCTURED EXTRACTION (Monthly Roster only)
                # ---------------------------------------------------------
                if "monthly roster" in sname.lower():
                    structured, warns = self._extract_monthly_roster(tbl)
                    mgmt["structured"].extend(structured)
                    mgmt["warnings"].extend(warns)

        return mgmt

    # ---------------------------------------------------------
    # INTERNAL: EXTRACT STRUCTURED MANAGER ROSTER
    # ---------------------------------------------------------
    def _extract_monthly_roster(self, df: pd.DataFrame) -> (List[Dict[str, Any]], List[str]):
        df = df.copy()
        warnings = []

        # ---------------------------------------------------------
        # 1. Detect the real header row (contains "Employee Name")
        # ---------------------------------------------------------
        header_row_idx = None
        for i in range(min(20, len(df))):
            row_vals = [str(x).strip().lower() for x in df.iloc[i].tolist()]
            if any("employee name" in v for v in row_vals):
                header_row_idx = i
                break

        if header_row_idx is None:
            warnings.append("Could not detect header row in management roster")
            return [], warnings

        # ---------------------------------------------------------
        # 2. Rebuild DataFrame with correct header
        # ---------------------------------------------------------
        new_header = df.iloc[header_row_idx]
        df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
        df.columns = sanitize_headers([str(x) for x in new_header.tolist()])

        # ---------------------------------------------------------
        # 3. Detect key columns
        # ---------------------------------------------------------
        # common_utils.sanitize_headers preserves spaces (does not snake_case),
        # so support both "employee name" and "employee_name" patterns.
        name_col = next(
            (
                c
                for c in df.columns
                if any(
                    k in str(c).lower()
                    for k in ("employee name", "employee_name")
                )
                or str(c).strip().lower() in ("name", "employee")
            ),
            None,
        )

        pos_col = next(
            (c for c in df.columns if "position" in c.lower() or "role" in c.lower()),
            None,
        )

        # Date columns (Mon 25, Tue 26, etc.)
        date_cols = [
            c
            for c in df.columns
            if c not in {name_col, pos_col} and is_date_column(str(c)) and "unnamed" not in str(c).lower()
        ]

        if not name_col:
            warnings.append("Missing 'Employee Name' column in management roster")
            return [], warnings

        if not date_cols:
            warnings.append("No date columns detected in management roster")
            return [], warnings

        structured: List[Dict[str, Any]] = []
        token_map = {
            "S": "1F",
        }

        # ---------------------------------------------------------
        # 4. Parse each manager row
        # ---------------------------------------------------------
        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()

            # Skip empty or header-like rows
            if not name or name.lower() in ("employee name", "nan", "none"):
                continue

            # Skip bullet rows or key points
            if name.startswith(("•", "-", "key", "point")):
                continue

            position = str(row.get(pos_col, "")).strip() if pos_col else ""

            # Extract daily shift tokens
            shifts: Dict[str, str] = {}
            for c in date_cols:
                token_raw = str(row.get(c, "")).strip()
                token = token_map.get(token_raw, token_raw)
                if token and token.lower() not in ("nan", "", "none"):
                    shifts[c] = token

            if not shifts:
                warnings.append(f"Manager '{name}' has no shift entries")
                continue

            # ---------------------------------------------------------
            # Normalize date labels to match ComplianceAgent expectations
            # (e.g. "Mon 9" / "Mon 09"). Keep a space (not underscore).
            # ---------------------------------------------------------
            normalized_dates = {}
            for col, tok in shifts.items():
                col_clean = " ".join(str(col).replace("\n", " ").split()).strip()
                match = re.match(r"^([A-Za-z]{3,})\s*(\d{1,2})$", col_clean)
                if match:
                    weekday, day = match.group(1).title(), int(match.group(2))
                    normalized_dates[f"{weekday} {day}"] = tok
                else:
                    normalized_dates[col_clean] = tok
                    warnings.append(f"Unrecognized date label '{col_clean}' in management roster")

                # Optional: validate shift token
                if not is_shift_token(tok):
                    warnings.append(f"Unrecognized shift token '{tok}' for manager '{name}'")

            structured.append(
                {
                    "manager_id": name.lower().replace(" ", "_"),
                    "name": name,
                    "position": position,
                    "shifts": normalized_dates,
                }
            )

        return structured, warnings
