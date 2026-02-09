# ingestion/task_demand_ingestor.py
# Production‑grade task demand ingestion with semantic validation

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .common_utils import (
    drop_junk_rows,
    sanitize_headers,
    safe_float,
    is_date_column,
)


class TaskDemandIngestor:
    def extract(self, ingested: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts:
        - structured task demand (tasks)
        - metadata: raw tables for auditability
        - warnings: semantic validation issues
        """

        tasks: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}
        warnings: List[str] = []

        for fname, sheets in ingested.items():
            if "fixed_hours_template" not in fname.lower():
                continue
            if not isinstance(sheets, dict):
                continue

            for sname, obj in sheets.items():
                if not isinstance(obj, dict):
                    continue

                tbl = obj.get("table")
                if not isinstance(tbl, pd.DataFrame) or tbl.empty:
                    continue

                # Clean table
                tbl = drop_junk_rows(tbl)
                tbl.columns = sanitize_headers(tbl.columns.tolist())

                # Save raw table for metadata
                metadata[f"{fname}/{sname}"] = tbl.to_dict("records")

                # Heuristic: only sheets containing weekday schedule data
                if any(is_date_column(c) for c in tbl.columns):
                    extracted, warns = self._extract_from_table(tbl)
                    tasks.extend(extracted)
                    warnings.extend(warns)

        return {
            "tasks": tasks,
            "metadata": metadata,
            "warnings": warnings,
        }

    # ---------------------------------------------------------
    # INTERNAL: EXTRACT TASK DEMAND FROM ONE TABLE
    # ---------------------------------------------------------
    def _extract_from_table(self, df: pd.DataFrame) -> (List[Dict[str, Any]], List[str]):
        df = df.copy()
        warnings = []

        # Detect task column
        task_col = next(
            (c for c in df.columns if "fixed hours type" in c.lower() or "task" in c.lower()),
            df.columns[0],
        )

        # Detect day columns using helper
        day_cols = [c for c in df.columns if is_date_column(c)]

        if not day_cols:
            warnings.append("No weekday columns detected in fixed hours template")

        # Detect flexibility column (optional)
        flex_col = next((c for c in df.columns if "flexible" in c.lower()), None)

        out: List[Dict[str, Any]] = []

        for _, r in df.iterrows():
            task_name = str(r.get(task_col, "")).strip()

            # Skip template/example/footer rows + skip "nan"
            if (
                task_name == ""
                or task_name.lower() == "nan"
                or task_name.lower().startswith(("template", "example", "time schedule", "task", "•"))
            ):
                continue

            row_dict: Dict[str, Any] = {"task": task_name}

            # Flexibility flag
            if flex_col:
                flex_val = str(r.get(flex_col, "")).strip().lower()
                row_dict["is_flexible"] = flex_val in ("yes", "true", "1", "☒", "x")
            else:
                row_dict["is_flexible"] = None

            numeric_values = []
            weekly_total = 0.0

            for c in day_cols:
                val_str = str(r.get(c, "")).strip()

                # Map day name (Mon, Tue, Wed…)
                day_name = None
                c_low = c.lower()
                for d in [
                    "monday", "tuesday", "wednesday",
                    "thursday", "friday", "saturday", "sunday"
                ]:
                    if d in c_low:
                        day_name = d[:3]  # mon, tue, wed…
                        break

                if not day_name:
                    continue

                # Safe numeric conversion
                num = safe_float(val_str, default=0.0)
                if num is None:
                    num = 0.0
                    warnings.append(f"Non-numeric value '{val_str}' in task '{task_name}'")

                row_dict[day_name] = num
                numeric_values.append(num)
                weekly_total += num

            # Skip rows where ALL day values are zero or NaN
            if all((v is None or v == 0 or (isinstance(v, float) and np.isnan(v)))
                   for v in numeric_values):
                continue

            row_dict["weekly_total"] = round(weekly_total, 2)

            out.append(row_dict)

        return out, warnings
