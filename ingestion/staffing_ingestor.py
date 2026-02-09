# ingestion/staffing_ingestor.py
# Production‑grade staffing ingestion with semantic validation + multi-store structure

from typing import Dict, Any
import pandas as pd

from .common_utils import (
    drop_junk_rows,
    sanitize_headers,
    normalize_store_id,
    safe_float,
)


class StaffingIngestor:
    def extract(self, ingested: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts:
        - raw_staff_estimates: cleaned raw table
        - structured: staffing requirements per store_id + period_type
        - metadata: preserved raw tables for auditability
        - warnings: semantic validation issues
        """

        staffing: Dict[str, Any] = {
            "raw_staff_estimates": [],
            "structured": {},
            "metadata": {},
            "warnings": [],
        }

        for fname, sheets in ingested.items():
            if "store_structure_staff_estimate" not in fname.lower():
                continue
            if not isinstance(sheets, dict):
                continue

            for sname, obj in sheets.items():
                if not isinstance(obj, dict):
                    continue

                tbl = obj.get("table")
                if not isinstance(tbl, pd.DataFrame) or tbl.empty:
                    continue

                # ---------------------------------------------------------
                # CLEAN TABLE
                # ---------------------------------------------------------
                tbl = drop_junk_rows(tbl)
                tbl.columns = sanitize_headers(tbl.columns.tolist())

                # Remove fully empty rows
                def row_is_empty(row):
                    vals = [str(v).strip().lower() for v in row.tolist()]
                    return all(v in ("", "nan", "none") for v in vals)

                tbl = tbl[~tbl.apply(row_is_empty, axis=1)].reset_index(drop=True)

                # Save raw table + metadata
                staffing["raw_staff_estimates"] = tbl.to_dict(orient="records")
                staffing["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")

                # ---------------------------------------------------------
                # STRUCTURED EXTRACTION (multi-store)
                # ---------------------------------------------------------
                structured = {}

                for _, row in tbl.iterrows():
                    raw_store_id = str(row.get("store_id", "")).strip()
                    store_id = normalize_store_id(raw_store_id)

                    location = str(row.get("store_location_type", "")).strip()
                    period = str(row.get("period_type", "")).strip()

                    # Skip incomplete rows
                    if not store_id or not period or store_id.lower() in ("nan", "none"):
                        staffing["warnings"].append(
                            f"Skipping incomplete staffing row: {row.to_dict()}"
                        )
                        continue

                    # Safe numeric conversion
                    def _num(x):
                        val = safe_float(x, default=None)
                        if val is None:
                            staffing["warnings"].append(
                                f"Invalid numeric value '{x}' in staffing row: {row.to_dict()}"
                            )
                            return None
                        if val < 0:
                            staffing["warnings"].append(
                                f"Negative staffing value detected for {store_id}/{period}: {x}"
                            )
                        return val

                    # Initialize store bucket
                    if store_id not in structured:
                        structured[store_id] = {}

                    # Initialize period bucket
                    structured[store_id][period] = {
                        "store_id": store_id,
                        "store_location_type": location,
                        "period_type": period,
                        "kitchen_staff": _num(row.get("kitchen_staff")),
                        "counter_staff": _num(row.get("counter_staff")),
                        "mccafe_staff": _num(row.get("mccafe_staff")),
                        "dessert_station_staff": _num(row.get("dessert_station_staff")),
                        "offline_dessert_station_staff": _num(row.get("offline_dessert_station_staff")),
                    }

                staffing["structured"] = structured

        return staffing
