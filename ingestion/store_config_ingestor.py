# ingestion/store_config_ingestor.py
# Production‑grade store configuration ingestion with multi‑store support

from typing import Dict, Any
import pandas as pd
import re

from .common_utils import (
    drop_junk_rows,
    sanitize_headers,
    normalize_store_id,
    safe_time,
    time_to_minutes,
    validate_time_range,
)


class StoreConfigIngestor:
    def extract(self, ingested: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts:
        - raw_config_table: cleaned raw table
        - structured: parsed store attributes per store_id
        - metadata: preserved raw rows for auditability
        - warnings: semantic validation issues
        """

        cfg: Dict[str, Any] = {
            "raw_config_table": [],
            "structured": {},
            "metadata": {},
            "warnings": [],
        }

        # ---------------------------------------------------------
        # Locate the correct file
        # ---------------------------------------------------------
        for fname, sheets in ingested.items():
            if "store_configurations" not in fname.lower():
                continue
            if not isinstance(sheets, dict):
                continue

            # ---------------------------------------------------------
            # Locate the correct sheet
            # ---------------------------------------------------------
            for sname, obj in sheets.items():
                if not isinstance(obj, dict):
                    continue

                tbl = obj.get("table")
                if not isinstance(tbl, pd.DataFrame) or tbl.empty:
                    continue

                if "store configurations" not in sname.lower():
                    continue

                # ---------------------------------------------------------
                # Clean table
                # ---------------------------------------------------------
                tbl = drop_junk_rows(tbl)
                tbl.columns = sanitize_headers(tbl.columns.tolist())

                # Remove fully empty rows
                def row_is_empty(row):
                    vals = [str(v).strip().lower() for v in row.tolist()]
                    return all(v in ("", "nan", "none") for v in vals)

                tbl = tbl[~tbl.apply(row_is_empty, axis=1)].reset_index(drop=True)

                # Save raw table
                cfg["raw_config_table"] = tbl.to_dict(orient="records")
                cfg["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")

                # ---------------------------------------------------------
                # STRUCTURED EXTRACTION (multi-store)
                # ---------------------------------------------------------
                structured = {}

                # The input is a "Configuration Item" key column + one column per store.
                # In this dataset the store columns are typically "Unnamed: 1", "Unnamed: 2" etc,
                # and the *first row* contains the store labels (e.g. "Store 1: CBD Core Area").
                key_col = tbl.columns[0]

                header_idx = None
                for i in range(min(10, len(tbl))):
                    v = str(tbl.iloc[i].get(key_col, "")).strip().lower()
                    if v == "configuration item":
                        header_idx = i
                        break
                if header_idx is None:
                    header_idx = 0

                store_cols = [c for c in tbl.columns if c != key_col]
                if not store_cols:
                    cfg["warnings"].append("No store columns found in store_configurations.xlsx")
                    return cfg

                store_label_row = tbl.iloc[header_idx]

                def store_id_from_label(label: str, fallback_col: str) -> str:
                    s = (label or "").strip()
                    m = re.match(r"(?i)^store\s*(\d+)\b", s)
                    if m:
                        return f"Store_{m.group(1)}"
                    return normalize_store_id(s) or normalize_store_id(fallback_col)

                store_col_to_id = {}
                for c in store_cols:
                    label = str(store_label_row.get(c, "") or "").strip()
                    store_col_to_id[c] = store_id_from_label(label, str(c))

                # Normalize first column text
                def norm_key(row):
                    return " ".join(str(row.get(key_col, "")).strip().lower().split())

                # Build structure per store
                for store_col in store_cols:
                    store_id = store_col_to_id[store_col]
                    structured[store_id] = {
                        "store_id": store_id,
                        "store_label": str(store_label_row.get(store_col, "") or "").strip() or None,
                        "structure": {},
                        "operating_hours": {},
                        "attributes": {},
                        "characteristics": {},
                    }

                # ---------------------------------------------------------
                # Populate structured config
                # ---------------------------------------------------------
                for _, row in tbl.iloc[header_idx + 1 :].iterrows():
                    key = norm_key(row)
                    if not key:
                        continue

                    for store_col in store_cols:
                        store_id = store_col_to_id[store_col]
                        val = str(row.get(store_col, "")).strip()
                        if val.lower() in ("", "nan", "none"):
                            val = None

                        # Store structure
                        if key.startswith("store structure"):
                            structured[store_id]["structure"]["store_structure"] = val

                        elif key.startswith("kitchen"):
                            structured[store_id]["structure"]["kitchen"] = val

                        elif key.startswith("counter"):
                            structured[store_id]["structure"]["counter"] = val

                        elif key.startswith("multiple mccafé"):
                            structured[store_id]["structure"]["mccafe"] = val

                        elif key.startswith("dessert station"):
                            structured[store_id]["structure"]["dessert_station"] = val

                        elif key.startswith("offline dessert"):
                            structured[store_id]["structure"]["offline_dessert_station"] = val

                        # Operating hours
                        elif key.startswith("opening time"):
                            structured[store_id]["operating_hours"]["opening_time"] = val

                        elif key.startswith("closing time"):
                            structured[store_id]["operating_hours"]["closing_time"] = val

                        elif key.startswith("daily operating hours"):
                            structured[store_id]["operating_hours"]["daily_hours"] = val

                        # Attributes
                        elif key.startswith("revenue level"):
                            structured[store_id]["attributes"]["revenue_level"] = val

                        elif key.startswith("store type"):
                            structured[store_id]["attributes"]["store_type"] = val

                        # Characteristics
                        elif key.startswith("location type"):
                            structured[store_id]["characteristics"]["location_type"] = val

                        elif key.startswith("primary customers"):
                            structured[store_id]["characteristics"]["primary_customers"] = val

                        elif key.startswith("peak hours"):
                            structured[store_id]["characteristics"]["peak_hours"] = val

                        elif key.startswith("coffee demand"):
                            structured[store_id]["characteristics"]["coffee_demand"] = val

                        elif key.startswith("dessert demand"):
                            structured[store_id]["characteristics"]["dessert_demand"] = val

                        elif key.startswith("traffic pattern"):
                            structured[store_id]["characteristics"]["traffic_pattern"] = val

                        elif key.startswith("average daily customers"):
                            structured[store_id]["characteristics"]["average_daily_customers"] = val

                # ---------------------------------------------------------
                # SEMANTIC VALIDATION
                # ---------------------------------------------------------
                for store_id, data in structured.items():
                    oh = data["operating_hours"]

                    # Validate opening/closing times
                    if "opening_time" in oh and "closing_time" in oh:
                        start = oh["opening_time"]
                        end = oh["closing_time"]

                        if safe_time(start) is None or safe_time(end) is None:
                            cfg["warnings"].append(
                                f"{store_id}: Invalid time format in operating hours ({start}–{end})"
                            )
                        elif not validate_time_range(start, end):
                            cfg["warnings"].append(
                                f"{store_id}: Opening time is not before closing time ({start}–{end})"
                            )

                    # Validate daily hours
                    if "daily_hours" in oh:
                        try:
                            float(str(oh["daily_hours"]).split()[0])
                        except Exception:
                            cfg["warnings"].append(
                                f"{store_id}: Invalid daily operating hours format ({oh['daily_hours']})"
                            )

                cfg["structured"] = structured

        return cfg
