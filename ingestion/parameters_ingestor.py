# ingestion/parameters_ingestor.py
# Production‑grade parameter ingestion with semantic validation + structured rules

from typing import Dict, Any, List
import pandas as pd
import re

from .common_utils import (
    drop_junk_rows,
    normalize_notes,
    clean_notes_block,
    safe_float,
    safe_time,
    time_to_minutes,
    validate_time_range,
)


class ParametersIngestor:
    """
    Extracts structured rostering parameters from mixed Excel sheets.
    Produces:
      - basic parameters (value/unit/notes)
      - structured service periods
      - structured compliance rules (penalties, split shifts, Fair Work)
      - metadata for auditability
    """

    def extract(self, ingested: Dict[str, Any]) -> Dict[str, Any]:

        parameters: Dict[str, Any] = {
            "basic": {},
            "service_periods": [],
            "compliance_notes": [],
            "penalty_rates": {},
            "split_shift_rules": {},
            "fair_work_rules": {},
            "public_holidays": [],
            "other_parameters": {},
            "metadata": {},
            "warnings": [],
        }

        # ---------------------------------------------------------
        # Iterate through all ingested files + sheets
        # ---------------------------------------------------------
        for fname, sheets in ingested.items():
            if not isinstance(sheets, dict):
                continue

            for sname, obj in sheets.items():
                if not isinstance(obj, dict):
                    continue

                tbl = obj.get("table")
                if not isinstance(tbl, pd.DataFrame) or tbl.empty:
                    continue

                tbl = drop_junk_rows(tbl)
                sname_lower = sname.lower()

                # Helper: skip empty rows
                def row_is_empty(row):
                    vals = [str(v).strip().lower() for v in row.tolist()]
                    return all(v in ("", "nan", "none") for v in vals)

                # ---------------------------------------------------------
                # BASIC PARAMETERS
                # ---------------------------------------------------------
                if "basic parameters" in sname_lower:
                    tbl = tbl.rename(columns={c: c.strip() for c in tbl.columns})

                    for _, row in tbl.iterrows():
                        if row_is_empty(row):
                            continue

                        pname = str(row.get("Parameter Name", "")).strip()
                        if not pname or pname.lower() == "parameter name":
                            continue

                        parameters["basic"][pname] = {
                            "value": str(row.get("Value", "")).strip(),
                            "unit": str(row.get("Unit", "")).strip(),
                            "notes": str(row.get("Notes", "")).strip(),
                        }

                    parameters["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")
                    continue

                # ---------------------------------------------------------
                # SERVICE PERIODS
                # ---------------------------------------------------------
                if "service period" in sname_lower:

                    def norm(s):
                        return (
                            str(s)
                            .strip()
                            .lower()
                            .replace("\xa0", " ")
                            .replace("\u200b", "")
                            .replace("\u200c", "")
                            .replace("\u200d", "")
                            .replace("\u202f", " ")
                            .replace("\u205f", " ")
                        )

                    tbl.columns = [" ".join(norm(c).split()) for c in tbl.columns]

                    for _, row in tbl.iterrows():
                        if row_is_empty(row):
                            continue

                        period = str(row.get("service period", "")).strip()
                        start = (
                            str(row.get("start time", "")).strip()
                            or str(row.get("start", "")).strip()
                        )
                        end = (
                            str(row.get("end time", "")).strip()
                            or str(row.get("end", "")).strip()
                        )
                        desc = str(row.get("description", "")).strip()

                        if not (period and start and end):
                            parameters["warnings"].append(
                                f"Service period missing fields: {row.to_dict()}"
                            )
                            continue

                        # Validate time format using shared helpers
                        if safe_time(start) is None or safe_time(end) is None:
                            parameters["warnings"].append(
                                f"Invalid time format in service period: {row.to_dict()}"
                            )
                            continue

                        parameters["service_periods"].append(
                            {
                                "period": period,
                                "start": start,
                                "end": end,
                                "description": desc,
                            }
                        )

                    parameters["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")
                    continue

                # ---------------------------------------------------------
                # COMPLIANCE NOTES
                # ---------------------------------------------------------
                if "compliance notes" in sname_lower:
                    notes = []

                    for _, row in tbl.iterrows():
                        if row_is_empty(row):
                            continue

                        line = " ".join(
                            str(x).strip() for x in row.tolist() if str(x).strip() != ""
                        )
                        if line:
                            notes.append(line)

                    cleaned = clean_notes_block(normalize_notes(notes))
                    parameters["compliance_notes"] += cleaned
                    parameters["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")
                    continue

                # ---------------------------------------------------------
                # OTHER PARAMETERS (fallback)
                # ---------------------------------------------------------
                cleaned_rows = [
                    r.to_dict() for _, r in tbl.iterrows() if not row_is_empty(r)
                ]

                if cleaned_rows:
                    parameters["other_parameters"][f"{fname}/{sname}"] = cleaned_rows

                parameters["metadata"][f"{fname}/{sname}"] = tbl.to_dict("records")

        # ---------------------------------------------------------
        # STRUCTURE COMPLIANCE RULES
        # ---------------------------------------------------------
        self._extract_penalty_rates(parameters)
        self._extract_split_shift_rules(parameters)
        self._extract_fair_work_rules(parameters)
        self._extract_public_holidays(parameters)
        self._validate_service_periods(parameters)

        return parameters

    # ---------------------------------------------------------
    # PENALTY RATES
    # ---------------------------------------------------------
    def _extract_penalty_rates(self, parameters):
        notes = " ".join(parameters.get("compliance_notes", [])).lower()

        def find_rate(pattern):
            match = re.search(pattern, notes)
            if match:
                return safe_float(match.group(1))
            return None

        parameters["penalty_rates"] = {
            "saturday": find_rate(r"saturday:\s*([0-9.]+)"),
            "sunday": find_rate(r"sunday:\s*([0-9.]+)"),
            "public_holiday": find_rate(r"public holidays:\s*([0-9.]+)"),
            "evening_after_9pm": find_rate(r"evening.*after 9.*?([0-9.]+)"),
        }

    # ---------------------------------------------------------
    # SPLIT SHIFT RULES
    # ---------------------------------------------------------
    def _extract_split_shift_rules(self, parameters):
        notes = " ".join(parameters.get("compliance_notes", [])).lower()

        def find(pattern):
            match = re.search(pattern, notes)
            return safe_float(match.group(1)) if match else None

        parameters["split_shift_rules"] = {
            "max_segments": find(r"maximum segments per shift.*?(\d+)"),
            "max_gap_hours": find(r"maximum gap.*?(\d+)"),
            "max_span_hours": find(r"must not exceed\s*(\d+)\s*hours"),
        }

    # ---------------------------------------------------------
    # FAIR WORK RULES
    # ---------------------------------------------------------
    def _extract_fair_work_rules(self, parameters):
        basic = parameters.get("basic", {})

        def get(name, default=None):
            cell = basic.get(name, {})
            return safe_float(cell.get("value", default), default)

        parameters["fair_work_rules"] = {
            "min_shift_hours": get("Minimum Hours Per Shift", 3),
            "max_shift_hours": get("Maximum Hours Per Shift", 12),
            "min_rest_hours": get("Minimum Hours Between Shifts", 10),
            "max_consecutive_days": get("Maximum Consecutive Working Days", 6),
            "monthly_standard_hours": get("Monthly Standard Required Hours", 152),
        }

    # ---------------------------------------------------------
    # PUBLIC HOLIDAYS
    # ---------------------------------------------------------
    def _extract_public_holidays(self, parameters):
        notes = parameters.get("compliance_notes", [])
        holidays = []

        for n in notes:
            if "day" in n.lower() and "holiday" in n.lower():
                holidays.append({"name": n, "source": "compliance_notes"})

        # Melbourne Cup Day inference (existing logic)
        if any("melbourne cup day" in n.lower() for n in notes):
            holidays.append(
                {
                    "name": "Melbourne Cup Day",
                    "date": "2025-11-04",
                    "penalty_multiplier": 2.25,
                }
            )

        parameters["public_holidays"] = holidays

    # ---------------------------------------------------------
    # VALIDATE SERVICE PERIODS
    # ---------------------------------------------------------
    def _validate_service_periods(self, parameters):
        periods = parameters.get("service_periods", [])
        warnings = parameters["warnings"]

        for p in periods:
            start = p["start"]
            end = p["end"]

            if safe_time(start) is None or safe_time(end) is None:
                warnings.append(f"Invalid time format in service period: {p}")
                continue

            if not validate_time_range(start, end):
                warnings.append(f"Service period has invalid range: {p}")

        # Check overlaps
        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                p1, p2 = periods[i], periods[j]
                s1, e1 = time_to_minutes(p1["start"]), time_to_minutes(p1["end"])
                s2, e2 = time_to_minutes(p2["start"]), time_to_minutes(p2["end"])

                if None in (s1, e1, s2, e2):
                    continue

                if not (e1 <= s2 or e2 <= s1):
                    warnings.append(
                        f"Service period overlap: {p1['period']} & {p2['period']}"
                    )
