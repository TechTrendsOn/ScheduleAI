# pyright: ignore
# agents/swap_agent.py
# Suggests swaps to address violations without introducing new ones

import os
import json
from copy import deepcopy
from datetime import datetime
from typing import cast

import pandas as pd


class SwapAgent:
    """
    SwapAgent:
      - Reads roster_solution.csv + compliance_report.json + parameters
      - Suggests swaps that:
          * avoid overlapping shifts
          * avoid obvious new violations (light simulation)
          * address Coverage, Weekly Hours, Rest Period, Shift Length
      - Writes swaps_report.json and can be used by Streamlit UI
    """

    def __init__(
        self,
        roster_path: str = "data/artifacts/roster_solution.csv",
        compliance_report_path: str = "data/artifacts/compliance_report.json",
        params_path: str = "data/artifacts/rostering_parameters.json",
        availability_path: str = "data/artifacts/availability_tidy.csv",
        out_dir: str = "data/artifacts",
    ):
        self.roster_path = roster_path
        self.compliance_report_path = compliance_report_path
        self.params_path = params_path
        self.availability_path = availability_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.roster_df: pd.DataFrame = pd.DataFrame()
        self.availability_df: pd.DataFrame = pd.DataFrame()
        self.params: dict = {}
        self.violations: list[dict] = []
        self.swaps: list[dict] = []

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------
    def load(self):
        if not os.path.exists(self.roster_path):
            raise FileNotFoundError(f"Roster file not found: {self.roster_path}")
        self.roster_df = pd.read_csv(self.roster_path)

        if os.path.exists(self.availability_path):
            self.availability_df = pd.read_csv(self.availability_path)
        else:
            self.availability_df = pd.DataFrame()

        if "hours" in self.roster_df.columns:
            cleaned_hours = []
            for v in self.roster_df["hours"].tolist():
                try:
                    if v is None or str(v).strip().lower() in ("", "nan", "none"):
                        cleaned_hours.append(0.0)
                    else:
                        cleaned_hours.append(float(v))
                except Exception:
                    cleaned_hours.append(0.0)
            self.roster_df["hours"] = cleaned_hours
        else:
            self.roster_df["hours"] = 0.0

        if os.path.exists(self.params_path):
            with open(self.params_path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
        else:
            self.params = {}


        if not os.path.exists(self.compliance_report_path):
            raise FileNotFoundError(
                f"Compliance report not found: {self.compliance_report_path}"
            )
        with open(self.compliance_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.violations = report.get("violations", [])

        self._normalize_dates()

    def _normalize_dates(self):
        def parse_date(s):
            try:
                return pd.to_datetime(s, errors="coerce")
            except Exception:
                return None

        def parse_time(s):
            try:
                hh, mm = s.split(":")
                return int(hh), int(mm)
            except Exception:
                return None

        parsed_dates, start_dt, end_dt = [], [], []
        for _, row in self.roster_df.iterrows():
            d = parse_date(row.get("date", ""))
            parsed_dates.append(d)
            st = parse_time(row.get("start_time", ""))
            et = parse_time(row.get("end_time", ""))
            if d is not None and st and et:
                start_dt.append(d.replace(hour=st[0], minute=st[1]))
                end_dt.append(d.replace(hour=et[0], minute=et[1]))
            else:
                start_dt.append(None)
                end_dt.append(None)

        self.roster_df["date_parsed"] = parsed_dates
        self.roster_df["start_dt"] = start_dt
        self.roster_df["end_dt"] = end_dt

    def _resolve_day_label(self, day_label: str):
        if day_label in set(self.roster_df["date"].astype(str)):
            return day_label
        try:
            dt = pd.to_datetime(day_label, errors="coerce")
        except Exception:
            return day_label
        if pd.isna(dt):
            return day_label
        day_key = dt.date()
        for _, row in self.roster_df.iterrows():
            d = row.get("date_parsed")
            if d is None:
                continue
            if d.date() == day_key:
                return row.get("date")
        return day_label

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _param_value(self, name, default=None):
        cell = self.params.get("basic", {}).get(name, {})
        try:
            return float(cell.get("value", default))
        except Exception:
            return default

    def _is_employee_available(self, employee, date, start_dt, end_dt):
        # Enforce one-shift-per-day
        emp_shifts = self.roster_df[
            (self.roster_df["employee"] == employee)
            & (self.roster_df["date"] == date)
        ]
        if not emp_shifts.empty:
            return False
        for _, row in emp_shifts.iterrows():
            row_start = row.get("start_dt")
            row_end = row.get("end_dt")
            if row_start is None or row_end is None:
                continue
            if not (end_dt <= row_start or start_dt >= row_end):
                return False
        if not self._is_available_in_source(employee, date, start_dt, end_dt):
            return False
        return True

    def _has_existing_shift_on_date(self, employee, date) -> bool:
        return not self.roster_df[
            (self.roster_df["employee"] == employee)
            & (self.roster_df["date"] == date)
        ].empty

    def _is_available_in_source(self, employee, date, start_dt, end_dt) -> bool:
        if self.availability_df.empty:
            return True

        try:
            target_start = start_dt.strftime("%H:%M")
            target_end = end_dt.strftime("%H:%M")
        except Exception:
            return False

        avail = cast(
            pd.DataFrame,
            self.availability_df.loc[
                (self.availability_df["employee"] == employee)
                & (self.availability_df["date"] == date)
            ],
        )
        if avail.empty:
            return False

        avail = cast(
            pd.DataFrame,
            avail.loc[
                (avail["token"].astype(str) != "/")
                & (avail["availability_status"].astype(str) != "not_available")
            ],
        )
        if avail.empty:
            return False

        # Prefer exact time match
        exact = cast(
            pd.DataFrame,
            avail.loc[
                (avail["start_time"].astype(str) == target_start)
                & (avail["end_time"].astype(str) == target_end)
            ],
        )
        if not exact.empty:
            return True

        # Fallback: any available range that fully covers the shift
        for _, row in avail.iterrows():
            try:
                s = str(row.get("start_time") or "").strip()
                e = str(row.get("end_time") or "").strip()
                if not s or not e:
                    continue
                s_dt = pd.to_datetime(f"{date} {s}", errors="coerce")
                e_dt = pd.to_datetime(f"{date} {e}", errors="coerce")
                if pd.isna(s_dt) or pd.isna(e_dt):
                    continue
                if s_dt <= start_dt and e_dt >= end_dt:
                    return True
            except Exception:
                continue

        return False

    def _simulate_compliance(self, df: pd.DataFrame):
        violations = []

        min_hours = float(self._param_value("Minimum Hours Per Shift", default=3))
        for _, row in df.iterrows():
            if row["hours"] < min_hours:
                violations.append({"type": "Shift Length"})

        min_rest = float(self._param_value("Minimum Hours Between Shifts", default=10))
        if "date_parsed" in df.columns:
            for emp, group in df.groupby("employee"):
                g = group.sort_values("date_parsed")
                prev_end = None
                for _, row in g.iterrows():
                    row_start = row.get("start_dt")
                    row_end = row.get("end_dt")
                    if prev_end is not None and row_start is not None:
                        try:
                            rest_hours = (row_start - prev_end).total_seconds() / 3600.0
                        except Exception:
                            rest_hours = None
                        if rest_hours < min_rest:
                            violations.append({"type": "Rest Period"})
                    if row_end is not None:
                        prev_end = row_end

        return violations

    def _would_swap_create_new_violations(self, emp_a, emp_b, date, station):
        temp = self.roster_df.copy()

        row_a = temp[
            (temp["employee"] == emp_a)
            & (temp["date"] == date)
            & (temp["station"] == station)
        ]
        row_b = temp[
            (temp["employee"] == emp_b)
            & (temp["date"] == date)
            & (temp["station"] == station)
        ]
        if row_a.empty or row_b.empty:
            return False

        idx_a = row_a.index[0]
        idx_b = row_b.index[0]
        temp.at[idx_a, "employee"], temp.at[idx_b, "employee"] = (
            temp.at[idx_b, "employee"],
            temp.at[idx_a, "employee"],
        )

        simulated = self._simulate_compliance(temp)
        return len(simulated) == 0

    def _would_swap_be_safe_for_new_assignment(self, cand_emp, day, start_dt, end_dt):
        temp = self.roster_df.copy()

        if start_dt is None or end_dt is None:
            return False

        new_row = {
            "employee": cand_emp,
            "date": day,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "hours": (end_dt - start_dt).total_seconds() / 3600.0,
        }
        temp = pd.concat([temp, pd.DataFrame([new_row])], ignore_index=True)

        simulated = self._simulate_compliance(temp)
        return len(simulated) == 0

    def _build_swap_record(self, viol, suggested_employee, station, day):
        emp = viol.get("employee")
        swap_id = f"{emp}_{suggested_employee}_{day}_{station}"
        return {
            "swap_id": swap_id,
            "violation": deepcopy(viol),
            "suggested_employee": suggested_employee,
            "station": station,
            "date": day,
            "confirmed": False,
            "rejected": False,
        }

    # ---------------------------------------------------------
    # Swap generation (final unified version)
    # ---------------------------------------------------------
    def generate_swaps(self):
        swaps = []

        # Prioritize violations
        priority = ["Coverage", "Weekly Hours", "Rest Period", "Shift Length"]
        sorted_viol = sorted(
            self.violations,
            key=lambda v: priority.index(v["type"]) if v["type"] in priority else len(priority)
        )

        for v in sorted_viol:
            day = v.get("date")
            emp = v.get("employee")
            if not day:
                continue
            day = self._resolve_day_label(str(day))
            if not emp:
                continue

            viol_row = self.roster_df[
                (self.roster_df["employee"] == emp)
                & (self.roster_df["date"] == day)
            ]
            if viol_row.empty:
                continue

            v_row = viol_row.iloc[0]
            station = v_row.get("station")
            start_dt = v_row.get("start_dt")
            end_dt = v_row.get("end_dt")
            if start_dt is None or end_dt is None:
                continue

            # Candidate pool: same day, any employee
            # Candidate pool: all employees known in roster + availability
            roster_emps = (
                self.roster_df["employee"].dropna().unique()
                if "employee" in self.roster_df.columns
                else []
            )
            candidate_emps = set(roster_emps)
            if not self.availability_df.empty and "employee" in self.availability_df.columns:
                candidate_emps.update(self.availability_df["employee"].dropna().unique())
            candidate_emps = list(candidate_emps)

            valid = []
            for cand_emp in candidate_emps:
                if cand_emp == emp:
                    continue

                # Must be available
                if not self._is_employee_available(cand_emp, day, start_dt, end_dt):
                    continue

                # Must not create new violations (align with final roster rules)
                if self._has_existing_shift_on_date(cand_emp, day):
                    continue
                if not self._would_swap_create_new_violations(emp, cand_emp, day, station):
                    continue

                valid.append(cand_emp)

            # Casual fallback
            if not valid and "type" in self.roster_df.columns:
                casuals = (
                    self.roster_df[
                        self.roster_df["type"].str.contains("casual", case=False, na=False)
                    ]["employee"]
                    .dropna()
                    .unique()
                )
                for cand_emp in casuals:
                    if self._is_employee_available(cand_emp, day, start_dt, end_dt) and \
                       self._would_swap_be_safe_for_new_assignment(cand_emp, day, start_dt, end_dt):
                        valid.append(cand_emp)
                        break

            if not valid:
                continue

            suggested = valid[0]
            swaps.append(self._build_swap_record(v, suggested, station, day))


        self.swaps = swaps

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    def save(self):
        out_path = os.path.join(self.out_dir, "swaps_report.json")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "swaps": self.swaps,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("Swaps report written ->", out_path)

    # ---------------------------------------------------------
    # End-to-end
    # ---------------------------------------------------------
    def run(self):
        print("Running SwapAgent (final unified version)...")
        self.load()
        self.generate_swaps()
        self.save()
        return {"swaps": self.swaps}
