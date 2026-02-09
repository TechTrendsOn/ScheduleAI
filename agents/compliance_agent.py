# pyright: ignore
# type: ignore
# agents/compliance_agent.py
# Extended compliance checker: Fair Work rules, coverage, management, skills

import os
import json
from datetime import datetime, timedelta

import pandas as pd


class ComplianceAgent:
    def __init__(
        self,
        roster_path: str = "data/artifacts/roster_solution.csv",
        params_path: str = "data/artifacts/rostering_parameters.json",
        staffing_path: str = "data/artifacts/staffing_requirements.json",
        management_path: str = "data/artifacts/management_availability.json",
        out_dir: str = "data/artifacts",
        fallback_year: int = 2024,
    ):
        self.roster_path = roster_path
        self.params_path = params_path
        self.staffing_path = staffing_path
        self.management_path = management_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.fallback_year = fallback_year

        self.roster_df = pd.DataFrame()
        self.work_df = pd.DataFrame()
        self.params = {}
        self.staffing = {}
        self.management = {}
        self.service_periods = []
        self.primary_store_id = None

        self.report = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "roster_solution": roster_path,
                "rostering_parameters": params_path,
                "staffing_requirements": staffing_path,
                "management_availability": management_path,
            },
            "summary": {
                "assignments": 0,
                "employees": 0,
                "days": 0,
                "violations_count": 0,
                "violation_types": {},
            },
            "violations": [],
            "notes": [],
            "swaps": [],  # placeholder for SwapAgent to fill
            "coverage_checks": [],
            "management_coverage_checks": [],
        }

    # ---------------------------------------------------------
    # Loading
    # ---------------------------------------------------------
    def load(self):
        if not os.path.exists(self.roster_path):
            raise FileNotFoundError(f"Roster file not found: {self.roster_path}")
        if os.path.getsize(self.roster_path) == 0:
            raise ValueError("Roster file is empty. Ensure SolverAgent produced assignments.")

        self.roster_df = pd.read_csv(self.roster_path)

        if "hours" in self.roster_df.columns:
            hours_series = pd.to_numeric(self.roster_df["hours"], errors="coerce")
            self.roster_df["hours"] = hours_series.fillna(0.0)
        else:
            self.roster_df["hours"] = 0.0

        if os.path.exists(self.params_path):
            with open(self.params_path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
        else:
            self.params = {}
            self.report["notes"].append("Parameters file not found. Using defaults.")


        if os.path.exists(self.staffing_path):
            with open(self.staffing_path, "r", encoding="utf-8") as f:
                self.staffing = json.load(f)
        else:
            self.staffing = {}
            self.report["notes"].append(
                "Staffing requirements file not found. Skipping coverage compliance."
            )

        if os.path.exists(self.management_path):
            with open(self.management_path, "r", encoding="utf-8") as f:
                self.management = json.load(f)
        else:
            self.management = {}
            self.report["notes"].append(
                "Management availability file not found. Skipping management coverage checks."
            )

        self.service_periods = self.params.get("service_periods", [])
        self._normalize_dates()
        self._insert_auto_breaks_if_enabled()
        self._init_work_df()
        self._init_primary_store_id()

        self.report["summary"]["assignments"] = int(len(self.work_df))
        self.report["summary"]["employees"] = int(self.work_df["employee"].nunique())
        if "date_parsed" in self.work_df.columns:
            self.report["summary"]["days"] = int(
                self.work_df["date_parsed"].dt.date.nunique()
            )
        else:
            self.report["summary"]["days"] = 0

    def _normalize_dates(self):
        def parse_date_str(s):
            s = (s or "").strip()
            for fmt in ["%Y-%m-%d", "%a %b %d"]:
                try:
                    dt = datetime.strptime(s, fmt)
                    if fmt == "%a %b %d":
                        dt = dt.replace(year=self.fallback_year)
                    return dt
                except Exception:
                    continue
            try:
                return datetime.fromisoformat(s)
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
            d = parse_date_str(row.get("date", ""))
            parsed_dates.append(d)
            st = parse_time(row.get("start_time", ""))
            et = parse_time(row.get("end_time", ""))
            if d and st and et:
                start_dt.append(d.replace(hour=st[0], minute=st[1]))
                end_dt.append(d.replace(hour=et[0], minute=et[1]))
            else:
                start_dt.append(None)
                end_dt.append(None)

        self.roster_df["date_parsed"] = parsed_dates
        self.roster_df["start_dt"] = start_dt
        self.roster_df["end_dt"] = end_dt

        # Ensure datetimelike dtype for downstream .dt usage
        self.roster_df["date_parsed"] = pd.to_datetime(
            self.roster_df["date_parsed"], errors="coerce"
        )

    def _init_work_df(self):
        if "is_break" not in self.roster_df.columns:
            self.roster_df["is_break"] = False
        self.work_df = self.roster_df[self.roster_df["is_break"] != True].copy()

    def _is_break_row(self, row) -> bool:
        return self._break_kind(row) is not None

    def _break_kind(self, row):
        station = str(row.get("station", "")).lower()
        code = str(row.get("code", "")).lower()
        notes = str(row.get("notes", "")).lower()
        text = " ".join([station, code, notes])

        if "meal" in text:
            return "meal"
        if "rest" in text:
            return "rest"
        if "break" in text:
            return "break"
        return None

    def _insert_auto_breaks_if_enabled(self):
        if self.roster_df.empty:
            return
        if self._param_value("Auto Insert Breaks", default=1) == 0:
            self.roster_df["is_break"] = self.roster_df.apply(self._is_break_row, axis=1)
            return

        base_df = self.roster_df.copy()
        base_df["is_break"] = base_df.apply(self._is_break_row, axis=1)
        base_df = base_df[base_df["is_break"] != True].copy()

        break_rows = []
        for _, row in base_df.iterrows():
            hours = float(row.get("hours") or 0)
            if hours <= 5:
                continue

            start_dt = row.get("start_dt")
            end_dt = row.get("end_dt")
            if start_dt is None or end_dt is None:
                continue

            if hours > 6:
                break_minutes = 30
                break_code = "MEAL_BREAK"
                break_label = "Meal Break"
            else:
                break_minutes = 10
                break_code = "REST_BREAK"
                break_label = "Rest Break"

            shift_minutes = int((end_dt - start_dt).total_seconds() / 60)
            if shift_minutes <= break_minutes:
                continue

            break_start = start_dt + pd.Timedelta(minutes=(shift_minutes - break_minutes) / 2)
            break_end = break_start + pd.Timedelta(minutes=break_minutes)

            break_rows.append(
                {
                    "employee_id": row.get("employee_id"),
                    "employee": row.get("employee"),
                    "type": row.get("type"),
                    "station": break_label,
                    "date": row.get("date"),
                    "code": break_code,
                    "start_time": break_start.strftime("%H:%M"),
                    "end_time": break_end.strftime("%H:%M"),
                    "hours": round(break_minutes / 60.0, 2),
                    "notes": "Auto break",
                    "date_parsed": row.get("date_parsed"),
                    "start_dt": break_start,
                    "end_dt": break_end,
                    "is_break": True,
                }
            )

        if break_rows:
            self.roster_df = pd.concat([self.roster_df, pd.DataFrame(break_rows)], ignore_index=True)
        self.roster_df["is_break"] = self.roster_df.apply(self._is_break_row, axis=1)

    def _init_primary_store_id(self):
        """
        Mirror SolverAgent logic: derive primary_store_id from staffing structured
        """
        structured = self.staffing.get("structured", {})
        if isinstance(structured, dict) and structured:
            store_ids = list(structured.keys())
            self.primary_store_id = store_ids[0]
        elif isinstance(structured, list) and structured:
            # Backward compatibility with list-of-rows format
            self.primary_store_id = structured[0].get("store_id", "Store_1")
        else:
            self.primary_store_id = "Store_1"
            if self.staffing:
                self.report["notes"].append(
                    "Staffing structure not in expected format; defaulting primary_store_id to 'Store_1'."
                )

    # ---------------------------------------------------------
    # Parameter & domain helpers
    # ---------------------------------------------------------
    def _param_value(self, name, default=None):
        cell = self.params.get("basic", {}).get(name, {})
        try:
            return float(cell.get("value", default))
        except Exception:
            return default

    def _weekly_bands(self):
        return {
            "full-time": (35, 45),
            "part-time": (20, 32),
            "casual": (8, 24),
        }

    def _employment_type(self, typ_str):
        t = (typ_str or "").lower()
        if "full" in t:
            return "full-time"
        if "part" in t:
            return "part-time"
        if "casual" in t:
            return "casual"
        return None

    def _period_type_for_service(self, period_name: str) -> str:
        name = period_name.lower()
        if any(k in name for k in ["lunch", "dinner", "closing", "peak", "rush"]):
            return "Peak"
        return "Normal"

    def _staff_field_for_station(self, station: str) -> str:
        s = (station or "").lower()
        if "mccafe" in s or "mc cafe" in s:
            return "mccafe_staff"
        if "dessert" in s:
            return "dessert_station_staff"
        if "counter" in s or "front" in s:
            return "counter_staff"
        if "kitchen" in s:
            return "kitchen_staff"
        return "kitchen_staff"

    # Simple skills model: infer skills from station and type
    def _infer_skill_tags(self, row):
        tags = set()
        station = (row.get("station") or "").lower()
        typ = self._employment_type(row.get("type", ""))

        if "kitchen" in station:
            tags.add("kitchen-trained")
        if "counter" in station:
            tags.add("front-counter")
        if "mccafe" in station:
            tags.add("barista")
        if "dessert" in station:
            tags.add("dessert-station")

        if typ == "full-time":
            tags.add("core-crew")
        elif typ == "part-time":
            tags.add("part-time-crew")
        elif typ == "casual":
            tags.add("casual-crew")

        return tags

    def _required_skill_tags_for_station(self, station):
        s = (station or "").lower()
        req = set()
        if "kitchen" in s:
            req.add("kitchen-trained")
        if "counter" in s:
            req.add("front-counter")
        if "mccafe" in s:
            req.add("barista")
        if "dessert" in s:
            req.add("dessert-station")
        return req

    # ---------------------------------------------------------
    # Violation helper
    # ---------------------------------------------------------
    def _add_violation(self, vtype, row, extra=None):
        item = {
            "type": vtype,
            "employee": row.get("employee"),
            "date": row.get("date"),
            "station": row.get("station"),
            "code": row.get("code"),
        }
        if extra:
            item.update(extra)
        self.report["violations"].append(item)

    # ---------------------------------------------------------
    # Core Fair Work style checks
    # ---------------------------------------------------------
    def check_min_and_max_shift_length(self):
        min_hours = float(self._param_value("Minimum Hours Per Shift", default=3))
        max_hours = float(self._param_value("Maximum Hours Per Shift", default=12))
        for _, row in self.work_df.iterrows():
            h = row["hours"]
            if h < min_hours:
                self._add_violation(
                    "Shift Length",
                    row,
                    {"hours": h, "rule": f"Minimum {min_hours} hours"},
                )
            if h > max_hours:
                self._add_violation(
                    "Shift Length",
                    row,
                    {"hours": h, "rule": f"Maximum {max_hours} hours"},
                )

    def check_rest_periods(self):
        min_rest = float(self._param_value("Minimum Hours Between Shifts", default=10))
        for emp, group in self.work_df.groupby("employee"):
            g = group.sort_values("date_parsed")
            prev_end = None
            for _, row in g.iterrows():
                if prev_end and row["start_dt"]:
                    rest_hours = (row["start_dt"] - prev_end).total_seconds() / 3600.0
                    if rest_hours < min_rest:
                        self._add_violation(
                            "Rest Period",
                            row,
                            {
                                "rest_hours": round(rest_hours, 2),
                                "rule": f"Minimum {min_rest} hours",
                            },
                        )
                if row["end_dt"]:
                    prev_end = row["end_dt"]

    def check_consecutive_days(self):
        max_consec = float(
            self._param_value("Maximum Consecutive Working Days", default=6)
        )
        if "date_parsed" not in self.work_df.columns:
            return

        df = self.work_df.dropna(subset=["date_parsed"]).copy()
        df["date_only"] = df["date_parsed"].dt.date

        for emp, group in df.groupby("employee"):
            days = sorted(group["date_only"].unique())
            if not days:
                continue

            streak = 1
            for i in range(1, len(days)):
                if (days[i] - days[i - 1]) == timedelta(days=1):
                    streak += 1
                    if streak > max_consec:
                        self.report["violations"].append(
                            {
                                "type": "Consecutive Days",
                                "employee": emp,
                                "date": str(days[i]),
                                "station": None,
                                "code": None,
                                "streak": streak,
                                "rule": f"Maximum {max_consec} consecutive working days",
                            }
                        )
                else:
                    streak = 1

    def check_daily_hours(self):
        max_daily = float(self._param_value("Maximum Hours Per Shift", default=12))
        if "date_parsed" not in self.work_df.columns:
            return

        df = self.work_df.dropna(subset=["date_parsed"]).copy()
        df["date_only"] = df["date_parsed"].dt.date

        daily = df.groupby(["employee", "date_only"])["hours"].sum().reset_index()
        for _, row in daily.iterrows():
            if row["hours"] > max_daily:
                self.report["violations"].append(
                    {
                        "type": "Daily Hours",
                        "employee": row["employee"],
                        "date": str(row["date_only"]),
                        "station": None,
                        "code": None,
                        "hours": round(row["hours"], 2),
                        "rule": f"Maximum {max_daily} hours per day",
                    }
                )

    def check_meal_breaks(self):
        if self._param_value("Meal Breaks Enabled", default=1) == 0:
            return
        for _, row in self.work_df.iterrows():
            h = row["hours"]
            employee = row.get("employee")
            date = row.get("date")
            if h > 6:
                breaks = self.roster_df[
                    (self.roster_df["employee"] == employee)
                    & (self.roster_df["date"] == date)
                    & (self.roster_df["is_break"] == True)
                ]
                has_meal = any(self._break_kind(b) == "meal" for _, b in breaks.iterrows())
                if not has_meal:
                    self._add_violation(
                        "Meal Break",
                        row,
                        {"hours": h, "rule": "30 min unpaid meal break required"},
                    )
            elif h > 5:
                breaks = self.roster_df[
                    (self.roster_df["employee"] == employee)
                    & (self.roster_df["date"] == date)
                    & (self.roster_df["is_break"] == True)
                ]
                has_rest = any(self._break_kind(b) in {"rest", "break"} for _, b in breaks.iterrows())
                if not has_rest:
                    self._add_violation(
                        "Rest Break",
                        row,
                        {"hours": h, "rule": "10 min paid rest break required"},
                    )

    def check_weekly_hours(self):
        bands = self._weekly_bands()
        if "type" not in self.work_df.columns:
            return

        df = self.work_df.dropna(subset=["date_parsed"]).copy()
        df["iso_year"] = df["date_parsed"].dt.isocalendar().year
        df["iso_week"] = df["date_parsed"].dt.isocalendar().week

        for (emp, year, week), group in df.groupby(
            ["employee", "iso_year", "iso_week"]
        ):
            etype = self._employment_type(group.iloc[0].get("type", ""))
            if etype not in bands:
                continue

            lo, hi = bands[etype]
            total_hours = group["hours"].sum()
            if total_hours < lo or total_hours > hi:
                self.report["violations"].append(
                    {
                        "type": "Weekly Hours",
                        "employee": emp,
                        "iso_year": int(year),
                        "iso_week": int(week),
                        "total_hours": round(total_hours, 2),
                        "rule": f"{lo}–{hi} hours for {etype}",
                    }
                )

    # ---------------------------------------------------------
    # Coverage vs staffing_requirements
    # ---------------------------------------------------------
    def check_station_coverage(self):
        if not self.staffing or not self.service_periods:
            return

        structured = self.staffing.get("structured", {})
        staffing_index = {}

        # New nested format: structured[store_id][period_type] -> dict
        if isinstance(structured, dict) and structured:
            for store_id, per_store in structured.items():
                if not isinstance(per_store, dict):
                    continue
                for period_type, row in per_store.items():
                    if isinstance(row, dict):
                        staffing_index[(store_id, period_type)] = row
        # Backward compatible: list of rows
        elif isinstance(structured, list):
            for r in structured:
                if not isinstance(r, dict):
                    continue
                sid = r.get("store_id", "Store_1")
                ptype = r.get("period_type", "Normal")
                staffing_index[(sid, ptype)] = r

        store_id = self.primary_store_id or "Store_1"

        df = self.work_df.dropna(subset=["date_parsed"]).copy()
        df["date_only"] = df["date_parsed"].dt.date

        for d in sorted(df["date_only"].unique()):
            day_df = df[df["date_only"] == d]

            for station in day_df["station"].dropna().unique():
                station_rows = day_df[day_df["station"] == station]

                for p in self.service_periods:
                    pname = p["period"]
                    start_str = p["start"]
                    end_str = p["end"]

                    def _parse_min(s):
                        try:
                            hh, mm = s.split(":")
                            return int(hh) * 60 + int(mm)
                        except Exception:
                            return None

                    p_start = _parse_min(start_str)
                    p_end = _parse_min(end_str)
                    if p_start is None or p_end is None:
                        continue

                    count = 0
                    for _, r in station_rows.iterrows():
                        st = r["start_dt"]
                        et = r["end_dt"]
                        if not st or not et:
                            continue
                        smin = st.hour * 60 + st.minute
                        emin = et.hour * 60 + et.minute
                        if not (emin <= p_start or p_end <= smin):
                            count += 1

                    period_type = self._period_type_for_service(pname)
                    cfg = staffing_index.get((store_id, period_type))
                    if not cfg:
                        continue

                    field = self._staff_field_for_station(station)
                    try:
                        required = float(cfg.get(field, 0.0))
                    except Exception:
                        required = 0.0
                    if required <= 0:
                        continue

                    shortfall = required - count
                    self.report["coverage_checks"].append(
                        {
                            "date": str(d),
                            "station": station,
                            "service_period": pname,
                            "period_type": period_type,
                            "required": required,
                            "actual": count,
                            "shortfall": shortfall if shortfall > 0 else 0,
                        }
                    )

                    if shortfall > 0:
                        self.report["violations"].append(
                            {
                                "type": "Coverage",
                                "employee": None,
                                "date": str(d),
                                "station": station,
                                "code": None,
                                "service_period": pname,
                                "period_type": period_type,
                                "required": required,
                                "actual": count,
                                "rule": "Insufficient staffing vs staffing_requirements",
                            }
                        )

    # ---------------------------------------------------------
    # Management presence / coverage
    # ---------------------------------------------------------
    def check_management_presence(self):
        if not self.management or not self.service_periods:
            return

        structured = self.management.get("structured", [])
        if not structured:
            return

        mgr_df = pd.DataFrame(structured)

        if "date_parsed" not in self.work_df.columns:
            return

        all_dates = sorted(self.work_df["date_parsed"].dropna().dt.date.unique())
        if not all_dates:
            return

        def _fmt_day_no_pad(dt):
            # Windows strftime does not support "%-d".
            # We need labels like "Mon 9" to match management_availability.json.
            return f"{dt.strftime('%a')} {dt.day}"

        def _fmt_day_pad(dt):
            return f"{dt.strftime('%a')} {dt.day:02d}"

        def date_label(d):
            try:
                return _fmt_day_no_pad(d)
            except Exception:
                return str(d)

        manager_shifts_by_label = {}
        for _, row in mgr_df.iterrows():
            name = row.get("name")
            shifts = row.get("shifts", {})
            for lbl, token in shifts.items():
                manager_shifts_by_label.setdefault(lbl, []).append((name, str(token)))

        def label_for_date(d):
            dt = datetime(d.year, d.month, d.day)
            candidates = [
                _fmt_day_no_pad(dt),  # Mon 9
                _fmt_day_pad(dt),     # Mon 09
            ]
            for c in candidates:
                if c in manager_shifts_by_label:
                    return c
            return None

        for d in all_dates:
            lbl = label_for_date(d)
            if not lbl:
                continue

            mgr_shifts = manager_shifts_by_label.get(lbl, [])

            for p in self.service_periods:
                pname = p["period"]
                start_str = p["start"]
                end_str = p["end"]

                def _parse_min(s):
                    try:
                        hh, mm = s.split(":")
                        return int(hh) * 60 + int(mm)
                    except Exception:
                        return None

                p_start = _parse_min(start_str)
                p_end = _parse_min(end_str)
                if p_start is None or p_end is None:
                    continue

                manager_on_duty = False
                for _, token in mgr_shifts:
                    tok = (token or "").strip()
                    if tok and tok not in ["/", "NA"]:
                        manager_on_duty = True
                        break

                self.report["management_coverage_checks"].append(
                    {
                        "date": str(d),
                        "service_period": pname,
                        "manager_on_duty": manager_on_duty,
                    }
                )

                if not manager_on_duty:
                    self.report["violations"].append(
                        {
                            "type": "Management Coverage",
                            "employee": None,
                            "date": str(d),
                            "station": "Management",
                            "code": None,
                            "service_period": pname,
                            "rule": "At least one manager must be on duty at all times.",
                        }
                    )

    # ---------------------------------------------------------
    # Skill matching / station qualification
    # ---------------------------------------------------------
    def check_skill_matching(self):
        if "station" not in self.work_df.columns:
            return

        for _, row in self.work_df.iterrows():
            station = row.get("station")
            if not station:
                continue

            emp_skills = self._infer_skill_tags(row)
            req_skills = self._required_skill_tags_for_station(station)

            missing = req_skills - emp_skills
            if missing:
                self._add_violation(
                    "Skill Matching",
                    row,
                    {
                        "required_skills": sorted(req_skills),
                        "employee_skills": sorted(emp_skills),
                        "rule": "Employee not qualified for station based on inferred skills.",
                    },
                )

    # ---------------------------------------------------------
    # Notes generator
    # ---------------------------------------------------------
    def _generate_notes(self):
        v = self.report["violations"]
        notes = []

        meal_breaks = [x for x in v if x["type"] == "Meal Break"]
        if meal_breaks:
            notes.append(
                "Meal break violations are expected unless the solver explicitly models break windows."
            )

        rest = [x for x in v if x["type"] == "Rest Period"]
        if rest:
            notes.append(
                f"{len(rest)} rest period violations detected. Consider adjusting solver rest constraints or shift spacing."
            )

        weekly = [x for x in v if x["type"] == "Weekly Hours"]
        if weekly:
            notes.append(
                "Weekly hours violations indicate imbalance in shift distribution across employees."
            )

        coverage = [x for x in v if x["type"] == "Coverage"]
        if coverage:
            notes.append(
                "Coverage violations indicate gaps between staffing requirements and scheduled headcount for certain stations/periods."
            )

        mgmt = [x for x in v if x["type"] == "Management Coverage"]
        if mgmt:
            notes.append(
                "Management coverage violations indicate periods where no manager was detected on duty."
            )

        skills = [x for x in v if x["type"] == "Skill Matching"]
        if skills:
            notes.append(
                "Skill matching violations indicate employees assigned to stations for which they may not be qualified."
            )

        if not notes:
            notes.append("Roster appears balanced under the current checks.")
        return notes

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------
    def save(self):
        self.report["summary"]["violations_count"] = len(self.report["violations"])
        vt = {}
        for v in self.report["violations"]:
            vt[v["type"]] = vt.get(v["type"], 0) + 1
        self.report["summary"]["violation_types"] = vt
        self.report["notes"] = self._generate_notes()

        out_path = os.path.join(self.out_dir, "compliance_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print("Compliance report written ->", out_path)

    # ---------------------------------------------------------
    # End-to-end
    # ---------------------------------------------------------
    def run(self):
        print("Running ComplianceAgent (fully extended, ingestion-aligned)...")
        self.load()
        if self.roster_df.empty:
            self.report["notes"].append("Roster has 0 assignments; skipping checks.")
            self.save()
            return self.report

        self.check_min_and_max_shift_length()
        self.check_rest_periods()
        self.check_consecutive_days()
        self.check_daily_hours()
        self.check_meal_breaks()
        self.check_weekly_hours()
        self.check_station_coverage()
        self.check_management_presence()
        self.check_skill_matching()
        self.save()
        return self.report
