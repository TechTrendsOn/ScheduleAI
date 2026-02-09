# agents/solver_agent.py
# CP-SAT roster optimization aligned with ingestion artifacts 

import os
import json
from datetime import datetime
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model


class SolverAgent:
    def __init__(
        self,
        artifact_dir: str = "data/artifacts",
        availability_file: str = "availability_tidy.csv",
        shift_codes_file: str = "shift_codes_cleaned.csv",
        params_file: str = "rostering_parameters.json",
        staffing_file: str = "staffing_requirements.json",
        store_config_file: str = "store_config.json",
        out_dir: str = "data/artifacts",
        time_limit_sec: int = 120,
        num_workers: int = 8,
    ):
        self.artifact_dir = artifact_dir
        self.availability_path = os.path.join(artifact_dir, availability_file)
        self.shift_codes_path = os.path.join(artifact_dir, shift_codes_file)
        self.params_path = os.path.join(artifact_dir, params_file)
        self.staffing_path = os.path.join(artifact_dir, staffing_file)
        self.store_config_path = os.path.join(artifact_dir, store_config_file)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.time_limit_sec = time_limit_sec
        self.num_workers = num_workers

        self.model = cp_model.CpModel()
        # Decision variables: (emp_id, date, token, station) -> Bool
        self.assign = {}
        self.hours = {}       # minutes
        self.start_min = {}   # from midnight
        self.end_min = {}     # from midnight
        self.emp_days = defaultdict(set)
        # Coverage: (period_name, date, station) -> IntVar
        self.coverage_vars = {}
        self.coverage_caps = {}

        # Hard coverage enforcement toggle (keep off for demo stability)
        self.enforce_peak_minimums = False

        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "availability_tidy": self.availability_path,
                "shift_codes_cleaned": self.shift_codes_path,
                "rostering_parameters": self.params_path,
                "staffing_requirements": self.staffing_path,
                "store_config": self.store_config_path,
            },
            "warnings": [],
            "outputs": {},
            "objective_weights": {
                "coverage": 25.0,
                "fairness": 1.0,
                "assignment": 1.0,
                "weekly_lower": 3.0,
                "staffing_alignment": 20.0,
            },
            "enforce_peak_minimums": self.enforce_peak_minimums,
        }

        # Hard constraints (aligned with Fair Work)
        self.MIN_SHIFT_HOURS = 3.0
        self.MIN_REST_HOURS = 10.0
        self.MAX_SHIFT_HOURS = 12.0
        self.MAX_CONSEC_DAYS = 6

        self.service_periods = []    # from rostering_parameters
        # (store_id, period_type) -> dict of staffing fields
        self.staff_targets = {}
        self.emp_types = {}          # emp_id -> type string
        self.weekly_bands = {}       # emp_id -> (lo_hours, hi_hours)
        self.primary_store_id = None
        self.day_index = {}

        # Store operating window (minutes from midnight) – optional
        self.store_open_min = None
        self.store_close_min = None

        # Data artifacts (initialized to avoid Optional typing issues)
        self.avail_df: pd.DataFrame = pd.DataFrame()
        self.shift_df: pd.DataFrame = pd.DataFrame()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _load_csv(self, path: str, required: bool = False) -> pd.DataFrame:
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            self.manifest["warnings"].append(f"Optional file not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path).fillna("")

    def _load_json(self, path, required=False):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required JSON: {path}")
            self.manifest["warnings"].append(f"Optional JSON not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_hhmm_to_min(self, s: str):
        s = (s or "").strip()
        try:
            hh, mm = s.split(":")
            return int(hh) * 60 + int(mm)
        except Exception:
            return None

    def _hours_between(self, start_str, end_str):
        s = self._parse_hhmm_to_min(start_str)
        e = self._parse_hhmm_to_min(end_str)
        if s is None or e is None or e <= s:
            return None
        return round((e - s) / 60.0, 2)

    def _date_str(self, label):
        return str(label).strip()

    def _build_emp_types(self):
        emp_types = {}
        for _, r in self.avail_df.iterrows():
            eid = str(r.get("employee_id", "")) or str(r.get("employee", ""))
            if eid not in emp_types:
                emp_types[eid] = str(r.get("type", "")).lower()
        return emp_types

    def _derive_weekly_bands(self):
        bands = {
            "full-time": (35.0, 45.0),
            "part-time": (20.0, 32.0),
            "casual": (8.0, 24.0),
        }

        basic = self.params.get("basic", {})
        monthly_std = basic.get("Monthly Standard Required Hours", {})
        try:
            monthly_val = float(monthly_std.get("value", 152))
        except Exception:
            monthly_val = 152.0

        weekly_std = monthly_val / 4.33
        bands["full-time"] = (weekly_std * 0.9, weekly_std * 1.2)

        emp_bands = {}
        for emp, typ in self.emp_types.items():
            if "full" in typ:
                lo, hi = bands["full-time"]
            elif "part" in typ:
                lo, hi = bands["part-time"]
            elif "casual" in typ:
                lo, hi = bands["casual"]
            else:
                continue
            emp_bands[emp] = (lo, hi)

        return emp_bands

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

    # ---------------------------------------------------------
    # Load artifacts and parameters
    # ---------------------------------------------------------
    def load(self):
        self.avail_df = self._load_csv(self.availability_path, required=True)
        self.shift_df = self._load_csv(self.shift_codes_path, required=True)
        self.params = self._load_json(self.params_path, required=True)
        self.staffing = self._load_json(self.staffing_path, required=False)
        self.store_config = self._load_json(self.store_config_path, required=False)

        for df in [self.avail_df, self.shift_df]:
            df.columns = [str(c).strip() for c in df.columns]

        # Ensure employee_id exists
        if "employee_id" not in self.avail_df.columns and "employee" in self.avail_df.columns:
            self.avail_df["employee_id"] = (
                self.avail_df["employee"].astype("category").cat.codes + 1000
            )

        # Drop legend / metadata rows (non-numeric IDs)
        if "employee_id" in self.avail_df.columns:
            mask_numeric = self.avail_df["employee_id"].astype(str).str.isnumeric()
            filtered = self.avail_df.loc[mask_numeric].copy()
            self.avail_df = pd.DataFrame(filtered)

        # Service periods from parameters
        self.service_periods = []
        for r in self.params.get("service_periods", []):
            if r.get("period") and r.get("start") and r.get("end"):
                self.service_periods.append(
                    {"name": r["period"], "start": r["start"], "end": r["end"]}
                )

        # Max consecutive days (parameter-driven)
        basic = self.params.get("basic", {})
        max_consec = basic.get("Maximum Consecutive Working Days", {})
        try:
            self.MAX_CONSEC_DAYS = int(float(max_consec.get("value", self.MAX_CONSEC_DAYS)))
        except Exception:
            pass

        # Staffing targets from new nested structure
        self.staff_targets = {}
        structured = self.staffing.get("structured", {})

        if isinstance(structured, dict) and structured:
            store_ids = list(structured.keys())
            self.primary_store_id = store_ids[0]
            self.manifest["primary_store_id"] = self.primary_store_id

            per_store = structured.get(self.primary_store_id, {})
            if isinstance(per_store, dict):
                for period_type, row in per_store.items():
                    if isinstance(row, dict):
                        self.staff_targets[(self.primary_store_id, period_type)] = row
        elif isinstance(structured, list):
            for row in structured:
                if not isinstance(row, dict):
                    continue
                store_id = row.get("store_id", "Store_1")
                period_type = row.get("period_type", "Normal")
                self.primary_store_id = self.primary_store_id or store_id
                self.staff_targets[(store_id, period_type)] = row

        if not self.primary_store_id:
            self.primary_store_id = "Store_1"
            self.manifest["warnings"].append(
                "No staffing structure found; defaulting primary_store_id to 'Store_1'."
            )

        # Store operating hours from store_config (if available)
        self.store_open_min = None
        self.store_close_min = None
        structured_cfg = self.store_config.get("structured", {})
        if isinstance(structured_cfg, dict) and structured_cfg:
            store_cfg = structured_cfg.get(self.primary_store_id) or next(
                iter(structured_cfg.values()), {}
            )
            # Try several likely keys
            open_str = store_cfg.get("opening_time") or store_cfg.get("Opening Time")
            close_str = store_cfg.get("closing_time") or store_cfg.get("Closing Time")
            if open_str and close_str:
                self.store_open_min = self._parse_hhmm_to_min(str(open_str))
                self.store_close_min = self._parse_hhmm_to_min(str(close_str))
                if self.store_open_min is None or self.store_close_min is None:
                    self.store_open_min = None
                    self.store_close_min = None
                    self.manifest["warnings"].append(
                        "Failed to parse store opening/closing times; store-hour constraint disabled."
                    )
        else:
            self.manifest["warnings"].append(
                "No structured store_config found; store-hour constraint disabled."
            )

        self.emp_types = self._build_emp_types()
        self.weekly_bands = self._derive_weekly_bands()
        self.manifest["weekly_bands"] = self.weekly_bands

    # ---------------------------------------------------------
    # Build decision variables
    # ---------------------------------------------------------
    def build_vars(self):
        dedup_cols = [
            "employee_id",
            "employee",
            "type",
            "station",
            "date",
            "token",
            "start_time",
            "end_time",
            "hours",
        ]
        avail_subset: pd.DataFrame = self.avail_df.filter(items=dedup_cols)
        avail_clean: pd.DataFrame = (
            avail_subset.copy().drop_duplicates().reset_index(drop=True)
        )

        for _, r in avail_clean.iterrows():
            emp_id = r.get("employee_id", None)
            if emp_id is None:
                continue
            emp = str(emp_id)

            day = self._date_str(r.get("date"))
            date_parsed = r.get("date_parsed", "")
            token = str(r.get("token")).strip()
            station = str(r.get("station")).strip()

            if token in ["", "/"]:
                continue

            start_t = str(r.get("start_time", "")).strip()
            end_t = str(r.get("end_time", "")).strip()
            h_val = r.get("hours", "")

            try:
                hours = float(str(h_val or 0))
            except Exception:
                hours = self._hours_between(start_t, end_t) or 0.0

            if hours < self.MIN_SHIFT_HOURS or hours > self.MAX_SHIFT_HOURS:
                continue

            smin = self._parse_hhmm_to_min(start_t)
            emin = self._parse_hhmm_to_min(end_t)
            if smin is None or emin is None or emin <= smin:
                continue

            # Store operating hours constraint (filter candidates)
            if self.store_open_min is not None and self.store_close_min is not None:
                if smin < self.store_open_min or emin > self.store_close_min:
                    continue

            key = (emp, day, token, station)
            if key in self.assign:
                continue

            var = self.model.NewBoolVar(f"x_{emp}_{day}_{token}_{station}")
            self.assign[key] = var
            self.hours[key] = int(round(hours * 60))
            self.start_min[key] = smin
            self.end_min[key] = emin
            self.emp_days[emp].add(day)

            if day not in self.day_index and date_parsed:
                try:
                    dt = pd.to_datetime(str(date_parsed), errors="coerce")
                    if pd.notna(dt):
                        self.day_index[day] = dt.date().toordinal()
                except Exception:
                    pass

    # ---------------------------------------------------------
    # Hard constraints
    # ---------------------------------------------------------
    def add_hard_constraints(self):
        # 1) At most one shift per employee per day across all stations
        by_emp_day = defaultdict(list)
        for (emp, day, token, station), v in self.assign.items():
            by_emp_day[(emp, day)].append(v)
        for (emp, day), vars_list in by_emp_day.items():
            self.model.Add(sum(vars_list) <= 1)

        # Build day-level indicator vars for consecutive-days checks
        day_vars = {}
        for (emp, day), vars_list in by_emp_day.items():
            dv = self.model.NewIntVar(0, 1, f"day_{emp}_{day}")
            self.model.Add(dv == sum(vars_list))
            day_vars[(emp, day)] = dv

        # 2) 10-hour rest period between consecutive days
        def day_order_key(day_str):
            return day_str  # assumes lexicographic ordering of date labels

        for emp, days in self.emp_days.items():
            days_sorted = sorted(list(days), key=day_order_key)
            for i in range(len(days_sorted) - 1):
                d1, d2 = days_sorted[i], days_sorted[i + 1]
                for (e1, dd1, t1, st1), v1 in self.assign.items():
                    if e1 != emp or dd1 != d1:
                        continue
                    end1 = self.end_min[(e1, dd1, t1, st1)]
                    for (e2, dd2, t2, st2), v2 in self.assign.items():
                        if e2 != emp or dd2 != d2:
                            continue
                        start2 = self.start_min[(e2, dd2, t2, st2)]
                        self.model.Add(start2 - end1 >= self.MIN_REST_HOURS * 60).OnlyEnforceIf(
                            [v1, v2]
                        )

        # 2b) Max consecutive working days
        if self.MAX_CONSEC_DAYS and self.day_index:
            max_consec = int(self.MAX_CONSEC_DAYS)
            for emp, days in self.emp_days.items():
                days_with_idx = [(d, self.day_index.get(d)) for d in days if d in self.day_index]
                days_with_idx = [(d, idx) for d, idx in days_with_idx if idx is not None]
                if len(days_with_idx) <= max_consec:
                    continue
                days_with_idx = sorted(days_with_idx, key=lambda x: x[1])
                for i in range(len(days_with_idx) - max_consec):
                    window = days_with_idx[i:i + max_consec + 1]
                    if any(window[j][1] != window[0][1] + j for j in range(len(window))):
                        continue
                    window_vars = [day_vars[(emp, d)] for d, _ in window]
                    self.model.Add(sum(window_vars) <= max_consec)

        # 3) Coverage variables per service period, day, and station
        def overlaps(s1, e1, s2, e2):
            return not (e2 <= s1 or e1 <= s2)

        all_days = sorted(set([d for (_, d, _, _) in self.assign.keys()]))
        all_stations = sorted(set([st for (_, _, _, st) in self.assign.keys()]))

        for d in all_days:
            for st in all_stations:
                for p in self.service_periods:
                    p_start = self._parse_hhmm_to_min(p["start"])
                    p_end = self._parse_hhmm_to_min(p["end"])
                    pname = p["name"]

                    cov = self.model.NewIntVar(0, 1000, f"cov_{pname}_{d}_{st}")
                    contributors = []
                    for (emp, day, token, station), v in self.assign.items():
                        if day != d or station != st:
                            continue
                        smin = self.start_min[(emp, day, token, station)]
                        emin = self.end_min[(emp, day, token, station)]
                        if overlaps(p_start, p_end, smin, emin):
                            contributors.append(v)
                    if contributors:
                        self.model.Add(cov == sum(contributors))
                    else:
                        self.model.Add(cov == 0)
                    self.coverage_vars[(pname, d, st)] = cov
                    self.coverage_caps[(pname, d, st)] = len(contributors)

        # 4) Weekly upper bounds per employee
        emp_total_minutes = defaultdict(list)
        for (emp, day, token, station), v in self.assign.items():
            emp_total_minutes[emp].append((v, self.hours[(emp, day, token, station)]))

        for emp, items in emp_total_minutes.items():
            if emp not in self.weekly_bands:
                continue
            lo, hi = self.weekly_bands[emp]
            hi_min = int(hi * 60)
            total_min = self.model.NewIntVar(0, 100000, f"total_min_{emp}")
            self.model.Add(total_min == sum(v * m for (v, m) in items))
            self.model.Add(total_min <= hi_min)

        # 5) Hard minimum coverage for staffed periods (Peak)
        store_id = self.primary_store_id
        for (pname, d, st), cov in self.coverage_vars.items():
            period_type = self._period_type_for_service(pname)
            key = (store_id, period_type)
            row = self.staff_targets.get(key)
            if not row:
                continue
            field = self._staff_field_for_station(st)
            try:
                required = float(row.get(field, 0.0))
            except Exception:
                required = 0.0

            # Only enforce hard minimum for Peak; Normal is handled via objective penalties
            if period_type == "Peak" and required > 0 and self.enforce_peak_minimums:
                cap = self.coverage_caps.get((pname, d, st), 0)
                if int(required) > cap:
                    self.manifest["warnings"].append(
                        f"Peak minimum {required} exceeds possible coverage ({cap}) for {pname} {d} {st}; "
                        "skipping hard constraint."
                    )
                else:
                    self.model.Add(cov >= int(required))

    # ---------------------------------------------------------
    # Objective: coverage + fairness + weekly lower + staffing alignment
    # ---------------------------------------------------------
    def add_objective(self):
        terms = []

        w_cov = self.manifest["objective_weights"]["coverage"]
        w_fair = self.manifest["objective_weights"]["fairness"]
        w_assign = self.manifest["objective_weights"]["assignment"]
        w_weekly = self.manifest["objective_weights"]["weekly_lower"]
        w_staff = self.manifest["objective_weights"]["staffing_alignment"]

        # 1) Coverage reward
        for (_, cov) in self.coverage_vars.items():
            terms.append(w_cov * cov)

        # 2) Fairness: reward total assignments per employee
        by_emp = defaultdict(list)
        for (emp, day, token, station), v in self.assign.items():
            by_emp[emp].append(v)
        for emp, vs in by_emp.items():
            terms.append(int(round(w_fair)) * sum(vs))

        # 3) Base reward per assignment
        for (_, v) in self.assign.items():
            terms.append(w_assign * v)

        # 4) Soft weekly lower bounds (encourage minimum hours)
        emp_minutes = defaultdict(list)
        for (emp, day, token, station), v in self.assign.items():
            emp_minutes[emp].append((v, self.hours[(emp, day, token, station)]))

        weekly_under_penalties = []
        for emp, items in emp_minutes.items():
            if emp not in self.weekly_bands:
                continue
            lo, hi = self.weekly_bands[emp]
            lo_min = int(lo * 60)
            total = sum(v * m for (v, m) in items)

            under = self.model.NewIntVar(0, 100000, f"under_{emp}")
            self.model.Add(under >= lo_min - total)
            weekly_under_penalties.append(under)

        if weekly_under_penalties:
            terms.append(-w_weekly * sum(weekly_under_penalties))

        # 5) Staffing alignment: penalize shortfall vs staffing_requirements
        staffing_shortfall_penalties = []
        store_id = self.primary_store_id

        for (pname, d, st), cov in self.coverage_vars.items():
            period_type = self._period_type_for_service(pname)
            key = (store_id, period_type)
            row = self.staff_targets.get(key)
            if not row:
                continue

            field = self._staff_field_for_station(st)
            try:
                required = float(row.get(field, 0.0))
            except Exception:
                required = 0.0

            if required <= 0:
                continue

            short = self.model.NewIntVar(0, 1000, f"short_{pname}_{d}_{st}")
            self.model.Add(short >= int(required) - cov)
            staffing_shortfall_penalties.append(short)

        if staffing_shortfall_penalties:
            terms.append(-w_staff * sum(staffing_shortfall_penalties))

        self.model.Maximize(sum(terms))

    # ---------------------------------------------------------
    # Solve and export
    # ---------------------------------------------------------
    def solve(self):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.time_limit_sec)
        solver.parameters.num_search_workers = int(self.num_workers)

        status = solver.Solve(self.model)
        self.manifest["solver_status"] = solver.StatusName(status)

        # IMPORTANT: CpSolver.BooleanValue/Value are only valid when a solution exists.
        # If the model is infeasible/invalid, avoid exporting garbage values.
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Total assignment variables:", len(self.assign))
            print("Selected assignments:", 0)

            out_path = os.path.join(self.out_dir, "roster_solution.csv")
            empty_cols = [
                "employee_id",
                "employee",
                "type",
                "station",
                "date",
                "code",
                "start_time",
                "end_time",
                "hours",
            ]
            pd.DataFrame().reindex(columns=empty_cols).to_csv(out_path, index=False)
            self.manifest["outputs"]["roster_solution"] = out_path
            self.manifest["coverage_summary"] = []
            self.manifest["summary"] = {
                "rows": 0,
                "employees_scheduled": 0,
                "total_hours": 0.0,
            }
            self.manifest["warnings"].append(
                "Solver did not find a feasible solution; roster_solution.csv is empty."
            )

            man_path = os.path.join(self.out_dir, "solver_manifest.json")
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)

            print("Wrote roster_solution.csv ->", out_path)
            print("Wrote solver_manifest.json ->", man_path)
            print("Status:", self.manifest["solver_status"])
            return status

        selected = [(k, v) for k, v in self.assign.items() if solver.BooleanValue(v)]
        print("Total assignment variables:", len(self.assign))
        print("Selected assignments:", len(selected))

        sol_rows = []
        for (emp, day, token, station), v in self.assign.items():
            if not solver.BooleanValue(v):
                continue

            mask = (
                (self.avail_df["employee_id"].astype(str) == emp)
                & (self.avail_df["date"].astype(str) == day)
                & (self.avail_df["token"].astype(str) == token)
                & (self.avail_df["station"].astype(str) == station)
            )
            row_df: pd.DataFrame = self.avail_df.loc[mask]

            if row_df.empty:
                mask2 = (
                    (self.avail_df["employee_id"].astype(str) == emp)
                    & (self.avail_df["date"].astype(str) == day)
                    & (self.avail_df["token"].astype(str) == token)
                )
                row_df = self.avail_df.loc[mask2]
                if row_df.empty:
                    continue

            row = row_df.iloc[0]

            sol_rows.append(
                {
                    "employee_id": row.get("employee_id"),
                    "employee": row.get("employee"),
                    "type": row.get("type"),
                    "station": row.get("station"),
                    "date": day,
                    "code": token,
                    "start_time": row.get("start_time", ""),
                    "end_time": row.get("end_time", ""),
                    "hours": round(self.hours[(emp, day, token, station)] / 60.0, 2),
                }
            )

        out_path = os.path.join(self.out_dir, "roster_solution.csv")
        pd.DataFrame(sol_rows).to_csv(out_path, index=False)
        self.manifest["outputs"]["roster_solution"] = out_path

        cov_summary = []
        for ((pname, d, st), cov) in self.coverage_vars.items():
            cov_summary.append(
                {
                    "period": pname,
                    "day": d,
                    "station": st,
                    "coverage": int(solver.Value(cov)),
                }
            )
        self.manifest["coverage_summary"] = cov_summary

        # Aggregate some high-level stats for dashboard/manifest
        df_sol = pd.DataFrame(sol_rows)
        if not df_sol.empty:
            self.manifest["summary"] = {
                "rows": len(df_sol),
                "employees_scheduled": df_sol["employee_id"].nunique(),
                "total_hours": float(df_sol["hours"].sum()),
            }
        else:
            self.manifest["summary"] = {
                "rows": 0,
                "employees_scheduled": 0,
                "total_hours": 0.0,
            }

        man_path = os.path.join(self.out_dir, "solver_manifest.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

        print("Wrote roster_solution.csv ->", out_path)
        print("Wrote solver_manifest.json ->", man_path)
        print("Status:", self.manifest["solver_status"])
        return status

    # ---------------------------------------------------------
    # End-to-end
    # ---------------------------------------------------------
    def run(self):
        print("Running SolverAgent (CP-SAT, ingestion-aligned, single-store, real-world)...")
        self.load()
        self.build_vars()
        self.add_hard_constraints()
        self.add_objective()
        return self.solve()
