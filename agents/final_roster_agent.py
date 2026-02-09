# pyright: ignore
# agents/final_roster_agent.py
# FinalRosterAgent: applies confirmed swaps and produces the final roster + manifest

import os
import json
from typing import Any, Dict
import pandas as pd
from datetime import datetime


class FinalRosterAgent:
    def __init__(
        self,
        roster_path: str = "data/artifacts/roster_solution.csv",
        compliance_path: str = "data/artifacts/compliance_report.json",
        swaps_path: str = "data/artifacts/swaps_report.json",
        out_dir: str = "data/artifacts",
    ):
        self.roster_path = roster_path
        self.compliance_path = compliance_path
        self.swaps_path = swaps_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Load roster
        if not os.path.exists(roster_path):
            raise FileNotFoundError(f"Roster file not found: {roster_path}")
        self.roster_df = pd.read_csv(roster_path)

        # Load compliance report
        if os.path.exists(compliance_path):
            with open(compliance_path, "r", encoding="utf-8") as f:
                self.compliance_report = json.load(f)
        else:
            self.compliance_report = {"violations": [], "summary": {}, "notes": []}

        # Load swaps
        if os.path.exists(swaps_path):
            with open(swaps_path, "r", encoding="utf-8") as f:
                self.swaps_report = json.load(f)
        else:
            self.swaps_report = {"swaps": []}

        # Tracking
        self.applied_swaps = []
        self.rejected_swaps = []
        self.pending_swaps = []
        self.summary_stats: Dict[str, Any] = {
            "swaps_applied": 0,
            "swaps_rejected": 0,
        }

    # ---------------------------------------------------------
    # Apply confirmed swaps
    # ---------------------------------------------------------
    def _has_shift_on_date(self, employee: str, date: str) -> bool:
        if not employee or not date:
            return False
        return not self.roster_df[
            (self.roster_df["employee"] == employee) & (self.roster_df["date"] == date)
        ].empty

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

    def _insert_auto_breaks(self):
        if self.roster_df.empty:
            return

        def parse_date_label(label: str) -> datetime | None:
            label = (label or "").strip()
            for fmt in ["%Y-%m-%d", "%a %b %d"]:
                try:
                    dt = datetime.strptime(label, fmt)
                    if fmt == "%a %b %d":
                        dt = dt.replace(year=2024)
                    return dt
                except Exception:
                    continue
            try:
                return datetime.fromisoformat(label)
            except Exception:
                return None

        base_df = self.roster_df.copy()
        base_df["is_break"] = base_df.apply(self._is_break_row, axis=1)
        base_df = base_df[base_df["is_break"] != True].copy()

        existing_breaks = self.roster_df[
            self.roster_df.apply(self._is_break_row, axis=1)
        ]

        new_rows = []
        columns = list(self.roster_df.columns)

        for _, row in base_df.iterrows():
            hours = float(row.get("hours") or 0)
            if hours <= 5:
                continue

            start_time = row.get("start_time")
            end_time = row.get("end_time")
            date = row.get("date")
            if not start_time or not end_time or not date:
                continue

            base_dt = parse_date_label(str(date))
            if base_dt is None:
                continue
            try:
                st_h, st_m = [int(x) for x in str(start_time).split(":")]
                en_h, en_m = [int(x) for x in str(end_time).split(":")]
                start_dt = base_dt.replace(hour=st_h, minute=st_m)
                end_dt = base_dt.replace(hour=en_h, minute=en_m)
            except Exception:
                continue

            if hours > 6:
                break_minutes = 30
                break_code = "MEAL_BREAK"
                break_label = "Meal Break"
            else:
                break_minutes = 10
                break_code = "REST_BREAK"
                break_label = "Rest Break"

            if all(col in existing_breaks.columns for col in ["employee", "date"]):
                candidate_breaks = existing_breaks[
                    (existing_breaks["employee"] == row.get("employee"))
                    & (existing_breaks["date"] == date)
                ]
                desired_kind = "meal" if break_code == "MEAL_BREAK" else "rest"
                has_break = any(
                    self._break_kind(b) in {desired_kind, "break"}
                    for _, b in candidate_breaks.iterrows()
                )
            else:
                has_break = False
            if has_break:
                continue

            shift_minutes = int((end_dt - start_dt).total_seconds() / 60)
            if shift_minutes <= break_minutes:
                continue

            break_start = start_dt + pd.Timedelta(minutes=(shift_minutes - break_minutes) / 2)
            break_end = break_start + pd.Timedelta(minutes=break_minutes)

            new_row = {c: None for c in columns}
            for k, v in {
                "employee_id": row.get("employee_id"),
                "employee": row.get("employee"),
                "type": row.get("type"),
                "station": break_label,
                "date": date,
                "code": break_code,
                "start_time": break_start.strftime("%H:%M"),
                "end_time": break_end.strftime("%H:%M"),
                "hours": round(break_minutes / 60.0, 2),
                "notes": "Auto break",
            }.items():
                new_row[k] = v
            new_rows.append(new_row)

        if new_rows:
            self.roster_df = pd.concat([self.roster_df, pd.DataFrame(new_rows)], ignore_index=True)

    def apply_swaps(self):
        swaps = self.swaps_report.get("swaps", [])

        for s in swaps:
            if not s.get("confirmed", False):
                self.pending_swaps.append(s)
                continue

            emp_a = s.get("violation", {}).get("employee")
            emp_b = s.get("suggested_employee")
            date = s.get("date")
            station = s.get("station")

            if not all([emp_b, date, station]):
                self.rejected_swaps.append(s)
                continue

            row_a = self.roster_df[
                (self.roster_df["employee"] == emp_a)
                & (self.roster_df["date"] == date)
                & (self.roster_df["station"] == station)
            ] if emp_a else pd.DataFrame()

            # Only reassign existing shifts; do not add duplicates
            if row_a.empty:
                s["rejected"] = True
                s["reason"] = "No matching shift to reassign"
                self.rejected_swaps.append(s)
                continue

            # Enforce one-shift-per-day: suggested employee must be free that date
            if self._has_shift_on_date(emp_b, date):
                s["rejected"] = True
                s["reason"] = "Suggested employee already scheduled on date"
                self.rejected_swaps.append(s)
                continue

            idx_a = row_a.index[0]
            self.roster_df.at[idx_a, "employee"] = emp_b
            self.roster_df.at[idx_a, "notes"] = "Swap applied"
            self.applied_swaps.append(s)
            self.summary_stats["swaps_applied"] += 1

        self.summary_stats["swaps_rejected"] = len(self.rejected_swaps)
        self.summary_stats["swaps_pending"] = len(self.pending_swaps)

    # ---------------------------------------------------------
    # Save final roster + manifest
    # ---------------------------------------------------------
    def generate_final(
        self,
        out_csv="data/artifacts/final_roster.csv",
        out_json="data/artifacts/final_roster_manifest.json",
    ):

        # Apply swaps before generating final roster
        self.apply_swaps()

        # Auto-insert breaks for compliance
        self._insert_auto_breaks()

        # Add summary note
        summary: Dict[str, Any] = dict(self.summary_stats)
        if self.summary_stats["swaps_applied"] == 0:
            summary["note"] = "No swaps applied — final roster equals draft roster."
        else:
            summary["note"] = "Swaps applied to generate final roster."
        if self.summary_stats.get("swaps_pending"):
            summary["pending_note"] = "Some swaps were not confirmed and were not applied."

        # Save final roster
        self.roster_df.to_csv(out_csv, index=False)

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "final_roster_rows": len(self.roster_df),
            "compliance_used": self.compliance_path,
            "swaps_used": self.swaps_path,
            "summary": summary,
            "applied_swaps": self.applied_swaps,
            "rejected_swaps": self.rejected_swaps,
            "pending_swaps": self.pending_swaps,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "final_roster": self.roster_df,
            "manifest_path": out_json,
            "summary": summary,
            "applied_swaps": self.applied_swaps,
            "rejected_swaps": self.rejected_swaps,
            "pending_swaps": self.pending_swaps,
        }

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------
    def summary_report(self):
        return (
            f"Final Roster Summary:\n"
            f"- Swaps applied: {self.summary_stats['swaps_applied']}\n"
            f"- Swaps rejected: {self.summary_stats['swaps_rejected']}\n"
            f"- Total rows: {len(self.roster_df)}\n"
        )
