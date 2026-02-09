# cross_artifact_validator.py
# Cross-artifact consistency checks for ingestion outputs

import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd


class CrossArtifactValidator:
    """
    Validates consistency across ingestion artifacts:
      - availability_tidy.csv vs shift_codes_cleaned.csv
      - store_config.json vs staffing_requirements.json
      - rostering_parameters.json service_periods sanity
      - management_availability.json vs shift_codes_cleaned.csv (optional)

    Produces a structured report with:
      - errors, warnings, info
      - per-check results
      - health_score and status
    """

    def __init__(self, artifact_dir: str = "data/artifacts"):
        self.artifact_dir = artifact_dir

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "artifact_dir": self.artifact_dir,
            "checks": {},
            "errors": [],
            "warnings": [],
            "info": [],
            "health_score": 0,
            "status": "UNKNOWN",
        }

        # Load all artifacts we might need (best-effort)
        artifacts = self._load_all_artifacts(report)

        # If critical artifacts are missing, bail early
        shift_codes = artifacts.get("shift_codes")
        availability = artifacts.get("availability")
        if shift_codes is None or availability is None:
            report["errors"].append(
                "Critical artifacts missing: need both shift_codes_cleaned.csv and availability_tidy.csv"
            )
            report["status"] = "FAIL"
            report["health_score"] = 0
            return report

        # Run checks
        checks = {}

        checks["availability_vs_shiftcodes"] = self._check_availability_vs_shiftcodes(
            artifacts["availability"], artifacts["shift_codes"], report
        )

        if artifacts.get("store_config") and artifacts.get("staffing"):
            checks["store_ids_consistency"] = self._check_store_ids_consistency(
                artifacts["store_config"], artifacts["staffing"], report
            )

        if artifacts.get("parameters"):
            checks["service_periods_sanity"] = self._check_service_periods_sanity(
                artifacts["parameters"], report
            )

        if artifacts.get("management"):
            checks["management_vs_shiftcodes"] = self._check_management_vs_shiftcodes(
                artifacts["management"], artifacts["shift_codes"], report
            )

        report["checks"] = checks

        # Derive health score
        report["health_score"], report["status"] = self._derive_health_score(report)

        return report

    # ---------------------------------------------------------
    # Artifact loading
    # ---------------------------------------------------------
    def _csv_path(self, name: str) -> str:
        return os.path.join(self.artifact_dir, name)

    def _json_path(self, name: str) -> str:
        return os.path.join(self.artifact_dir, name)

    def _load_csv(self, name: str, report: Dict[str, Any]) -> Optional[pd.DataFrame]:
        path = self._csv_path(name)
        if not os.path.exists(path):
            report["warnings"].append(f"CSV artifact missing: {name}")
            return None
        try:
            df = pd.read_csv(path)
            if df.empty:
                report["warnings"].append(f"CSV artifact is empty: {name}")
            return df
        except Exception as e:
            report["errors"].append(f"Failed to load CSV {name}: {e}")
            return None

    def _load_json(self, name: str, report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = self._json_path(name)
        if not os.path.exists(path):
            report["warnings"].append(f"JSON artifact missing: {name}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                report["warnings"].append(f"JSON artifact is empty: {name}")
            return data
        except Exception as e:
            report["errors"].append(f"Failed to load JSON {name}: {e}")
            return None

    def _load_all_artifacts(self, report: Dict[str, Any]) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}

        artifacts["shift_codes"] = self._load_csv("shift_codes_cleaned.csv", report)
        artifacts["availability"] = self._load_csv("availability_tidy.csv", report)
        artifacts["parameters"] = self._load_json("rostering_parameters.json", report)
        artifacts["store_config"] = self._load_json("store_config.json", report)
        artifacts["task_demand"] = self._load_json("task_demand.json", report)
        artifacts["staffing"] = self._load_json("staffing_requirements.json", report)
        artifacts["management"] = self._load_json("management_availability.json", report)

        return artifacts

    # ---------------------------------------------------------
    # Check 1: availability_tidy.csv vs shift_codes_cleaned.csv
    # ---------------------------------------------------------
    def _check_availability_vs_shiftcodes(
        self,
        availability_df: pd.DataFrame,
        shift_df: pd.DataFrame,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        check = {
            "name": "availability_vs_shiftcodes",
            "status": "UNKNOWN",
            "unmapped_tokens": [],
            "token_counts": {},
        }

        if availability_df is None or shift_df is None:
            check["status"] = "SKIPPED"
            return check

        # Normalize tokens
        known_codes = set(shift_df["code"].astype(str).str.strip())
        ignored_tokens = {"/", "NA"}
        tokens = availability_df["token"].astype(str).str.strip()

        token_counts = tokens.value_counts().to_dict()
        check["token_counts"] = token_counts

        # Ignore special/non-shift tokens
        ignore = {"", "nan", "NA", "/", "NaN", "None"}

        unmapped = sorted(
            tok for tok in tokens.unique()
            if tok not in known_codes and tok not in ignore
        )
        check["unmapped_tokens"] = unmapped

        if unmapped:
            msg = f"{len(unmapped)} availability tokens do not map to known shift codes: {unmapped}"
            report["warnings"].append(msg)
            check["status"] = "WARN"
        else:
            report["info"].append("All availability tokens map to known shift codes.")
            check["status"] = "OK"

        return check

    # ---------------------------------------------------------
    # Check 2: store IDs across store_config.json & staffing_requirements.json
    # ---------------------------------------------------------
    def _check_store_ids_consistency(
        self,
        store_cfg: Dict[str, Any],
        staffing: Dict[str, Any],
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        check = {
            "name": "store_ids_consistency",
            "status": "UNKNOWN",
            "store_ids_in_config": [],
            "store_ids_in_staffing": [],
            "missing_in_staffing": [],
            "missing_in_config": [],
        }

        structured_cfg = store_cfg.get("structured", {})
        structured_staff = staffing.get("structured", {})

        cfg_ids = set(structured_cfg.keys())
        staff_ids = set(structured_staff.keys())

        check["store_ids_in_config"] = sorted(cfg_ids)
        check["store_ids_in_staffing"] = sorted(staff_ids)

        missing_in_staffing = sorted(cfg_ids - staff_ids)
        missing_in_config = sorted(staff_ids - cfg_ids)

        check["missing_in_staffing"] = missing_in_staffing
        check["missing_in_config"] = missing_in_config

        if missing_in_staffing:
            report["warnings"].append(
                f"Stores present in store_config but missing in staffing_requirements: {missing_in_staffing}"
            )

        if missing_in_config:
            report["warnings"].append(
                f"Stores present in staffing_requirements but missing in store_config: {missing_in_config}"
            )

        if not missing_in_staffing and not missing_in_config:
            report["info"].append("Store IDs match between store_config and staffing_requirements.")
            check["status"] = "OK"
        else:
            check["status"] = "WARN"

        return check

    # ---------------------------------------------------------
    # Check 3: service_periods sanity in rostering_parameters.json
    # ---------------------------------------------------------
    def _check_service_periods_sanity(
        self,
        params: Dict[str, Any],
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        check = {
            "name": "service_periods_sanity",
            "status": "UNKNOWN",
            "periods_count": 0,
            "missing_fields": [],
        }

        periods = params.get("service_periods", [])
        check["periods_count"] = len(periods)

        missing_fields = []
        for p in periods:
            for field in ("period", "start", "end"):
                if not p.get(field):
                    missing_fields.append({"period": p, "missing_field": field})

        check["missing_fields"] = missing_fields

        if not periods:
            report["warnings"].append("No service_periods found in rostering_parameters.json")
            check["status"] = "WARN"
        elif missing_fields:
            report["warnings"].append(
                f"Some service_periods are missing required fields: {missing_fields}"
            )
            check["status"] = "WARN"
        else:
            report["info"].append("Service periods present and structurally sound.")
            check["status"] = "OK"

        return check

    # ---------------------------------------------------------
    # Check 4: management_availability vs shift_codes
    # ---------------------------------------------------------
    def _check_management_vs_shiftcodes(
        self,
        management: Dict[str, Any],
        shift_df: pd.DataFrame,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        check = {
            "name": "management_vs_shiftcodes",
            "status": "UNKNOWN",
            "unknown_tokens": [],
        }

        if not isinstance(management, dict) or shift_df is None:
            check["status"] = "SKIPPED"
            return check

        known_codes = set(shift_df["code"].astype(str).str.strip())
        ignored_tokens = {"/", "NA", "S"}

        structured = management.get("structured", [])
        unknown_tokens = set()

        for entry in structured:
            shifts = entry.get("shifts", {})
            for date_label, token in shifts.items():
                tok = str(token).strip()
                if not tok or tok.lower() in ("nan", "none"):
                    continue
                if tok not in known_codes and tok not in ignored_tokens:
                    unknown_tokens.add(tok)

        check["unknown_tokens"] = sorted(unknown_tokens)

        if unknown_tokens:
            report["warnings"].append(
                f"Management roster uses unknown shift tokens: {sorted(unknown_tokens)}"
            )
            check["status"] = "WARN"
        else:
            report["info"].append("All management shift tokens correspond to known shift codes.")
            check["status"] = "OK"

        return check

    # ---------------------------------------------------------
    # Health score derivation
    # ---------------------------------------------------------
    def _derive_health_score(self, report: Dict[str, Any]) -> tuple[int, str]:
        """
        Very simple scoring:
          - Any errors  -> FAIL, score 0–40
          - Only warnings -> WARN, score 60–80
          - No errors/warnings -> OK, score 100
        """
        errors = report.get("errors", [])
        warnings = report.get("warnings", [])

        if errors:
            # Scale with number of errors (but keep it simple)
            score = max(0, 40 - 5 * len(errors))
            return score, "FAIL"

        if warnings:
            score = max(60, 90 - 5 * len(warnings))
            return score, "WARN"

        return 100, "OK"


if __name__ == "__main__":
    validator = CrossArtifactValidator(artifact_dir="data/artifacts")
    result = validator.run()
    print(json.dumps(result, indent=2))
