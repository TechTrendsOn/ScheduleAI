# ingestion/ingestion_agent.py
# Orchestrates all ingestors and writes structured + metadata artifacts.

import os
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from .common_utils import (
    ingest_directory,
    save_pickle,
    save_json,
)
from .availability_ingestor import AvailabilityIngestor
from .shiftcode_ingestor import ShiftcodeIngestor
from .parameters_ingestor import ParametersIngestor
from .store_config_ingestor import StoreConfigIngestor
from .task_demand_ingestor import TaskDemandIngestor
from .staffing_ingestor import StaffingIngestor
from .management_ingestor import ManagementIngestor


class IngestionAgent:
    def __init__(self, input_dir: str = "data", out_dir: str = "data/artifacts"):
        self.input_dir = input_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.manifest: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {"directory": input_dir},
            "outputs": {},
            "warnings": [],
            "notes": [],
        }

    # ---------------------------------------------------------
    # MAIN ORCHESTRATION
    # ---------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        print("\n=== Running IngestionAgent ===")
        print("Looking for input files in:", os.path.abspath(self.input_dir))

        # ---------------------------------------------------------
        # 1) Raw ingestion of all files
        # ---------------------------------------------------------
        ingested = ingest_directory(self.input_dir)

        # Collect sheet-level warnings from common_utils
        for fname, sheets in ingested.items():
            if isinstance(sheets, dict) and "_warnings" in sheets:
                for w in sheets["_warnings"]:
                    self._warn(f"{fname}: {w}")

        ingested_pkl_path = os.path.join(self.out_dir, "ingested.pkl")
        save_pickle(ingested, ingested_pkl_path)
        self.manifest["outputs"]["ingested.pkl"] = ingested_pkl_path

        # ---------------------------------------------------------
        # 2) Shift codes (critical dependency)
        # ---------------------------------------------------------
        shift_ing = ShiftcodeIngestor()
        try:
            raw_codes = shift_ing.find_shift_codes_table(ingested)
            shift_codes = shift_ing.clean_and_parse(raw_codes)
            shift_codes_path = os.path.join(self.out_dir, "shift_codes_cleaned.csv")
            shift_codes.to_csv(shift_codes_path, index=False)
            self.manifest["outputs"]["shift_codes_cleaned"] = shift_codes_path
        except Exception as e:
            self._warn(f"Shift codes error: {e}")
            shift_codes = pd.DataFrame()

        # ---------------------------------------------------------
        # 3) Availability → availability_tidy.csv
        # ---------------------------------------------------------
        avail_ing = AvailabilityIngestor(out_dir=self.out_dir)
        try:
            raw_avail = avail_ing.find_availability_table(ingested)

            if shift_codes.empty:
                raise RuntimeError("Shift codes were not available")

            avail_result = avail_ing.normalize(raw_avail, shift_codes)
            self.manifest["outputs"]["availability_tidy"] = avail_result["path"]

            if avail_result["unmapped_tokens"]:
                self._warn(f"{len(avail_result['unmapped_tokens'])} availability tokens were not mapped")
                self.manifest["notes"].append(
                    {"unmapped_tokens": avail_result["unmapped_tokens"]}
                )

        except Exception as e:
            self._warn(f"Availability normalization error: {e}")

        # ---------------------------------------------------------
        # 4) Parameters → rostering_parameters.json
        # ---------------------------------------------------------
        params_ing = ParametersIngestor()
        try:
            params = params_ing.extract(ingested)
            params_path = os.path.join(self.out_dir, "rostering_parameters.json")
            save_json(params, params_path)
            self.manifest["outputs"]["rostering_parameters"] = params_path

            if params.get("warnings"):
                for w in params["warnings"]:
                    self._warn(f"Parameters: {w}")

        except Exception as e:
            self._warn(f"Parameters error: {e}")

        # ---------------------------------------------------------
        # 5) Store config → store_config.json
        # ---------------------------------------------------------
        store_ing = StoreConfigIngestor()
        try:
            store_cfg = store_ing.extract(ingested)
            store_path = os.path.join(self.out_dir, "store_config.json")
            save_json(store_cfg, store_path)
            self.manifest["outputs"]["store_config"] = store_path

            if store_cfg.get("warnings"):
                for w in store_cfg["warnings"]:
                    self._warn(f"Store config: {w}")

        except Exception as e:
            self._warn(f"Store config error: {e}")

        # ---------------------------------------------------------
        # 6) Task demand → task_demand.json
        # ---------------------------------------------------------
        task_ing = TaskDemandIngestor()
        try:
            task_cfg = task_ing.extract(ingested)
            task_path = os.path.join(self.out_dir, "task_demand.json")
            save_json(task_cfg, task_path)
            self.manifest["outputs"]["task_demand"] = task_path

            if task_cfg.get("warnings"):
                for w in task_cfg["warnings"]:
                    self._warn(f"Task demand: {w}")

        except Exception as e:
            self._warn(f"Task demand error: {e}")

        # ---------------------------------------------------------
        # 7) Staffing estimates → staffing_requirements.json
        # ---------------------------------------------------------
        staff_ing = StaffingIngestor()
        try:
            staff_cfg = staff_ing.extract(ingested)
            staff_path = os.path.join(self.out_dir, "staffing_requirements.json")
            save_json(staff_cfg, staff_path)
            self.manifest["outputs"]["staffing_requirements"] = staff_path

            if staff_cfg.get("warnings"):
                for w in staff_cfg["warnings"]:
                    self._warn(f"Staffing: {w}")

        except Exception as e:
            self._warn(f"Staffing error: {e}")

        # ---------------------------------------------------------
        # 8) Management roster → management_availability.json
        # ---------------------------------------------------------
        mgmt_ing = ManagementIngestor()
        try:
            mgmt_cfg = mgmt_ing.extract(ingested)
            mgmt_path = os.path.join(self.out_dir, "management_availability.json")
            save_json(mgmt_cfg, mgmt_path)
            self.manifest["outputs"]["management_availability"] = mgmt_path

            if mgmt_cfg.get("warnings"):
                for w in mgmt_cfg["warnings"]:
                    self._warn(f"Management roster: {w}")

        except Exception as e:
            self._warn(f"Management error: {e}")

        # ---------------------------------------------------------
        # 9) Write ingestion manifest
        # ---------------------------------------------------------
        manifest_path = os.path.join(self.out_dir, "ingestion_manifest.json")
        save_json(self.manifest, manifest_path)

        print("\n=== Ingestion Complete ===")
        print("Manifest written to:", manifest_path)

        return self.manifest

    # ---------------------------------------------------------
    # INTERNAL: Add warning to manifest
    # ---------------------------------------------------------
    def _warn(self, msg: str):
        print("WARNING:", msg)
        self.manifest["warnings"].append(msg)


if __name__ == "__main__":
    agent = IngestionAgent(input_dir="data", out_dir="data/artifacts")
    agent.run()
