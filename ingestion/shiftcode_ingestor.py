# ingestion/shiftcode_ingestor.py
# Clean, validated, production-grade shift code ingestion

from typing import Dict, Any, Optional, List
import pandas as pd
import re

from .common_utils import (
    sanitize_headers,
    drop_junk_rows,
    is_shift_token,
    safe_time,
    time_to_minutes,
    validate_time_range,
)


class ShiftcodeIngestor:
    def __init__(
        self,
        shift_file_pattern: str = "management_roster",
        codes_sheet_name: Optional[str] = None,
        out_dir: str = "data/artifacts",
    ):
        self.shift_file_pattern = shift_file_pattern
        self.codes_sheet_name = codes_sheet_name
        self.out_dir = out_dir

    # ---------------------------------------------------------
    # FIND SHIFT CODE TABLE
    # ---------------------------------------------------------
    def find_shift_codes_table(self, ingested: Dict[str, Any]) -> pd.DataFrame:
        """
        Locate the shift code table by scanning for:
        - filename containing shift_file_pattern
        - sheet containing both 'code' and 'time' columns
        """

        # Preferred: explicit management roster file
        for fname, sheets in ingested.items():
            if self.shift_file_pattern.lower() not in fname.lower():
                continue
            if not isinstance(sheets, dict):
                continue

            for sname, obj in sheets.items():
                tbl = obj.get("table")
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    cols = [str(c).lower() for c in tbl.columns]
                    if any("code" in c for c in cols) and any("time" in c for c in cols):
                        return drop_junk_rows(tbl.copy())

        # Fallback: any sheet with code/time columns
        for fname, sheets in ingested.items():
            if not isinstance(sheets, dict):
                continue

            for sname, obj in sheets.items():
                tbl = obj.get("table")
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    cols = [str(c).lower() for c in tbl.columns]
                    if any("code" in c for c in cols) and any("time" in c for c in cols):
                        return drop_junk_rows(tbl.copy())

        raise RuntimeError("Shift codes table not found")

    # ---------------------------------------------------------
    # SPLIT REAL SHIFT CODES VS METADATA
    # ---------------------------------------------------------
    def split_codes_vs_metadata(self, df: pd.DataFrame):
        """
        Real shift-code rows:
        - code matches known patterns (1F, 2F, 3F, SC, M, /, NA)
        - OR code is alphanumeric without spaces
        """

        df = df.copy()

        def is_shift_row(row):
            code = str(row.iloc[0]).strip().upper()

            # Skip empty or junk
            if code in ("", "NAN", "NONE", "NOT AVAILABLE"):
                return False

            # Skip bullet points, key points, notes
            if code.startswith(("•", "-", "KEY", "POINT", "AT LEAST")):
                return False

            # Use shared helper
            if is_shift_token(code):
                return True

            return False

        mask = df.apply(is_shift_row, axis=1)

        codes_df = df[mask].reset_index(drop=True)
        metadata_df = df[~mask].reset_index(drop=True)

        return codes_df, metadata_df

    # ---------------------------------------------------------
    # INTERNAL: PARSE TIME STRINGS
    # ---------------------------------------------------------
    def _parse_time_range(self, t: str):
        """
        Parse time strings into (start, end, kind)
        Handles:
        - "06:30 - 15:00"
        - "06:30–15:00"
        - "06:30 to 15:00"
        - "Varies"
        - "-"
        - ""
        """

        if not isinstance(t, str):
            return None, None, "empty"

        t = t.strip()

        if t in ("", "-", "\\-"):
            return None, None, "dash"

        if t.lower().startswith("varies"):
            return None, None, "varies"

        # Normalize separators
        t = t.replace("–", "-")

        # Pattern: HH:MM - HH:MM
        match = re.match(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", t)
        if match:
            return match.group(1), match.group(2), "range"

        # Pattern: HH:MM to HH:MM
        match = re.match(r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})", t, flags=re.I)
        if match:
            return match.group(1), match.group(2), "range"

        # Single time (rare)
        if re.match(r"\d{1,2}:\d{2}", t):
            return t, None, "single"

        return None, None, "unparsed"

    # ---------------------------------------------------------
    # CLEAN + PARSE SHIFT CODES
    # ---------------------------------------------------------
    def clean_and_parse(
        self,
        raw_codes: pd.DataFrame,
        out_name: str = "shift_codes_cleaned.csv",
        meta_name: str = "shiftcode_metadata.csv",
    ) -> pd.DataFrame:

        df = raw_codes.copy()
        df.columns = sanitize_headers(df.columns.tolist())
        df = drop_junk_rows(df)

        # Detect primary columns safely
        code_cols = [c for c in df.columns if "code" in c.lower()]
        time_cols = [c for c in df.columns if "time" in c.lower()]

        if not code_cols or not time_cols:
            raise RuntimeError("Shift code table missing required columns")

        code_col = code_cols[0]
        time_col = time_cols[0]

        # Split into real shift codes vs metadata
        codes_df, metadata_df = self.split_codes_vs_metadata(df)

        # Save metadata
        meta_path = f"{self.out_dir}/{meta_name}"
        metadata_df.to_csv(meta_path, index=False)

        # Normalize code column
        codes_df[code_col] = (
            codes_df[code_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": ""})
        )

        # Parse time column
        parsed = codes_df[time_col].astype(str).map(self._parse_time_range)
        starts, ends, kinds = zip(*parsed)

        codes_df = codes_df.assign(
            start_time=list(starts),
            end_time=list(ends),
            time_kind=list(kinds),
        )

        # Normalize column names
        cols = list(codes_df.columns)
        if len(cols) >= 2:
            codes_df = codes_df.rename(columns={cols[0]: "code", cols[1]: "time"})

        # Special case: "/" means day off
        codes_df.loc[codes_df["code"] == "/", ["start_time", "end_time", "time_kind"]] = [
            None,
            None,
            "day_off",
        ]

        # Special case: "NA" means not available
        codes_df.loc[codes_df["code"] == "NA", ["start_time", "end_time", "time_kind"]] = [
            None,
            None,
            "not_available",
        ]

        # Drop empty codes
        codes_df = codes_df[codes_df["code"].astype(str).str.strip() != ""].copy()

        # ORDER COLUMNS
        cols_out = ["code", "time", "time_kind", "start_time", "end_time"]
        cols_out += [c for c in codes_df.columns if c not in cols_out]
        codes_df = codes_df[cols_out]

        # Save cleaned shift codes
        out_path = f"{self.out_dir}/{out_name}"
        codes_df.to_csv(out_path, index=False)

        return codes_df
