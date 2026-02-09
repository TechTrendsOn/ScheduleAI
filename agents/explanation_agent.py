# agents/explanation_agent.py

import json
from typing import Dict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline


class ExplanationAgent:
    def __init__(
        self,
        compliance_path: str,
        manifest_path: str,
        rules_path: str = "",
        llm_model_name: str = "google/flan-t5-small",
    ):
        with open(compliance_path, "r", encoding="utf-8") as f:
            self.compliance_report = json.load(f)

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.final_manifest = json.load(f)

        self.rules = {}
        if rules_path:
            with open(rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        self.prompt_template = (
            "You are an ExplanationAgent for a restaurant rostering system.\n"
            "Write a concise, human-readable explanation in 4–6 sentences.\n"
            "Do not repeat raw JSON. Summarize key outcomes only.\n\n"
            "Compliance Summary (JSON):\n{compliance}\n\n"
            "Final Roster Summary (JSON):\n{manifest}\n\n"
            "Compliance Rules (JSON):\n{rules}\n\n"
            "Explain:\n"
            "- main violation types and counts\n"
            "- whether swaps were applied, pending, or rejected\n"
            "- what fixes were applied (e.g., breaks)\n"
            "- any key caveats"
        )

    def generate_explanation(self, out_json: str = "explanation_report.json") -> Dict:
        def _truncate(s: str, max_chars: int = 2000) -> str:
            if len(s) <= max_chars:
                return s
            return s[:max_chars].rsplit(" ", 1)[0] + "..."

        def _is_junk(text: str) -> bool:
            if not text:
                return True
            stripped = text.strip()
            if not stripped:
                return True
            junk_chars = ",.;:|/\\-_'`~"
            for ch in junk_chars:
                stripped = stripped.replace(ch, "")
            stripped = "".join(ch for ch in stripped if not ch.isspace())
            return len(stripped) < 12

        compliance_summary = {
            "summary": self.compliance_report.get("summary", {}),
            "violation_types": self.compliance_report.get("summary", {}).get("violation_types", {}),
        }
        manifest_summary = {
            "final_roster_rows": self.final_manifest.get("final_roster_rows"),
            "summary": self.final_manifest.get("summary", {}),
            "swaps_applied": self.final_manifest.get("summary", {}).get("swaps_applied"),
            "swaps_rejected": self.final_manifest.get("summary", {}).get("swaps_rejected"),
            "swaps_pending": self.final_manifest.get("summary", {}).get("swaps_pending"),
        }

        prompt = self.prompt_template.format(
            compliance=_truncate(json.dumps(compliance_summary, indent=2)),
            manifest=_truncate(json.dumps(manifest_summary, indent=2)),
            rules=_truncate(json.dumps(self.rules, indent=2)),
        )

        explanation_text = ""
        try:
            out = self.llm.invoke(prompt)
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                explanation_text = out[0].get("generated_text", "").strip()
            else:
                explanation_text = str(out).strip()
        except Exception:
            explanation_text = ""

        if _is_junk(explanation_text):
            explanation_text = self._fallback_summary()
        elif len(explanation_text) > 800:
            explanation_text = explanation_text[:800].rsplit(" ", 1)[0] + "..."

        explanation = {
            "compliance_file_used": self.final_manifest.get("compliance_used"),
            "explanation_text": explanation_text,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2)

        return explanation

    def human_report(self) -> str:
        return self.generate_explanation()["explanation_text"]

    def _fallback_summary(self) -> str:
        summary = self.compliance_report.get("summary", {})
        vtypes = summary.get("violation_types", {})
        swaps = self.final_manifest.get("summary", {})

        parts = []
        if vtypes:
            vlist = ", ".join(f"{k}: {v}" for k, v in vtypes.items())
            parts.append(f"Violations detected: {vlist}.")
        else:
            parts.append("No violation type breakdown was found in the compliance summary.")

        swaps_applied = swaps.get("swaps_applied", 0)
        swaps_rejected = swaps.get("swaps_rejected", 0)
        swaps_pending = swaps.get("swaps_pending", 0)
        parts.append(
            f"Swaps applied: {swaps_applied}, rejected: {swaps_rejected}, pending: {swaps_pending}."
        )

        final_rows = self.final_manifest.get("final_roster_rows")
        if final_rows is not None:
            parts.append(f"Final roster has {final_rows} rows (including breaks if present).")

        note = swaps.get("note")
        if note:
            parts.append(note)

        return " ".join(parts)
