 # agents/orchestration_tools.py

import os
import pandas as pd
import json

from ingestion.ingestion_agent import IngestionAgent
from ingestion.cross_artifact_validator import CrossArtifactValidator
from agents.solver_agent import SolverAgent
from agents.compliance_agent import ComplianceAgent
from agents.swap_agent import SwapAgent
from agents.final_roster_agent import FinalRosterAgent


ARTIFACT_DIR = "data/artifacts"


# ---------------------------------------------------------
# 1) INGESTION
# ---------------------------------------------------------
def run_ingestion():
    agent = IngestionAgent(input_dir="data", out_dir=ARTIFACT_DIR)
    manifest = agent.run()
    return {
        "manifest": manifest,
        "status": "ok",
        "artifact_dir": ARTIFACT_DIR,
    }


# ---------------------------------------------------------
# 1b) VALIDATE INGESTION
# ---------------------------------------------------------
def run_validate_ingestion():
    validator = CrossArtifactValidator(ARTIFACT_DIR)
    report = validator.run()
    report_path = os.path.join(ARTIFACT_DIR, "ingestion_health_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {
        "report": report,
        "path": report_path,
    }


# ---------------------------------------------------------
# 2) SOLVER
# ---------------------------------------------------------
def run_solver():
    agent = SolverAgent(
        artifact_dir=ARTIFACT_DIR,
        out_dir=ARTIFACT_DIR,
    )
    status = agent.run()

    sol_path = os.path.join(ARTIFACT_DIR, "roster_solution.csv")
    sol_df = pd.read_csv(sol_path) if os.path.exists(sol_path) else pd.DataFrame()

    man_path = os.path.join(ARTIFACT_DIR, "solver_manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}

    return {
        "solver_status": status,
        "solution": sol_df,
        "solution_dict": sol_df.to_dict(orient="records"),
        "solution_path": sol_path,
        "manifest": manifest,
    }


# ---------------------------------------------------------
# 3) COMPLIANCE
# ---------------------------------------------------------
def run_compliance():
    agent = ComplianceAgent(
        roster_path=os.path.join(ARTIFACT_DIR, "roster_solution.csv"),
        params_path=os.path.join(ARTIFACT_DIR, "rostering_parameters.json"),
        out_dir=ARTIFACT_DIR,
    )
    report = agent.run()

    return {
        "report": report,
        "violations": report.get("violations", []),
        "summary": report.get("summary", {}),
        "notes": report.get("notes", []),
        "path": os.path.join(ARTIFACT_DIR, "compliance_report.json"),
    }


# ---------------------------------------------------------
# 4) SWAPS
# ---------------------------------------------------------
def run_swaps():
    agent = SwapAgent(
        roster_path=os.path.join(ARTIFACT_DIR, "roster_solution.csv"),
        compliance_report_path=os.path.join(ARTIFACT_DIR, "compliance_report.json"),
        params_path=os.path.join(ARTIFACT_DIR, "rostering_parameters.json"),
        out_dir=ARTIFACT_DIR,
    )
    result = agent.run()

    return {
        "swaps": result.get("swaps", []),
        "path": os.path.join(ARTIFACT_DIR, "swaps_report.json"),
    }


# ---------------------------------------------------------
# 5) FINAL ROSTER
# ---------------------------------------------------------
def run_final_roster():
    agent = FinalRosterAgent(
        roster_path=os.path.join(ARTIFACT_DIR, "roster_solution.csv"),
        compliance_path=os.path.join(ARTIFACT_DIR, "compliance_report.json"),
        swaps_path=os.path.join(ARTIFACT_DIR, "swaps_report.json"),
        out_dir=ARTIFACT_DIR,
    )

    result = agent.generate_final(
        out_csv=os.path.join(ARTIFACT_DIR, "final_roster.csv"),
        out_json=os.path.join(ARTIFACT_DIR, "final_roster_manifest.json"),
    )

    final_df = result.get("final_roster")
    summary = result.get("summary", {})

    if final_df is None:
        final_dict = []
    else:
        final_dict = final_df.to_dict(orient="records") if hasattr(final_df, "to_dict") else []
    return {
        "final_roster": final_df,
        "final_roster_dict": final_dict,
        "manifest_path": result.get("manifest_path"),
        "summary": summary,
        "applied_swaps": result.get("applied_swaps", []),
        "rejected_swaps": result.get("rejected_swaps", []),
    }


# ---------------------------------------------------------
# 6) KNOWLEDGE RAG
# ---------------------------------------------------------
def run_knowledge_rag(query: str, artifact_dir: str = ARTIFACT_DIR):
    try:
        from agents.knowledge_agent import KnowledgeAgentRAG
    except Exception as e:
        return {"error": f"KnowledgeAgentRAG unavailable: {e}"}

    agent = KnowledgeAgentRAG(
        artifact_dir=artifact_dir,
        persistence_dir=os.path.join(artifact_dir, "chroma_knowledge"),
    )

    files = [
        {"path": os.path.join(artifact_dir, "rostering_parameters.json"), "type": "json"},
        {"path": os.path.join(artifact_dir, "ingestion_health_report.json"), "type": "json"},
        {"path": os.path.join(artifact_dir, "shift_codes_cleaned.csv"), "type": "csv"},
        {"path": os.path.join(artifact_dir, "staffing_requirements.json"), "type": "json"},
        {"path": os.path.join(artifact_dir, "compliance_report.json"), "type": "json"},
    ]

    agent.ingest_files(files)
    return agent.answer(query)


# ---------------------------------------------------------
# 7) EXPLANATION AGENT
# ---------------------------------------------------------
def run_explanation_agent(artifact_dir: str = ARTIFACT_DIR):
    try:
        from agents.explanation_agent import ExplanationAgent
    except Exception as e:
        return {"error": f"ExplanationAgent unavailable: {e}"}

    comp_path = os.path.join(artifact_dir, "compliance_report.json")
    manifest_path = os.path.join(artifact_dir, "final_roster_manifest.json")
    rules_path = os.path.join(artifact_dir, "rostering_parameters.json")

    agent = ExplanationAgent(
        compliance_path=comp_path,
        manifest_path=manifest_path,
        rules_path=rules_path,
    )

    out_path = os.path.join(artifact_dir, "explanation_report.json")
    return agent.generate_explanation(out_json=out_path)


# ---------------------------------------------------------
# TOOL REGISTRY
# ---------------------------------------------------------
TOOL_REGISTRY = {
    "ingestion": run_ingestion,
    "validate_ingestion": run_validate_ingestion,
    "solver": run_solver,
    "compliance": run_compliance,
    "swaps": run_swaps,
    "final_roster": run_final_roster,
    "knowledge_rag": run_knowledge_rag,
    "explanation": run_explanation_agent,
}
