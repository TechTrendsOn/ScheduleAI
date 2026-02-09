# pyright: ignore
# agents/controller_agent.py

from agents.orchestration_tools import TOOL_REGISTRY


class ControllerAgent:
    """
    ControllerAgent:
      - Classifies user intent (ingestion, solver, compliance, swaps, final roster)
      - Builds a plan (list of tool names)
      - Executes tools in order
      - Returns a unified pipeline state for Streamlit or CLI
    """

    # ---------------------------------------------------------
    # Intent classification
    # ---------------------------------------------------------
    def classify_intent(self, user_task: str, mode: str = ""):
        task = (user_task or "").lower().strip()

        # 1. Ingestion
        if any(k in task for k in ["ingest", "load", "import"]):
            return "run_ingestion"

        # 2. Validation
        if "validate" in task and "ingestion" in task:
            return "run_validate_ingestion"

        # 3. Solver (draft roster)
        if "draft" in task or "solve" in task:
            return "run_solver"

        # "roster" alone → solver, but "final roster" → final
        if "roster" in task and "final" not in task:
            return "run_solver"

        # 4. Compliance
        if "compliance" in task or "check" in task:
            return "run_compliance"

        # 5. Swaps
        if "swap" in task or "fix" in task:
            return "run_swaps"

        # 6. Final roster
        if "final" in task or "publish" in task or "merge" in task:
            return "run_final_roster"

        # 7. Knowledge / explanations
        if "knowledge" in task or "rag" in task:
            return "run_knowledge_rag"
        if "explain" in task or "explanation" in task:
            return "run_explanation"

        # Default
        return "run_ingestion"

    # ---------------------------------------------------------
    # Plan builder
    # ---------------------------------------------------------
    def build_plan(self, intent: str):
        return {
            "run_ingestion": ["ingestion"],
            "run_validate_ingestion": ["validate_ingestion"],
            "run_solver": ["solver"],
            "run_compliance": ["compliance"],
            "run_swaps": ["swaps"],
            "run_final_roster": ["final_roster"],
            "run_knowledge_rag": ["knowledge_rag"],
            "run_explanation": ["explanation"],
        }.get(intent, ["ingestion"])

    # ---------------------------------------------------------
    # Execute plan
    # ---------------------------------------------------------
    def execute_plan(self, plan: list):
        pipeline_state = {}

        for step in plan:
            tool_fn = TOOL_REGISTRY.get(step)

            if tool_fn is None:
                pipeline_state[step] = {
                    "error": f"No tool registered for '{step}'"
                }
                continue

            try:
                output = tool_fn()
                pipeline_state[step] = output
            except Exception as e:
                pipeline_state[step] = {"error": str(e)}

        return pipeline_state

    # ---------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------
    def run(self, user_task: str = "", mode: str = ""):
        intent = self.classify_intent(user_task, mode)
        plan = self.build_plan(intent)
        state = self.execute_plan(plan)

        return {
            "intent": intent,
            "plan": plan,
            "state": state,
        }
