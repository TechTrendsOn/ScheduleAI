import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from agents.controller_agent import ControllerAgent
from agents.orchestration_tools import run_knowledge_rag
from agents.orchestration_tools import run_explanation_agent
from agents.orchestration_tools import run_validate_ingestion
from agents.orchestration_tools import ARTIFACT_DIR
from agents.compliance_agent import ComplianceAgent


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def artifact_path(name: str) -> str:
    return os.path.join(ARTIFACT_DIR, name)


def load_json_if_exists(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_csv_if_exists(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def clear_artifacts_dir(path: str) -> int:
    if not os.path.exists(path):
        return 0
    removed = 0
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            os.remove(full_path)
            removed += 1
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
            removed += 1
    return removed


def render_last_updated(label: str, filename: str):
    path = artifact_path(filename)
    ts = None
    if os.path.exists(path) and filename.endswith(".json"):
        obj = load_json_if_exists(path)
        if isinstance(obj, dict):
            ts = obj.get("timestamp")
    if ts is None and os.path.exists(path):
        ts = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    st.caption(f"{label}: {ts or '—'}")


def init_session_state():
    defaults = {
        "controller": ControllerAgent(),
        "ingestion_result": None,
        "validation_result": None,
        "solver_result": None,
        "compliance_result": None,
        "swaps_result": None,
        "final_result": None,
        "swap_confirm_state": {},
        "last_mode": None,
        "auto_clear_on_mode_switch": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------
# UI Sections
# ---------------------------------------------------------
def section_overview():
    st.title("MAS Rostering Demo Dashboard")

    st.caption("Mode: Standard")
    st.info(
        "Standard mode enforces availability-based swaps and full compliance checks, including coverage targets."
    )
    st.caption("End-to-end roster generation, validation, and finalisation.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Artifacts directory", ARTIFACT_DIR)
    with col2:
        roster_exists = os.path.exists(artifact_path("roster_solution.csv"))
        st.metric("Draft roster generated", "Yes" if roster_exists else "No")
    with col3:
        final_exists = os.path.exists(artifact_path("final_roster.csv"))
        st.metric("Final roster generated", "Yes" if final_exists else "No")

    st.markdown("### Pipeline steps")
    st.markdown(
        """
1. **Ingestion** — Load raw Excel/CSV files.
2. **Validation** — Cross-artifact consistency checks.
3. **Draft roster** — Optimisation via `SolverAgent`.
4. **Compliance** — Validate the draft roster.
5. **Swaps** — Generate and confirm fixes.
6. **Final roster** — Apply swaps and produce audit-ready output.
"""
    )


def section_ingestion():
    st.header("Step 1 — Ingestion")
    render_last_updated("Last updated", "ingestion_manifest.json")

    st.write("Run the ingestion pipeline to read all input files.")

    st.markdown("#### Input files")
    data_dir = Path("data")
    input_files = [p for p in data_dir.iterdir() if p.is_file()]
    if input_files:
        for p in sorted(input_files):
            with open(p, "rb") as f:
                st.download_button(
                    label=f"Download {p.name}",
                    data=f,
                    file_name=p.name,
                    mime="application/octet-stream",
                    key=f"download_input_{p.name}",
                )
    else:
        st.info("No input files found in data/.")

    if st.button("Run ingestion", type="primary"):
        with st.spinner("Running ingestion..."):

            result = st.session_state.controller.run("ingest")
            st.session_state.ingestion_result = result

    result = st.session_state.ingestion_result
    if result is not None:
        state = result.get("state", {})
        ingest_state = state.get("ingestion", {})
        st.success("Ingestion completed.")
        manifest = ingest_state.get("manifest", {})
        st.json(manifest)

        outputs = manifest.get("outputs", {})
        st.markdown("#### Key artifacts")
        for name, path in outputs.items():
            st.write(f"- **{name}** → `{path}`")

        if manifest.get("warnings"):
            with st.expander("Warnings"):
                for w in manifest["warnings"]:
                    st.warning(w)

        if manifest.get("notes"):
            with st.expander("Notes"):
                st.json(manifest["notes"])

    else:
        st.info("Click **Run ingestion** to start.")

    st.markdown("#### Ingestion self-test")
    st.write("Run cross-artifact checks to confirm ingestion outputs are consistent.")
    if st.button("Run ingestion self-test"):
        with st.spinner("Running CrossArtifactValidator..."):
            result = st.session_state.controller.run("validate ingestion")
            st.session_state.validation_result = result
            report = result.get("state", {}).get("validate_ingestion", {}).get("report")
            if not isinstance(report, dict):
                report = {}
            if report:
                report_path = artifact_path("ingestion_health_report.json")
                os.makedirs(ARTIFACT_DIR, exist_ok=True)
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            st.success("Self-test completed.")


def section_validation():
    st.header("Ingestion Validation")
    render_last_updated("Last updated", "ingestion_health_report.json")

    st.write("Run cross‑artifact consistency checks on all ingestion outputs.")

    if st.button("Run validation", type="primary"):
        with st.spinner("Running CrossArtifactValidator..."):
            result = st.session_state.controller.run("validate ingestion")
            st.session_state.validation_result = result

    result = st.session_state.validation_result
    report_path = artifact_path("ingestion_health_report.json")
    report = load_json_if_exists(report_path)
    if not isinstance(report, dict) or not report or not os.path.exists(report_path):
        with st.spinner("Generating validation report..."):
            run_result = run_validate_ingestion()
            report = run_result.get("report") if isinstance(run_result, dict) else None
            st.session_state.validation_result = {
                "state": {"validate_ingestion": run_result}
            }
    if not isinstance(report, dict) or not report:
        st.info("Click **Run validation** to begin.")
        return
    if isinstance(report, dict):
        report_path = artifact_path("ingestion_health_report.json")
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if result:
        state = result.get("state", {})
        val_state = state.get("validate_ingestion", {})
        report = val_state.get("report", report)
        if not isinstance(report, dict):
            report = {}

    if not report:
        report = load_json_if_exists(artifact_path("ingestion_health_report.json")) or {}

    st.subheader("Summary")

    status = report.get("status", "UNKNOWN")
    score = report.get("health_score", 0)

    if status == "OK":
        st.success(f"Health Score: {score} — All checks passed.")
    elif status == "WARN":
        st.warning(f"Health Score: {score} — Some warnings detected.")
    else:
        st.error(f"Health Score: {score} — Errors detected.")

    if report.get("errors"):
        with st.expander("Errors", expanded=True):
            for e in report["errors"]:
                st.error(e)

    if report.get("warnings"):
        with st.expander("Warnings", expanded=False):
            for w in report["warnings"]:
                st.warning(w)

    if report.get("info"):
        with st.expander("Info", expanded=False):
            for i in report["info"]:
                st.info(i)

    st.subheader("Detailed Checks")

    checks = report.get("checks", {})
    for check_name, check_data in checks.items():
        with st.expander(check_name, expanded=False):
            if isinstance(check_data, dict):
                filtered = {}
                for k, v in check_data.items():
                    if k.startswith("missing_") and isinstance(v, list) and len(v) == 0:
                        continue
                    filtered[k] = v
                st.json(filtered)
            else:
                st.json(check_data)

    report_path = artifact_path("ingestion_health_report.json")
    if os.path.exists(report_path):
        st.download_button(
            label="Download validation report JSON",
            data=open(report_path, "rb"),
            file_name="ingestion_health_report.json",
            mime="application/json",
        )
    else:
        st.download_button(
            label="Download validation report JSON",
            data=json.dumps(report, indent=2),
            file_name="ingestion_health_report.json",
            mime="application/json",
        )


def section_draft_roster():
    st.header("Step 2 — Draft roster (Solver)")
    render_last_updated("Last updated", "solver_manifest.json")

    st.write("Generate a draft roster using the optimisation solver.")

    if st.button("Run solver", type="primary"):
        with st.spinner("Running solver..."):
            result = st.session_state.controller.run("solve roster")
            st.session_state.solver_result = result

    result = st.session_state.solver_result
    if result is None:
        st.info("Click **Run solver** to generate a draft roster.")
        sol_df = load_csv_if_exists(artifact_path("roster_solution.csv"))
        if sol_df is not None:
            st.markdown("#### Latest solver roster (from disk)")
            st.dataframe(sol_df, use_container_width=True)
        return

    state = result.get("state", {})
    solver_state = state.get("solver", {})
    sol_df = solver_state.get("solution")
    if not isinstance(sol_df, pd.DataFrame):
        sol_df = load_csv_if_exists(artifact_path("roster_solution.csv"))

    st.success("Solver run completed.")
    st.markdown("#### Solver manifest")
    manifest = solver_state.get("manifest")
    if not manifest:
        manifest = load_json_if_exists(artifact_path("solver_manifest.json")) or {}
    st.json(manifest)

    if isinstance(sol_df, pd.DataFrame) and not sol_df.empty:
        st.markdown("#### Draft roster")
        st.dataframe(sol_df, use_container_width=True)
    else:
        st.warning("No solution DataFrame returned. Check `roster_solution.csv` in artifacts.")

    path = solver_state.get("solution_path") or artifact_path("roster_solution.csv")
    if path and os.path.exists(path):
        st.download_button(
            label="Download draft roster CSV",
            data=open(path, "rb"),
            file_name="draft_roster.csv",
            mime="text/csv",
        )


def section_compliance():
    st.header("Step 3 — Compliance")
    render_last_updated("Last updated", "compliance_report.json")

    st.write("Validate the draft roster against rostering rules and constraints.")

    if st.button("Run compliance", type="primary"):
        with st.spinner("Running compliance checks..."):
            result = st.session_state.controller.run("compliance")
            st.session_state.compliance_result = result

    result = st.session_state.compliance_result
    if result is None:
        st.info("Click **Run compliance** to generate a compliance report.")
        comp = load_json_if_exists(artifact_path("compliance_report.json"))
        if comp is not None:
            st.markdown("#### Latest compliance report (from disk)")
            st.json(comp)
        return

    state = result.get("state", {})
    comp_state = state.get("compliance", {})
    report = comp_state.get("report", {})

    st.success("Compliance run completed.")

    summary = report.get("summary", {})
    violations = report.get("violations", [])
    notes = report.get("notes", [])

    st.markdown("#### Summary")
    st.json(summary)

    st.markdown("#### Violations")
    if violations:
        df = pd.DataFrame(violations)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No violations reported.")

    if notes:
        with st.expander("Notes", expanded=False):
            st.json(notes)

    # ---------------------------------------------------------
    # Coverage Heatmap
    # ---------------------------------------------------------
    coverage_checks = report.get("coverage_checks", [])

    if coverage_checks:
        st.markdown("### Coverage Heatmap")

        cov_df = pd.DataFrame(coverage_checks)
        cov_df["station_period"] = cov_df["station"] + " - " + cov_df["service_period"]

        heatmap_df = cov_df.pivot_table(
            index="date",
            columns="station_period",
            values="shortfall",
            aggfunc="sum",
            fill_value=0,
        )

        fig = px.imshow(
            heatmap_df,
            labels=dict(x="Station / Period", y="Date", color="Shortfall"),
            color_continuous_scale="Reds",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No coverage data available for heatmap.")

    path = comp_state.get("path")
    if path and os.path.exists(path):
        st.download_button(
            label="Download compliance report JSON",
            data=open(path, "rb"),
            file_name="compliance_report.json",
            mime="application/json",
        )


def section_swaps():
    st.header("Step 4 — Swaps & fixes")
    render_last_updated("Last updated", "swaps_report.json")

    st.write("Review and confirm suggested swaps to fix compliance issues.")

    if st.button("Generate swap suggestions", type="primary"):
        with st.spinner("Running SwapAgent..."):
            result = st.session_state.controller.run("swap suggestions")
            st.session_state.swaps_result = result

    result = st.session_state.swaps_result
    swaps_path = artifact_path("swaps_report.json")

    if result is not None:
        state = result.get("state", {})
        swaps_state = state.get("swaps", {})
        swaps = swaps_state.get("swaps", [])
    else:
        payload = load_json_if_exists(swaps_path)
        swaps = payload.get("swaps", []) if payload else []

    if not swaps:
        st.info("No swaps found yet. Run swap suggestions after compliance.")
        return

    st.markdown("#### Suggested swaps")

    bulk_col1, bulk_col2 = st.columns(2)
    confirm_all = bulk_col1.button("Confirm all swaps")
    clear_all = bulk_col2.button("Clear all confirmations")

    if confirm_all:
        for idx in range(len(swaps)):
            st.session_state[f"swap_confirm_{idx}"] = True
    if clear_all:
        for idx in range(len(swaps)):
            st.session_state[f"swap_confirm_{idx}"] = False

    updated_swaps = []

    for idx, swap in enumerate(swaps):
        viol = swap.get("violation", {})
        emp = viol.get("employee", "?")
        cand = swap.get("suggested_employee", "?")
        day = swap.get("date", "?")
        station = swap.get("station", "?")

        with st.expander(f"Swap {idx + 1}: {emp} → {cand} on {day} at {station}"):
            st.json(swap)
            confirmed = st.checkbox(
                "Confirm this swap",
                value=swap.get("confirmed", False),
                key=f"swap_confirm_{idx}",
            )
            swap["confirmed"] = confirmed
            if confirmed:
                swap["rejected"] = False
            updated_swaps.append(swap)

    if st.button("Save confirmations", type="primary"):
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        payload = {"swaps": updated_swaps}
        with open(swaps_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        st.success("Swap confirmations saved to swaps_report.json.")


def section_final_roster():
    st.header("Step 5 — Final roster")
    render_last_updated("Last updated", "final_roster_manifest.json")

    st.write("Apply confirmed swaps and generate the audit-ready final roster.")

    if st.button("Generate final roster", type="primary"):
        with st.spinner("Generating final roster..."):
            result = st.session_state.controller.run("final roster")
            st.session_state.final_result = result

    result = st.session_state.final_result
    final_df = None
    manifest_path = artifact_path("final_roster_manifest.json")
    final_csv_path = artifact_path("final_roster.csv")

    if result is not None:
        state = result.get("state", {})
        fr_state = state.get("final_roster", {})
        final_df = fr_state.get("final_roster")
        manifest_path = fr_state.get("manifest_path", manifest_path)
        summary = fr_state.get("summary", {})

        st.success("Final roster generated.")
        st.markdown("#### Summary")
        st.json(summary)

        swaps_applied = summary.get("swaps_applied", 0)
        swaps_pending = summary.get("swaps_pending", 0)
        st.info(f"Swaps applied: {swaps_applied} | Pending: {swaps_pending}")

        note = summary.get("note", "")
        if "No swaps applied" in note:
            st.warning("Final roster equals draft roster (no swaps applied).")
        pending_note = summary.get("pending_note", "")
        if pending_note:
            st.info(pending_note)
    else:
        st.info("Click **Generate final roster** to run FinalRosterAgent.")

    if final_df is None:
        final_df = load_csv_if_exists(final_csv_path)

    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        st.markdown("#### Final roster")
        if "code" in final_df.columns:
            break_mask = final_df["code"].astype(str).str.contains("BREAK", case=False, na=False)
            break_count = int(break_mask.sum())
            work_count = int((~break_mask).sum())
        else:
            break_mask = None
            break_count = 0
            work_count = len(final_df)

        st.caption(f"Work shifts: {work_count} | Break rows: {break_count}")
        show_breaks = st.checkbox("Show break rows", value=False)
        if break_mask is not None and not show_breaks:
            st.dataframe(final_df[~break_mask], use_container_width=True)
        else:
            st.dataframe(final_df, use_container_width=True)

        st.download_button(
            label="Download final roster CSV",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="final_roster.csv",
            mime="text/csv",
        )

    draft_df = load_csv_if_exists(artifact_path("roster_solution.csv"))

    if draft_df is not None and isinstance(final_df, pd.DataFrame):

        st.markdown("### Before / After Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Before (Draft roster)**")
            st.dataframe(draft_df, use_container_width=True)

        with col2:
            st.markdown("**After (Final roster)**")
            st.dataframe(final_df, use_container_width=True)

        st.markdown("### Changes Highlighted")

        common_cols = [c for c in draft_df.columns if c in final_df.columns]
        draft_aligned = draft_df.filter(items=common_cols).copy()
        final_aligned = final_df.filter(items=common_cols).copy()

        # Streamlit can re-run steps and leave stale artifacts behind.
        # If the row counts differ, a direct elementwise comparison will raise.
        if len(draft_aligned) != len(final_aligned):
            st.warning(
                f"Draft ({len(draft_aligned)} rows) and final ({len(final_aligned)} rows) differ in row count. "
                "Showing added/removed rows instead of row-by-row highlighting."
            )

            added_rows = final_aligned.merge(
                draft_aligned.drop_duplicates(),
                how="left",
                on=common_cols,
                indicator=True,
            )
            added_rows = added_rows[added_rows["_merge"] == "left_only"].drop(columns=["_merge"])

            removed_rows = draft_aligned.merge(
                final_aligned.drop_duplicates(),
                how="left",
                on=common_cols,
                indicator=True,
            )
            removed_rows = removed_rows[removed_rows["_merge"] == "left_only"].drop(columns=["_merge"])

            if not added_rows.empty:
                st.markdown("**Added in final**")
                st.dataframe(added_rows, use_container_width=True)
            if not removed_rows.empty:
                st.markdown("**Removed from draft**")
                st.dataframe(removed_rows, use_container_width=True)

            diff_mask = pd.Series([False] * len(final_aligned), index=final_aligned.index)
            diff_df = final_aligned.copy()
            diff_df["changed"] = diff_mask
        else:
            sort_cols = [
                c
                for c in ["date", "station", "start_time", "end_time", "code", "employee_id", "employee"]
                if c in common_cols
            ]
            if sort_cols:
                by_arg = sort_cols[0] if len(sort_cols) == 1 else sort_cols
                draft_aligned = draft_aligned.sort_values(by=by_arg).reset_index(drop=True)
                final_aligned = final_aligned.sort_values(by=by_arg).reset_index(drop=True)
            else:
                draft_aligned = draft_aligned.reset_index(drop=True)
                final_aligned = final_aligned.reset_index(drop=True)

            diff_mask = (draft_aligned.fillna("") != final_aligned.fillna("")).any(axis=1)
            diff_df = final_aligned.copy()
            diff_df["changed"] = diff_mask

        def highlight_changes(row):
            return [
                "background-color: #ffeb99" if row.get("changed") else ""
                for _ in row
            ]

        st.dataframe(
            diff_df.style.apply(highlight_changes, axis=1),
            use_container_width=True,
        )

        st.markdown("### Rows that changed")

        changed_rows = (
            final_aligned.loc[diff_mask]
            if hasattr(final_aligned, "loc") and len(final_aligned) == len(diff_mask)
            else pd.DataFrame()
        )
        if not isinstance(changed_rows, pd.DataFrame):
            changed_rows = pd.DataFrame(changed_rows)

        if len(changed_rows) > 0:
            st.dataframe(changed_rows, use_container_width=True)
        else:
            st.info("No differences between draft and final roster.")

        comparison_df = pd.concat(
            [
                draft_aligned.add_prefix("before_"),
                final_aligned.add_prefix("after_"),
                diff_mask.rename("changed"),
            ],
            axis=1,
        )

        st.download_button(
            label="Download Before/After Comparison CSV",
            data=comparison_df.to_csv(index=False).encode("utf-8"),
            file_name="before_after_comparison.csv",
            mime="text/csv",
        )

    # ---------------------------------------------------------
    # Re-run compliance on final roster
    # ---------------------------------------------------------
    st.markdown("### Re-run compliance on final roster")

    if st.button("Run compliance on final roster"):
        with st.spinner("Re-running compliance on final roster..."):
            comp_agent = ComplianceAgent(
                roster_path=final_csv_path,
                params_path=artifact_path("rostering_parameters.json"),
                staffing_path=artifact_path("staffing_requirements.json"),
                management_path=artifact_path("management_availability.json"),
                out_dir=ARTIFACT_DIR,
            )
            final_comp_report = comp_agent.run()

        st.success("Compliance re-run completed.")

        st.markdown("#### Final roster compliance summary")
        st.json(final_comp_report.get("summary", {}))

        st.markdown("#### Final roster violations")
        final_viol = final_comp_report.get("violations", [])
        if final_viol:
            st.dataframe(pd.DataFrame(final_viol), use_container_width=True)
        else:
            st.info("No violations in final roster — all good!")

        st.markdown("#### Notes")
        st.json(final_comp_report.get("notes", []))

    manifest = load_json_if_exists(manifest_path)
    if manifest:
        with st.expander("Final roster manifest", expanded=False):
            st.json(manifest)


def section_artifacts():
    st.header("Artifacts browser")
    st.caption(f"Artifacts directory: {ARTIFACT_DIR}")

    if st.button("Clear artifacts (active mode)"):
        removed = clear_artifacts_dir(ARTIFACT_DIR)
        st.success(f"Cleared {removed} items from {ARTIFACT_DIR}")

    st.write("Quick view of key artifacts under the active artifacts folder.")

    artifacts_dir = Path(ARTIFACT_DIR)
    if not artifacts_dir.exists():
        st.warning(f"Artifacts directory not found: {ARTIFACT_DIR}")
        return

    files = sorted(artifacts_dir.glob("*"))
    for f in files:
        st.write(f"- `{f.name}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### View CSV")
        csv_files = [f for f in files if f.suffix.lower() == ".csv"]
        csv_choice = st.selectbox(
            "Choose a CSV to preview", ["(none)"] + [f.name for f in csv_files]
        )
        if csv_choice != "(none)":
            df = pd.read_csv(artifacts_dir / csv_choice)
            st.dataframe(df, use_container_width=True)

    with col2:
        st.markdown("#### View JSON")
        json_files = [f for f in files if f.suffix.lower() == ".json"]
        json_choice = st.selectbox(
            "Choose a JSON to preview", ["(none)"] + [f.name for f in json_files]
        )
        if json_choice != "(none)":
            obj = load_json_if_exists(str(artifacts_dir / json_choice))
            st.json(obj)


def section_rag_qa():
    st.header("RAG Q&A")
    st.write("Ask questions about constraints, artifacts, and compliance results.")

    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = ""
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []
    if "rag_last_query" not in st.session_state:
        st.session_state.rag_last_query = ""

    query = st.text_area("Your question", placeholder="e.g., Why are there coverage violations on Tue Dec 10?")
    if query.strip() != st.session_state.rag_last_query:
        st.session_state.rag_answer = ""
        st.session_state.rag_sources = []
        st.session_state.rag_last_query = query.strip()

    if st.button("Ask") and query.strip():
        with st.spinner("Retrieving context and generating answer..."):
            rag = run_knowledge_rag(query)

            if rag.get("error"):
                st.error(rag.get("error"))
                return

            st.session_state.rag_answer = rag.get("answer", "")
            st.session_state.rag_sources = rag.get("sources", [])

    if st.session_state.rag_answer:
        st.markdown("#### Answer")
        st.write(st.session_state.rag_answer)

        if st.session_state.rag_sources:
            st.markdown("#### Sources")
            st.json(st.session_state.rag_sources)


def section_explanation():
    st.header("Explanation Report")
    st.write("Generate an auditor-friendly explanation of compliance outcomes and swaps.")

    if st.button("Generate explanation", type="primary"):
        with st.spinner("Generating explanation..."):
            result = run_explanation_agent()
            if result.get("error"):
                st.error(result.get("error"))
                return

            st.success("Explanation generated.")

            st.markdown("#### Explanation")
            st.write(result.get("explanation_text", ""))

            st.markdown("#### Output file")
            st.write("`data/artifacts/explanation_report.json`")


def section_pipeline():
    st.header("Full Pipeline Execution")

    st.write("Run the entire rostering pipeline end‑to‑end:")

    st.markdown(
        """
**Pipeline steps:**
1. Ingestion  
2. Validation  
3. Draft roster (Solver)  
4. Compliance  
5. Swap suggestions  
6. Final roster  
"""
    )

    if st.button("Run Full Pipeline", type="primary"):
        controller = st.session_state.controller

        # 1. INGESTION
        with st.spinner("Running ingestion..."):
            ingest_result = controller.run("ingest")
        st.success("Ingestion completed.")
        st.json(ingest_result.get("state", {}).get("ingestion", {}).get("manifest", {}))

        # 2. VALIDATION
        with st.spinner("Running validation..."):
            val_result = controller.run("validate ingestion")
        st.success("Validation completed.")
        st.json(val_result.get("state", {}).get("validate_ingestion", {}).get("report", {}))

        # 3. SOLVER
        with st.spinner("Running solver..."):
            solver_result = controller.run("solve roster")
        st.success("Draft roster generated.")
        st.json(solver_result.get("state", {}).get("solver", {}).get("manifest", {}))

        # 4. COMPLIANCE
        with st.spinner("Running compliance..."):
            comp_result = controller.run("compliance")
        st.success("Compliance completed.")
        st.json(
            comp_result.get("state", {})
            .get("compliance", {})
            .get("report", {})
            .get("summary", {})
        )

        # 5. SWAP SUGGESTIONS
        with st.spinner("Generating swap suggestions..."):
            swap_result = controller.run("swap suggestions")
        st.success("Swap suggestions generated.")
        st.json(swap_result.get("state", {}).get("swaps", {}))
        st.info("Go to the Swaps tab to confirm or reject individual swap suggestions.")

        # 6. FINAL ROSTER
        with st.spinner("Generating final roster..."):
            final_result = controller.run("final roster")
        st.success("Final roster generated.")

        st.markdown("### Final Summary")
        st.json(
            final_result.get("state", {})
            .get("final_roster", {})
            .get("summary", {})
        )

        st.success("Pipeline completed successfully.")


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main():
    init_session_state()

    page = "Overview"

    st.set_page_config(
        page_title="MAS Rostering Demo",
        page_icon="📅",
        layout="wide",
    )

    with st.sidebar:
        st.title("MAS Rostering")
        st.caption("Multi-agent roster generation and validation.")

        if st.button("Clear artifacts"):
            removed = clear_artifacts_dir(ARTIFACT_DIR)
            st.success(f"Cleared {removed} items from {ARTIFACT_DIR}")

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Ingestion",
                "Validation",
                "Draft roster",
                "Compliance",
                "Swaps",
                "Final roster",
                "RAG Q&A",
                "Explanation",
                "Artifacts",
            ],
        )

    if page == "Overview":
        section_overview()
    elif page == "Ingestion":
        section_ingestion()
    elif page == "Validation":
        section_validation()
    elif page == "Draft roster":
        section_draft_roster()
    elif page == "Compliance":
        section_compliance()
    elif page == "Swaps":
        section_swaps()
    elif page == "Final roster":
        section_final_roster()
    elif page == "RAG Q&A":
        section_rag_qa()
    elif page == "Explanation":
        section_explanation()
    elif page == "Artifacts":
        section_artifacts()


if __name__ == "__main__":
    main()
