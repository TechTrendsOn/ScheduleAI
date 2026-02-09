# MAS Rostering Demo (Agentic AI)

Multi-agent workforce scheduling prototype for restaurant operations.

This project ingests messy Excel/CSV inputs, generates a draft roster with OR-Tools CP-SAT, validates compliance, suggests swaps, produces a final roster, and adds RAG/Q&A + explanation agents for decision transparency.

## What This Demonstrates

- Multi-agent orchestration for a practical scheduling workflow
- Constraint-based optimization for roster generation
- Compliance validation against rostering rules
- Interactive swap confirmation flow
- RAG-based question answering over project artifacts
- Explanation agent for audit-friendly summaries

## Architecture

Core agents:

- `ingestion/ingestion_agent.py` - reads and normalizes messy source files
- `agents/solver_agent.py` - builds draft roster with CP-SAT
- `agents/compliance_agent.py` - runs rule checks and reports violations
- `agents/swap_agent.py` - proposes feasible swaps
- `agents/final_roster_agent.py` - applies confirmed swaps and writes final roster
- `agents/knowledge_agent.py` - RAG Q&A over artifacts (Chroma + MiniLM + Flan-T5)
- `agents/explanation_agent.py` - narrative explanation of outcomes
- `agents/controller_agent.py` - orchestrates step-level intent/tool execution

UI:

- `streamlit_dashboard.py`

## Project Structure

```text
agents/
ingestion/
data/
  artifacts/          # runtime-generated outputs (do not commit)
streamlit_dashboard.py
requirements.txt
```

## Inputs and Outputs

Input files (in `data/`):

- employee availability and manager roster
- shift code definitions
- store configuration
- staffing estimate/requirements
- rostering parameters

Generated outputs (in `data/artifacts/`):

- `availability_tidy.csv`
- `shift_codes_cleaned.csv`
- `rostering_parameters.json`
- `staffing_requirements.json`
- `management_availability.json`
- `ingestion_manifest.json`
- `ingestion_health_report.json`
- `roster_solution.csv`
- `solver_manifest.json`
- `compliance_report.json`
- `swaps_report.json`
- `final_roster.csv`
- `final_roster_manifest.json`
- `explanation_report.json`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_dashboard.py
```

## How To Use (Recommended Flow)

1. Ingestion
2. Validation
3. Draft roster (Solver)
4. Compliance
5. Swaps (confirm/reject)
6. Final roster
7. RAG Q&A / Explanation

## Notes

- 'data/artifacts/' is generated at runtime and should not be committed.
- Some datasets may produce no feasible swaps under strict rules (this is valid behavior).
- RAG agents use local open-source models and may be slow on first run due to model downloads.

## Tech Stack

- Python, Streamlit, Pandas
- openpyxl (for .xlsx ingestion)
- OR-Tools (CP-SAT)
- ChromaDB
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Hugging Face Transformers (`google/flan-t5-small`)
- LangChain (HF integration)
- Plotly (dashboard visualizations)

## Usage

This project is for demo and portfolio purposes only.
