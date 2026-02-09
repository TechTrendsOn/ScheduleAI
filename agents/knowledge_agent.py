# agents/knowledge_agent.py

import os
import json
import uuid
from typing import List, Dict, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class KnowledgeAgentRAG:
    """
    RAG agent over ingestion and compliance artifacts.
    Builds a local ChromaDB index and answers questions using an open-source LLM.
    """

    def __init__(
        self,
        artifact_dir: str = "data/artifacts",
        persistence_dir: str = "data/artifacts/chroma_knowledge",
        collection_name: str = "rostering_knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-small",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        ):
        self.artifact_dir = artifact_dir
        self.persistence_dir = persistence_dir
        self.collection_name = collection_name
        os.makedirs(self.persistence_dir, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.client = chromadb.PersistentClient(path=self.persistence_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        hf_pipe = pipeline(
            "text2text-generation",
            model=llm_model_name,
            device=-1,
            max_length=512,
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    # -------------------------------
    # Loaders
    # -------------------------------
    def load_excel(self, path: str, sheet: Optional[str] = None) -> List[Dict]:
        if sheet:
            df = pd.read_excel(path, sheet_name=sheet)
            return self._df_to_docs(df, source=path, sheet=sheet)

        xl = pd.ExcelFile(path)
        docs: List[Dict] = []
        for sh in xl.sheet_names:
            df = xl.parse(sh)
            docs.extend(self._df_to_docs(df, source=path, sheet=sh))
        return docs

    def load_csv(self, path: str) -> List[Dict]:
        df = pd.read_csv(path)
        return self._df_to_docs(df, source=path, sheet=None)

    def load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def flatten(d, prefix: str = "") -> List[str]:
            items: List[str] = []
            for k, v in d.items():
                new_key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten(v, prefix=new_key + "."))
                else:
                    items.append(f"{new_key}: {v}")
            return items

        text = "\n".join(flatten(data))
        meta = {"source": os.path.normpath(path), "sheet": "", "row_index": -1, "columns": ""}
        return [{"text": text, "metadata": meta}]

    def _df_to_docs(self, df: pd.DataFrame, source: str, sheet: Optional[str]) -> List[Dict]:
        docs: List[Dict] = []
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        source = os.path.normpath(source)

        for i, row in df.iterrows():
            kv_pairs = []
            for col in df.columns:
                val = row.get(col)
                if pd.isna(val):
                    continue
                kv_pairs.append(f"{col}: {val}")
            if not kv_pairs:
                continue

            text = "\n".join(kv_pairs)
            meta = {
                "source": source,
                "sheet": sheet or "",
                "row_index": int(i),
                "columns": ", ".join(df.columns.tolist()),
            }
            docs.append({"text": text, "metadata": meta})

        return docs

    # -------------------------------
    # Chunking + embeddings
    # -------------------------------
    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = list(text)
        chunks: List[str] = []
        start = 0
        n = len(tokens)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = "".join(tokens[start:end])
            chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    def add_documents(self, docs: List[Dict]):
        ids, texts, metadatas, embeddings = [], [], [], []

        for doc in docs:
            chunks = self._chunk_text(doc["text"])
            for ch in chunks:
                _id = str(uuid.uuid4())
                ids.append(_id)
                texts.append(ch)
                metadatas.append(doc["metadata"])
                vec = self.embedding_model.encode(ch, convert_to_numpy=True)
                embeddings.append(vec.tolist())

        if not ids:
            return

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        try:
            self.client.persist()
        except Exception:
            pass

    # -------------------------------
    # Retrieval + answer
    # -------------------------------
    def retrieve(self, query: str, k: int = 6) -> Tuple[List[str], List[Dict]]:
        q_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        allowed_sources = self._source_filter(query)
        where = None
        if allowed_sources:
            where = {"source": {"$in": [os.path.normpath(s) for s in allowed_sources]}}

        res = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            where=where,
        )
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        return docs, metas

    def _source_filter(self, query: str) -> List[str]:
        q = query.lower()
        sources = []
        if "violation" in q or "violations" in q:
            sources.append("data/artifacts/compliance_report.json")
        if any(k in q for k in ["coverage", "shortfall", "staffing", "required", "actual"]):
            sources.extend([
                "data/artifacts/compliance_report.json",
                "data/artifacts/staffing_requirements.json",
            ])
        if any(k in q for k in ["rest", "consecutive", "weekly", "hours", "shift length"]):
            sources.extend([
                "data/artifacts/compliance_report.json",
                "data/artifacts/rostering_parameters.json",
            ])
        if any(k in q for k in ["shift code", "code", "token"]):
            sources.append("data/artifacts/shift_codes_cleaned.csv")
        if "availability" in q:
            sources.append("data/artifacts/availability_tidy.csv")

        if not sources:
            sources = [
                "data/artifacts/compliance_report.json",
                "data/artifacts/rostering_parameters.json",
            ]

        seen = set()
        unique = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        # Hard cap total context length to avoid model overflow
        max_chars = 2000
        trimmed: List[str] = []
        total = 0
        for c in contexts:
            if total >= max_chars:
                break
            remaining = max_chars - total
            snippet = c[:remaining]
            trimmed.append(snippet)
            total += len(snippet)

        ctx_block = "\n\n".join(f"- {c}" for c in trimmed) if trimmed else "None."
        return (
            "You are a data analyst for the MAS Rostering system.\n"
            "Use ONLY the provided context to answer the question. "
            "If the answer is not in context, say you don’t have enough information and name the missing artifact.\n"
            "If the question mentions a date, employee, station, rule, or violation type, filter to that scope.\n"
            "When possible, cite the source file name and include key fields: date, employee, station, service_period, "
            "violation_type, and rule.\n"
            "Respond in clear, friendly language. If the user asks for a list or details, use a short bullet list.\n\n"
            f"Context:\n{ctx_block}\n\n"
            f"Question:\n{query}\n\n"
            "Answer succinctly and avoid dumping raw data."
        )

    def answer(self, query: str, k: int = 3) -> Dict:
        # Deterministic shortcuts for common questions
        direct = self._direct_answer(query)
        if direct is not None:
            return direct

        contexts, metas = self.retrieve(query, k=k)
        prompt = self._build_prompt(query, contexts)
        out = self.llm.invoke(prompt)

        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            answer_text = out[0].get("generated_text", "").strip()
        else:
            answer_text = str(out).strip()

        # Simple cleanup: truncate long raw dumps
        if "time_kind" in answer_text or "start_time" in answer_text:
            answer_text = (
                "Coverage violations usually mean required staffing was higher than scheduled "
                "headcount for that date/period. See compliance and staffing artifacts for details."
            )

        if len(answer_text) > 600:
            answer_text = answer_text[:600].rsplit(" ", 1)[0] + "..."

        sources = [
            {
                "source": (m or {}).get("source", ""),
                "sheet": (m or {}).get("sheet", ""),
                "row_index": (m or {}).get("row_index", -1),
            }
            for m in metas
        ]

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s.get("source"), s.get("sheet"), s.get("row_index"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "query": query,
            "answer": answer_text,
            "contexts": contexts,
            "sources": unique_sources,
        }

    def _direct_answer(self, query: str) -> Optional[Dict]:
        q = query.lower().strip()

        if "violation" in q or "violations" in q:
            path = os.path.join(self.artifact_dir, "compliance_report.json")
            if os.path.exists(path):
                data = json.load(open(path, "r", encoding="utf-8"))
                summary = data.get("summary", {})
                vtypes = summary.get("violation_types", {})
                if vtypes:
                    parts = [f"{k}: {v}" for k, v in vtypes.items()]
                    answer = "Current violation types are: " + ", ".join(parts) + "."
                else:
                    count = summary.get("violations_count", 0)
                    answer = f"There are {count} total violations."
                return {
                    "query": query,
                    "answer": answer,
                    "contexts": [],
                    "sources": [{"source": os.path.normpath(path), "sheet": "", "row_index": -1}],
                }

        if "shift code" in q or "shift codes" in q:
            path = os.path.join(self.artifact_dir, "shift_codes_cleaned.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                codes = df[["code", "start_time", "end_time"]].dropna().head(12)
                items = [
                    f"{r.code}: {r.start_time}-{r.end_time}" for r in codes.itertuples(index=False)
                ]
                answer = "Shift codes map to time ranges. Examples: " + ", ".join(items) + "."
                return {
                    "query": query,
                    "answer": answer,
                    "contexts": [],
                    "sources": [{"source": os.path.normpath(path), "sheet": "", "row_index": -1}],
                }

        if "coverage" in q and "dec" in q:
            path = os.path.join(self.artifact_dir, "compliance_report.json")
            if os.path.exists(path):
                data = json.load(open(path, "r", encoding="utf-8"))
                violations = data.get("violations", [])
                hits = [v for v in violations if v.get("type") == "Coverage" and "dec" in str(v.get("date", "")).lower()]
                if hits:
                    stations = sorted({v.get("station") for v in hits if v.get("station")})
                    answer = (
                        f"Coverage violations on those Dec dates indicate scheduled headcount fell below "
                        f"requirements in {len(hits)} cases. Affected stations include: {', '.join(stations) or 'Unknown'}."
                    )
                    return {
                        "query": query,
                        "answer": answer,
                        "contexts": [],
                        "sources": [{"source": os.path.normpath(path), "sheet": "", "row_index": -1}],
                    }
        return None

    # -------------------------------
    # Index build
    # -------------------------------
    def ingest_files(self, files: List[Dict[str, str]]):
        all_docs: List[Dict] = []
        for f in files:
            path = f["path"]
            ftype = f.get("type", "csv").lower()
            sheet = f.get("sheet")

            if not os.path.exists(path):
                continue

            if ftype == "excel":
                docs = self.load_excel(path, sheet=sheet)
            elif ftype == "csv":
                docs = self.load_csv(path)
            elif ftype == "json":
                docs = self.load_json(path)
            else:
                raise ValueError(f"Unsupported file type: {ftype}")

            all_docs.extend(docs)

        self.add_documents(all_docs)
