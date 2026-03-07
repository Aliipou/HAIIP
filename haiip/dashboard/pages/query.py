"""RAG Query page — natural-language knowledge base search."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_post, is_demo
from haiip.dashboard.components.theme import section_title

EXAMPLE_QUESTIONS = [
    "What causes Heat Dissipation Failure (HDF) in CNC machines?",
    "How do I interpret a high kurtosis value in bearing vibration data?",
    "What is the recommended maintenance interval for tool replacement?",
    "How should I respond to a critical torque anomaly alert?",
    "What does an anomaly score above 0.8 indicate?",
    "How does concept drift affect model accuracy over time?",
]


def render() -> None:
    st.markdown("## 💬 Knowledge Base Query")
    st.caption(
        "Ask natural-language questions about your machines, fault types, "
        "maintenance procedures, and AI model behaviour."
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    # ── Example questions ─────────────────────────────────────────────────────
    section_title("Example Questions")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state["rag_prefill"] = q

    st.markdown("---")

    # ── Query input ───────────────────────────────────────────────────────────
    prefill = st.session_state.pop("rag_prefill", "")
    col_q, col_m = st.columns([4, 1])
    with col_q:
        question = st.text_input(
            "Ask a question",
            value=prefill,
            placeholder="e.g. What are early signs of bearing failure?",
        )
    with col_m:
        machine_filter = st.text_input("Filter by machine", placeholder="CNC-001 (optional)")

    col_k, col_btn = st.columns([1, 3])
    with col_k:
        top_k = st.slider("Context depth", min_value=1, max_value=10, value=5)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        ask_btn = st.button("🔍 Ask", use_container_width=True, disabled=not question.strip())

    # ── Send query ────────────────────────────────────────────────────────────
    if ask_btn and question.strip():
        with st.spinner("Searching knowledge base…"):
            answer_data = _query(question.strip(), machine_filter.strip() or None, top_k)

        st.session_state["rag_history"].insert(
            0,
            {
                "question": question,
                "answer": answer_data.get("answer", "No answer available."),
                "sources": answer_data.get("sources", []),
                "confidence": answer_data.get("confidence", 0.0),
            },
        )

    # ── Document ingestion ────────────────────────────────────────────────────
    with st.expander("📄 Ingest document into knowledge base"):
        with st.form("ingest_form"):
            doc_title = st.text_input("Document title", placeholder="CNC-001 Maintenance Manual v3")
            doc_source = st.selectbox(
                "Source type",
                ["manual", "iso_standard", "fault_report", "sop", "other"],
            )
            doc_machine = st.text_input("Link to machine (optional)", placeholder="CNC-001")
            doc_content = st.text_area(
                "Document content",
                height=200,
                placeholder="Paste the document text here…",
            )
            ingest_btn = st.form_submit_button("📥 Ingest document", use_container_width=True)

        if ingest_btn:
            if not doc_title or not doc_content:
                st.error("Title and content are required.")
            elif len(doc_content) < 10:
                st.error("Content too short.")
            else:
                resp = _ingest(doc_title, doc_content, doc_source, doc_machine or None)
                if resp:
                    st.success(
                        f"✅ Ingested '{doc_title}' — "
                        f"{resp.get('chunks_created', 0)} chunks created. "
                        f"Total documents: {resp.get('total_documents', 0)}"
                    )
                else:
                    st.warning("Document ingested (demo mode — stored in memory only).")

    # ── Answer history ────────────────────────────────────────────────────────
    if st.session_state["rag_history"]:
        section_title("Conversation")

        for entry in st.session_state["rag_history"]:
            # Question bubble
            st.markdown(
                f"""
                <div style="background:#242C3E; border:1px solid #2D3748; border-radius:12px 12px 4px 12px;
                            padding:0.75rem 1rem; margin-bottom:0.4rem; max-width:85%; margin-left:auto;">
                    <div style="font-size:0.75rem; color:#718096; margin-bottom:4px;">YOU</div>
                    <div style="color:#E2E8F0;">{entry["question"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Answer bubble
            conf = entry["confidence"]
            conf_color = "#00C851" if conf > 0.7 else "#FFB400" if conf > 0.4 else "#718096"
            st.markdown(
                f"""
                <div style="background:#1C2333; border:1px solid #2D3748;
                            border-left:3px solid #00D4FF;
                            border-radius:4px 12px 12px 12px;
                            padding:0.75rem 1rem; margin-bottom:0.75rem; max-width:92%;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:0.75rem; color:#00D4FF;">⚙️ HAIIP</span>
                        <span style="font-size:0.72rem; color:{conf_color};">
                            Confidence: {conf * 100:.0f}%
                        </span>
                    </div>
                    <div style="color:#E2E8F0; line-height:1.6;">{entry["answer"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sources
            if entry["sources"]:
                with st.expander(f"📚 {len(entry['sources'])} source(s) retrieved"):
                    for src in entry["sources"]:
                        st.markdown(
                            f"**{src.get('title', '—')}** "
                            f"<span style='color:#718096;'>({src.get('source', '—')})</span> "
                            f"· score: `{src.get('score', 0):.3f}`<br>"
                            f"<span style='color:#A0AEC0; font-size:0.8rem;'>"
                            f"{src.get('excerpt', '')}</span>",
                            unsafe_allow_html=True,
                        )

        if st.button("🗑 Clear history"):
            st.session_state["rag_history"] = []
            st.rerun()


def _query(question: str, machine_id: str | None, top_k: int) -> dict:
    if is_demo():
        from haiip.core.rag import RAGEngine

        engine = RAGEngine(top_k=top_k)
        engine.initialize()
        _seed_demo_docs(engine)
        result = engine.query(question, machine_id=machine_id, top_k=top_k)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
        }
    payload: dict = {"question": question, "context_window": top_k}
    if machine_id:
        payload["machine_id"] = machine_id
    resp = api_post("/api/v1/query", payload)
    return resp or {"answer": "API unavailable.", "sources": [], "confidence": 0.0}


def _ingest(title: str, content: str, source: str, machine_id: str | None) -> dict | None:
    if is_demo():
        return {
            "chunks_created": max(1, len(content.split()) // 100),
            "total_documents": 5,
        }
    payload: dict = {"title": title, "content": content, "source": source}
    if machine_id:
        payload["machine_id"] = machine_id
    return api_post("/api/v1/documents/ingest", payload)


def _seed_demo_docs(engine) -> None:
    """Seed a few demo documents into the RAG engine."""
    from haiip.core.rag import Document

    if engine.document_count == 0:
        demo_docs = [
            Document(
                title="HDF Failure Guide",
                content=(
                    "Heat Dissipation Failure (HDF) occurs when the difference between "
                    "process temperature and air temperature is below 8.6K while the "
                    "rotational speed is below 1380 RPM. Check cooling system, fans, "
                    "and heat exchangers. Increase coolant flow rate. Schedule preventive "
                    "maintenance if HDF events occur more than twice per shift."
                ),
                source="manual",
            ),
            Document(
                title="Tool Wear Failure (TWF) Procedure",
                content=(
                    "Tool Wear Failure (TWF) is triggered when tool wear exceeds 200 minutes "
                    "for L-grade products or 240 minutes for H-grade. Replace cutting tool "
                    "immediately. Use HAIIP RUL prediction to schedule tool changes proactively. "
                    "High kurtosis in vibration signals (>6) combined with rising torque is "
                    "an early indicator."
                ),
                source="sop",
            ),
            Document(
                title="Anomaly Score Interpretation",
                content=(
                    "HAIIP anomaly scores range 0-1. Score < 0.4: normal operation. "
                    "Score 0.4-0.7: elevated risk, increase monitoring frequency. "
                    "Score > 0.7: anomaly detected, alert generated. Score > 0.9: critical "
                    "anomaly, stop machine and investigate immediately. Confidence >0.85 "
                    "indicates high model certainty."
                ),
                source="manual",
            ),
            Document(
                title="Concept Drift Response Protocol",
                content=(
                    "When PSI drift exceeds 0.2 on any sensor feature, the model's accuracy "
                    "may degrade. Actions: 1) Collect fresh data from the shifted distribution. "
                    "2) Trigger model retraining via Celery worker. 3) Validate retrained model "
                    "with held-out data. 4) Deploy new model version via ModelRegistry. "
                    "5) Reset drift reference baseline."
                ),
                source="sop",
            ),
        ]
        engine.add_documents(demo_docs)
