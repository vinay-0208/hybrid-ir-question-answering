"""
Streamlit Web Application for CSI4001 Case Study 2:
Hybrid Transformer-Based Information Retrieval and Question Answering.

Author: Vinay Vishwanathan (Register Number: 23MID0429)
Course: CSI4001 - Natural Language Processing and Computational Linguistics
Case Study: Case Study 2
"""

import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

# Ensure parent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.engine import HybridQAEngine
from app.ingestion import extract_text_from_bytes
from app.metrics import EvaluationRunner

# Streamlit Page Configuration
st.set_page_config(
    page_title="Hybrid IR & QA | 23MID0429",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
SAMPLE_DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")
DATA_DIR = os.path.join(BASE_DIR, "data")

if "engine" not in st.session_state:
    with st.spinner("Initializing Hybrid IR & QA Engine and loading sample knowledge base..."):
        engine = HybridQAEngine(sample_docs_dir=SAMPLE_DOCS_DIR)
        st.session_state.engine = engine

engine: HybridQAEngine = st.session_state.engine

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.shields.io/badge/CSI4001_NLP-Case_Study_2-blue?style=for-the-badge", use_container_width=True)
    st.title("Project Details")
    st.markdown(
        """
        - **Student:** Vinay Vishwanathan
        - **Reg. No:** `23MID0429`
        - **Course:** CSI4001 NLP
        - **Case Study:** Case Study 2
        - **Topic:** Hybrid IR & Question Answering
        """
    )
    st.divider()

    st.subheader("⚙️ System Control")
    force_offline = st.checkbox(
        "Simulate Lexical-Only / Offline Mode",
        value=engine.force_offline_mode,
        help="Disables Transformer models to demonstrate robust BM25 fallback behavior."
    )
    if force_offline != engine.force_offline_mode:
        engine.set_offline_mode(force_offline)
        st.rerun()

    st.divider()
    status = engine.get_system_status()
    st.subheader("🟢 Pipeline Component Status")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Documents", status["document_count"])
    with c2:
        st.metric("Passages", status["passage_count"])

    st.markdown(f"**BM25 Lexical:** {'✅ Active' if status['bm25_active'] else '❌ Inactive'}")
    st.markdown(f"**Dense Retrieval:** {'✅ Active' if status['dense_active'] else '⚠️ Offline / Fallback'}")
    st.markdown(f"**FAISS ANN Index:** {'✅ ' + status['faiss_index_type'] if status.get('faiss_active') else '⚠️ NumPy Fallback'}")
    st.markdown(f"**ReAct Rewriter:** {'✅ Heuristic (2-Step)' if status.get('agentic_rewriter_active') else '❌ Inactive'}")
    st.markdown(f"**HyDE Synthesis:** {'✅ Template Active' if status.get('hyde_active') else '⚠️ Offline'}")
    st.markdown(f"**Cross-Encoder:** {'✅ Active' if status['rerank_active'] else '⚠️ Offline / Fallback'}")
    st.markdown(f"**QA & Synthesis:** {'✅ DistilBERT + Extractive Citations' if status['qa_active'] else '⚠️ Sentence Fallback'}")
    st.markdown(f"**RAGAS Proxies:** {'✅ Heuristic Active' if status.get('ragas_active') else '❌ Inactive'}")
    st.caption(f"Compute Device: `{status['device'].upper()}`")

# Header Section
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); padding: 22px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <h2 style="margin: 0; padding: 0; color: white;">Hybrid Transformer-Based Information Retrieval and Question Answering</h2>
        <p style="margin: 6px 0 0 0; font-size: 15px; opacity: 0.95;">
            CSI4001: Natural Language Processing and Computational Linguistics | Case Study 2<br>
            <strong>Vinay Vishwanathan &bull; Register Number: 23MID0429</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Tabs
tab_query, tab_compare, tab_ingest, tab_eval, tab_arch = st.tabs([
    "🔍 Ask a Question",
    "⚖️ Side-by-Side Comparison",
    "📚 Document Ingestion & Corpus",
    "📊 Evaluation Dashboard",
    "🏛️ Architecture & Documentation"
])

# =========================================================================
# TAB 1: ASK A QUESTION
# =========================================================================
with tab_query:
    st.subheader("Query Execution, Heuristic ReAct, HyDE & Extractive Evidence Synthesis")
    st.markdown("Enter a natural language question to execute the retrieval, fusion, reranking, multi-passage extractive synthesis, and RAGAS-inspired proxy evaluation pipeline.")

    # Quick sample query pills
    sample_queries = [
        "What is the Okapi BM25 algorithm?",
        "What problem does dense semantic retrieval solve?",
        "How does Reciprocal Rank Fusion calculate the score?",
        "Why is a two-stage retrieval pipeline used with cross-encoders?",
        "What is the capital of planet Mars?"  # Negative test for abstention
    ]

    selected_sample = st.selectbox("💡 Or choose a sample question to test:", ["-- Custom Input --"] + sample_queries)

    default_q = "" if selected_sample == "-- Custom Input --" else selected_sample
    query_text = st.text_input("Enter Question:", value=default_q, placeholder="e.g., What is the role of parameter k1 in BM25?")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        mode_choice = st.selectbox(
            "Select Execution Pipeline Mode:",
            [
                ("agentic_hyde_rerank", "1. Heuristic ReAct + HyDE + FAISS + Cross-Encoder (Enhanced Pipeline)"),
                ("hybrid_rerank", "2. Hybrid Retrieval + Cross-Encoder Reranking"),
                ("hybrid_rrf", "3. Hybrid Retrieval (BM25 + Dense RRF Fusion)"),
                ("dense_only", "4. Dense Semantic Retrieval Only (FAISS HNSW + all-MiniLM-L6-v2)"),
                ("bm25_only", "5. Lexical Retrieval Only (Okapi BM25)")
            ],
            format_func=lambda x: x[1]
        )[0]
    with col_m2:
        top_k = st.slider("Top-k Passages", min_value=1, max_value=10, value=5)

    opt_c1, opt_c2 = st.columns(2)
    with opt_c1:
        enable_react = st.checkbox(
            "Enable Heuristic ReAct Query Rewriting",
            value=(mode_choice == "agentic_hyde_rerank"),
            help="Executes deterministic 2-step Reason+Act terminology expansion and retrieval fusion."
        )
    with opt_c2:
        enable_hyde = st.checkbox(
            "Enable Hypothetical Document Embeddings (HyDE)",
            value=(mode_choice == "agentic_hyde_rerank"),
            help="Synthesizes a template-based declarative document and blends its embedding for FAISS dense search."
        )

    if st.button("🚀 Search & Answer", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Please enter a question.")
        elif status["passage_count"] == 0:
            st.error("Knowledge base is empty. Please load sample documents or upload files in the Ingestion tab.")
        else:
            with st.spinner("Executing hybrid retrieval, extractive evidence synthesis, and RAGAS proxy evaluation..."):
                res = engine.execute_query(
                    query=query_text,
                    mode=mode_choice,
                    top_k=top_k,
                    use_agentic_rewrite=enable_react,
                    use_hyde=enable_hyde
                )

            # Display Answer Banner
            has_evidence = res.get("has_sufficient_evidence", False)
            answer_text = res.get("answer", "")
            synth_answer = res.get("synthesized_answer", answer_text)
            citations = res.get("citations", [])
            confidence = res.get("confidence_score", 0.0)
            method = res.get("answering_method", "")
            source_doc = res.get("source_doc", "")
            passage_id = res.get("passage_id", "")
            evidence_text = res.get("evidence", "")

            if has_evidence:
                st.markdown(
                    f"""
                    <div style="background-color: #f0fdf4; border-left: 6px solid #22c55e; padding: 18px; border-radius: 8px; margin: 15px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: bold; color: #15803d; font-size: 16px;">💡 Multi-Passage Extractive Synthesized Answer & Span</span>
                            <div>
                                <span style="background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;">
                                    Confidence: {confidence:.2f}
                                </span>
                                <span style="background-color: #dbeafe; color: #1e40af; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px;">
                                    Vector Index: {res.get('faiss_index_type', 'FAISS')}
                                </span>
                                <span style="background-color: #e0e7ff; color: #3730a3; padding: 3px 8px; border-radius: 4px; font-size: 12px;">
                                    {method}
                                </span>
                            </div>
                        </div>
                        <p style="font-size: 16px; color: #0f172a; margin: 0 0 10px 0; font-weight: 600;">{synth_answer}</p>
                        <div style="font-size: 13px; color: #475569; background-color: #ffffff; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0;">
                            <strong>🎯 Concise Extractive Span:</strong> <code>{answer_text}</code><br>
                            <strong>📌 Primary Evidence Sentence:</strong> <em>"{evidence_text}"</em><br>
                            <strong>📄 Primary Provenance:</strong> <code>{source_doc}</code> (Passage ID: <code>{passage_id}</code>)<br>
                            <strong>🛡️ Verification Status:</strong> {res.get('verification_status')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 18px; border-radius: 8px; margin: 15px 0;">
                        <h4 style="margin: 0 0 6px 0; color: #991b1b;">⚠️ Insufficient Evidence / Abstention Policy Triggered</h4>
                        <p style="margin: 0; color: #7f1d1d; font-size: 15px;">{answer_text}</p>
                        <small style="color: #991b1b;">Reason: {res.get('verification_status')} (Rule: {res.get('verification_rule')})</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Live RAGAS-Inspired Heuristic Proxy Metrics
            ragas = res.get("ragas_metrics", {})
            if ragas:
                st.markdown("##### 📐 Live RAGAS-Inspired Heuristic Proxy Metrics")
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                r_c1.metric("Faithfulness Proxy (Groundedness)", f"{ragas.get('faithfulness', 0.0):.3f}")
                r_c2.metric("Answer Relevance Proxy", f"{ragas.get('answer_relevance', 0.0):.3f}")
                r_c3.metric("Context Precision Proxy", f"{ragas.get('context_precision', 0.0):.3f}")
                r_c4.metric("Context Recall / Utilization Proxy", f"{ragas.get('context_recall', 0.0):.3f}")

            # Heuristic ReAct Trace & HyDE Inspection
            with st.expander("🧠 View Heuristic ReAct Workflow Trace & HyDE Hypothetical Document", expanded=True):
                st.markdown(f"**Rewritten / Expanded Query:** `{res.get('rewritten_query', query_text)}`")
                react_trace = res.get("react_trace", [])
                if react_trace:
                    st.markdown("**Deterministic ReAct Workflow Trace:**")
                    st.dataframe(pd.DataFrame(react_trace), use_container_width=True)
                else:
                    st.caption("Heuristic ReAct query rewriting is disabled for this execution.")
                hyde_doc = res.get("hypothetical_document", "")
                if hyde_doc:
                    st.markdown("**Synthesized HyDE Hypothetical Document ($\hat{d}$):**")
                    st.info(hyde_doc)
                else:
                    st.caption("HyDE hypothetical document generation is disabled for this execution.")
                if citations:
                    st.markdown("**Inline Source Citations:**")
                    st.dataframe(pd.DataFrame(citations), use_container_width=True)

            # Operational Latency Metrics
            st.markdown("##### ⏱️ Operational Latency Breakdown")
            l_col1, l_col2, l_col3, l_col4, l_col5 = st.columns(5)
            lats = res.get("latencies_ms", {})
            l_col1.metric("Retrieval (FAISS+BM25)", f"{lats.get('retrieval_ms', 0):.1f} ms")
            l_col2.metric("Fusion (RRF)", f"{lats.get('fusion_ms', 0):.1f} ms")
            l_col3.metric("Reranking", f"{lats.get('reranking_ms', 0):.1f} ms")
            l_col4.metric("QA & Synthesis", f"{lats.get('qa_ms', 0):.1f} ms")
            l_col5.metric("Total Latency", f"{lats.get('total_ms', 0):.1f} ms")

            # Retrieved Candidates Table
            st.markdown("##### 📑 Retrieved Candidate Passages & Multi-Stage Scores")
            candidates = res.get("candidates", [])
            if candidates:
                cand_rows = []
                for idx, c in enumerate(candidates, start=1):
                    cand_rows.append({
                        "Final Rank": idx,
                        "Doc Name": c.get("doc_name", ""),
                        "Passage ID": c.get("passage_id", ""),
                        "Cross-Encoder Score": round(c.get("rerank_score", 0.0), 3) if c.get("rerank_score") is not None else "N/A",
                        "RRF Score": round(c.get("rrf_score", 0.0), 4) if c.get("rrf_score") is not None else "N/A",
                        "BM25 Score": round(c.get("bm25_score", 0.0), 2) if c.get("bm25_score") is not None else "N/A",
                        "Dense Score (Cosine)": round(c.get("dense_score", 0.0), 3) if c.get("dense_score") is not None else "N/A",
                        "Passage Text Excerpt": c.get("text", "")[:130] + "..."
                    })
                df_cands = pd.DataFrame(cand_rows)
                st.dataframe(df_cands, use_container_width=True)

                # Expandable passage details
                with st.expander("🔍 View Full Passage Texts"):
                    for idx, c in enumerate(candidates, start=1):
                        st.markdown(f"**Candidate #{idx} - {c.get('doc_name')} ({c.get('passage_id')})**")
                        st.info(c.get("text", ""))

            # Export Query Results
            st.markdown("##### 💾 Export Results")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                json_str = engine.export_query_result_json(res)
                st.download_button(
                    label="📥 Download Query Execution (JSON)",
                    data=json_str,
                    file_name=f"query_result_23MID0429_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with exp_c2:
                csv_str = engine.export_candidates_csv(candidates)
                st.download_button(
                    label="📥 Download Retrieval Rankings (CSV)",
                    data=csv_str,
                    file_name=f"retrieval_ranking_23MID0429_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =========================================================================
# TAB 2: SIDE-BY-SIDE COMPARISON
# =========================================================================
with tab_compare:
    st.subheader("Comparative Analysis Across All 5 Retrieval Modalities")
    st.markdown("Execute the same query simultaneously across all five retrieval modalities to evaluate ranking changes, extractive citation synthesis, and RAGAS-inspired proxy scores.")

    comp_query = st.text_input(
        "Comparison Query:",
        value="What problem does dense semantic retrieval solve that BM25 struggles with?",
        key="comp_query_input"
    )

    if st.button("⚖️ Run 5-Way Comparison", type="primary"):
        if not comp_query.strip():
            st.warning("Please enter a comparison query.")
        else:
            with st.spinner("Running all 5 retrieval modes..."):
                comparisons = engine.compare_all_modes(comp_query, top_k=5)

            # Comparison Summary Table
            comp_table = []
            for m_key, m_name in [
                ("bm25_only", "1. BM25 Lexical"),
                ("dense_only", "2. Dense Semantic (FAISS)"),
                ("hybrid_rrf", "3. Hybrid RRF"),
                ("hybrid_rerank", "4. Hybrid + Cross-Encoder"),
                ("agentic_hyde_rerank", "5. Heuristic ReAct + HyDE + Rerank")
            ]:
                r = comparisons[m_key]
                top_cand = r["candidates"][0] if r["candidates"] else {}
                ragas = r.get("ragas_metrics", {})
                comp_table.append({
                    "Mode": m_name,
                    "Top-1 Passage ID": top_cand.get("passage_id", "None"),
                    "Confidence": r.get("confidence_score", 0.0),
                    "Faithfulness Proxy": ragas.get("faithfulness", 0.0),
                    "Answer Relevance Proxy": ragas.get("answer_relevance", 0.0),
                    "Total Latency (ms)": r["latencies_ms"].get("total_ms", 0.0),
                    "Synthesized Answer": r.get("synthesized_answer", r.get("answer", ""))[:140] + "..."
                })
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True)

# =========================================================================
# TAB 3: DOCUMENT INGESTION & CORPUS MANAGEMENT
# =========================================================================
with tab_ingest:
    st.subheader("Corpus Management & Passage Segmentation")
    st.markdown("Upload documents, configure sliding-window passage segmentation, and inspect indexed chunks.")

    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        st.markdown("##### 📤 Upload New Documents (.txt, .pdf)")
        uploaded_files = st.file_uploader(
            "Select text or PDF files:",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("📥 Process & Ingest Uploaded Files"):
                added_count = 0
                for up_file in uploaded_files:
                    file_bytes = up_file.read()
                    text, err = extract_text_from_bytes(file_bytes, up_file.name)
                    if err:
                        st.error(f"Error reading `{up_file.name}`: {err}")
                    else:
                        ok, msg = engine.repository.add_document(up_file.name, text)
                        if ok:
                            added_count += 1
                if added_count > 0:
                    engine.rebuild_indexes()
                    st.success(f"Successfully ingested and indexed {added_count} new file(s)!")
                    st.rerun()

    with col_up2:
        st.markdown("##### ⚙️ Segmentation Parameters")
        new_chunk_size = st.slider("Chunk Size (Words)", min_value=50, max_value=300, value=engine.repository.chunk_size, step=10)
        new_overlap = st.slider("Chunk Overlap (Words)", min_value=0, max_value=100, value=engine.repository.overlap, step=5)
        if st.button("🔄 Rebuild All Indexes with New Segmentation"):
            engine.repository.set_chunk_parameters(new_chunk_size, new_overlap)
            engine.rebuild_indexes()
            st.success(f"Indexes rebuilt! Total passages: {engine.repository.get_passage_count()}")
            st.rerun()

    st.divider()

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.markdown("##### 📁 Ingested Documents")
        doc_names = engine.repository.get_document_names()
        if doc_names:
            doc_rows = []
            for d in doc_names:
                content = engine.repository.documents[d]
                doc_rows.append({
                    "Filename": d,
                    "Words": len(content.split()),
                    "Characters": len(content)
                })
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

            del_target = st.selectbox("Select document to delete:", doc_names)
            if st.button("🗑️ Delete Selected Document"):
                engine.repository.remove_document(del_target)
                engine.rebuild_indexes()
                st.success(f"Deleted `{del_target}` and updated index.")
                st.rerun()
        else:
            st.warning("No documents in repository.")
            if st.button("Load Built-in Sample Documents"):
                engine.load_sample_documents(SAMPLE_DOCS_DIR)
                st.success("Loaded sample documents!")
                st.rerun()

    with col_c2:
        st.markdown("##### 📑 Segmented Passages Preview")
        passages = engine.repository.passages
        if passages:
            st.write(f"**Total Indexed Passages:** {len(passages)}")
            p_preview = []
            for p in passages[:8]:
                p_preview.append({
                    "Passage ID": p["passage_id"],
                    "Source File": p["doc_name"],
                    "Word Count": p["word_count"],
                    "Snippet": p["text"][:80] + "..."
                })
            st.dataframe(pd.DataFrame(p_preview), use_container_width=True)
        else:
            st.info("No passages indexed yet.")

# =========================================================================
# TAB 4: EVALUATION DASHBOARD
# =========================================================================
with tab_eval:
    st.subheader("Evaluation Dashboard: Classic IR/QA & RAGAS-Inspired Heuristic Proxy Metrics")
    st.markdown(
        """
        Evaluate and compare retrieval and extractive answer quality across all 5 modes on the labelled benchmark dataset.
        Metrics computed: **Precision@k**, **Recall@k**, **MRR**, **NDCG@k**, **Exact Match (EM)**, **Token F1**,
        and **RAGAS-Inspired Heuristic Proxies** (**Faithfulness**, **Answer Relevance**, **Context Precision**, **Context Recall**) plus **Latency** (with model initialization excluded).
        """
    )

    eval_file_path = os.path.join(DATA_DIR, "evaluation_dataset.json")
    benchmark_data = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    with st.expander("📋 Inspect Labelled Benchmark Dataset (10 Test Queries)"):
        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

    k_eval = st.slider("Evaluation k threshold (Precision@k, Recall@k, NDCG@k):", min_value=1, max_value=10, value=5)

    saved_csv_path = os.path.join(BASE_DIR, "reports", "benchmark_summary.csv")
    if os.path.exists(saved_csv_path) and "eval_results" not in st.session_state:
        st.markdown("#### 🏆 Latest Verified Benchmark Summary (`k=5`)")
        saved_df = pd.read_csv(saved_csv_path)
        st.dataframe(saved_df, use_container_width=True)

    if st.button("▶️ Run Automated Benchmark Evaluation (All 5 Modes)", type="primary"):
        if not benchmark_data:
            st.error("Evaluation dataset is empty.")
        elif engine.repository.get_passage_count() == 0:
            st.error("Knowledge base is empty. Please load documents before evaluating.")
        else:
            with st.spinner("Executing benchmark across BM25, Dense FAISS, Hybrid RRF, Cross-Encoder, and Heuristic ReAct + HyDE..."):
                runner = EvaluationRunner(engine)
                eval_results = runner.run_benchmark(benchmark_data, top_k=k_eval)
                st.session_state.eval_results = eval_results

            summary_df = eval_results["summary_df"]
            st.markdown("#### 🏆 Live Benchmark Evaluation Results")
            st.dataframe(summary_df, use_container_width=True)

            # Visual Comparison Charts
            st.markdown("#### 📈 Visual Performance Comparison")
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            with chart_col1:
                st.markdown(f"##### Precision@{k_eval} & Recall@{k_eval}")
                chart_df = summary_df.set_index("Retrieval Mode")[[f"Precision@{k_eval}", f"Recall@{k_eval}"]]
                st.bar_chart(chart_df)
            with chart_col2:
                st.markdown(f"##### Ranking: MRR & NDCG@{k_eval}")
                chart_rank = summary_df.set_index("Retrieval Mode")[["MRR", f"NDCG@{k_eval}"]]
                st.bar_chart(chart_rank)
            with chart_col3:
                st.markdown("##### RAGAS-Inspired Proxy Metrics")
                if "Faithfulness" in summary_df.columns:
                    chart_ragas = summary_df.set_index("Retrieval Mode")[["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall"]]
                    st.bar_chart(chart_ragas)

            # Detailed results per question
            with st.expander("🔎 View Query-by-Query Evaluation Breakdown"):
                det = eval_results["detailed_results"]
                for mode_key, rows in det.items():
                    st.markdown(f"**Mode: {mode_key}**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Export evaluation report
            eval_json = json.dumps({
                "student": "Vinay Vishwanathan (23MID0429)",
                "course": "CSI4001 NLP Case Study 2",
                "k_threshold": k_eval,
                "summary": summary_df.to_dict(orient="records"),
                "details": eval_results["detailed_results"]
            }, indent=2)

            st.download_button(
                label="📥 Download Full Benchmark Evaluation Report (JSON)",
                data=eval_json,
                file_name=f"evaluation_report_23MID0429_{int(time.time())}.json",
                mime="application/json"
            )

# =========================================================================
# TAB 5: ARCHITECTURE & DOCUMENTATION
# =========================================================================
with tab_arch:
    st.subheader("System Architecture & Theoretical Formulation")
    st.markdown(
        """
        The project implements a **multi-stage hybrid retrieval architecture with deterministic heuristic ReAct query rewriting,
        template-based HyDE, FAISS ANN vector search, cross-encoder reranking, multi-passage extractive answer synthesis with citations,
        and RAGAS-inspired heuristic evaluation proxies**.
        """
    )

    st.markdown(
        """
        ### 🔄 Multi-Stage Architecture Flow
        
        ```text
        [OFFLINE STAGE]
        Document Collection (.txt, .pdf)
               │
               ▼
        Text Extraction & Normalization
               │
               ▼
        Sliding-Window Passage Segmentation (Chunk Size=150, Overlap=30)
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
        Inverted Index (BM25Okapi)              FAISS ANN Vector Index (IndexHNSWFlat)
                                                (all-MiniLM-L6-v2, 384-d L2 Normalized)
        
        ─────────────────────────────────────────────────────────────────────────────
        
        [ONLINE STAGE]
        User Query: "What is BM25?"
               │
               ▼
        Stage 0: Deterministic 2-Step ReAct Query Rewriting & Template HyDE Synthesis
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
        BM25 Candidate Retrieval              FAISS HNSW Dense + HyDE Retrieval
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                         Candidate Fusion (RRF: k=60)
                                    │
                                    ▼
                 Stage 2: Cross-Encoder Reranking
                 (cross-encoder/ms-marco-MiniLM-L-6-v2)
                                    │
                                    ▼
                         Precision Reordered Passages
                                    │
                                    ▼
                 Extractive QA (distilbert-base-cased-distilled-squad)
                 + Multi-Passage Extractive Evidence Synthesis [1: doc_p0]
                                    │
                                    ▼
                 RAGAS-Inspired Proxy Evaluation (Faithfulness, Relevance, Context P/R)
        ```
        """
    )

    st.markdown(
        """
        ### 📐 Mathematical Formulations & Justifications
        
        #### 1. Deterministic ReAct Query Rewriting & HyDE Blending
        $$v_{\\text{HyDE}} = \\frac{\\alpha v_q + (1 - \\alpha) v_{\\hat{d}}}{\\|\\alpha v_q + (1 - \\alpha) v_{\\hat{d}}\\|_2}, \\quad \\alpha = 0.6$$
        - **Heuristic ReAct (`Thought` $\\rightarrow$ `Action` $\\rightarrow$ `Observation`):** Diagnoses domain terminology and expands acronyms (`BM25`, `RRF`, `MRR`, `NDCG`, `FAISS`) within a deterministic 2-step workflow (without requiring an autonomous LLM agent).
        - **Template HyDE:** Bridges the query-document stylistic asymmetry by synthesizing a declarative technical statement $\\hat{d}$ and blending its normalized embedding with the query vector.

        #### 2. Okapi BM25 & FAISS HNSW ANN Search
        $$\\text{Score}(D, Q) = \\sum_{t \\in Q} \\text{IDF}(t) \\cdot \\frac{f(t, D) \\cdot (k_1 + 1)}{f(t, D) + k_1 \\cdot \\left(1 - b + b \\cdot \\frac{|D|}{\\text{avgdl}}\\right)}$$
        - **FAISS `IndexHNSWFlat`:** Hierarchical Navigable Small World graph over 384-dimensional L2-normalized vectors using `METRIC_INNER_PRODUCT` (mathematically equivalent to Cosine Similarity).

        #### 3. Reciprocal Rank Fusion (RRF)
        $$\\text{RRF\\_Score}(d) = \\sum_{m \\in M} \\frac{1}{k_{\\text{rrf}} + r_m(d)}, \\quad k_{\\text{rrf}} = 60$$

        #### 4. RAGAS-Inspired Heuristic Proxy Metrics
        - **Faithfulness Proxy:** Sentence-level groundedness and content-token verification in retrieved contexts with negation-inversion penalties.
        - **Answer Relevance Proxy:** $0.65 \\cdot \\max(0, \\cos(v_q, v_a)) + 0.35 \\cdot \\text{QueryTermCoverage}$.
        - **Context Precision@k:** Average precision of unique relevant passages within top-$k$ retrieved ranks.
        - **Context Recall Proxy:** Proportion of reference ground-truth content tokens covered by the retrieved context passages.
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 13px;">
        CSI4001 Natural Language Processing and Computational Linguistics &bull; Case Study 2<br>
        Developed by <strong>Vinay Vishwanathan (23MID0429)</strong>
    </div>
    """,
    unsafe_allow_html=True
)
