"""
Script to generate the system architecture diagram as a high-resolution PNG image.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram(output_paths):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title Banner
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    title_box = patches.FancyBboxPatch(
        (0.5, 9.1), 13, 0.75,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#1e3a8a", facecolor="#1e3a8a"
    )
    ax.add_patch(title_box)
    ax.text(7.0, 9.47, "CSI4001: Hybrid Transformer-Based Information Retrieval and Question Answering",
            color="white", fontsize=14.5, weight="bold", ha="center", va="center")
    ax.text(7.0, 9.22, "Student: Vinay Vishwanathan | Register No: 23MID0429 | Case Study: Case Study 2",
            color="#93c5fd", fontsize=10.5, ha="center", va="center")

    # Section 1: Offline Ingestion & Indexing Stage
    offline_bg = patches.FancyBboxPatch(
        (0.5, 5.2), 13, 3.65,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        edgecolor="#cbd5e1", facecolor="#ffffff", linewidth=1.5
    )
    ax.add_patch(offline_bg)
    ax.text(0.8, 8.55, "OFFLINE STAGE: Ingestion, Segmentation & Multi-Modal Indexing",
            color="#0f172a", fontsize=12, weight="bold", va="center")

    # Offline nodes
    # Node 1: Documents
    n1 = patches.FancyBboxPatch((0.8, 6.7), 2.2, 1.4, boxstyle="round,pad=0.1", edgecolor="#0284c7", facecolor="#e0f2fe", lw=1.5)
    ax.add_patch(n1)
    ax.text(1.9, 7.55, "Corpus Collection", color="#0369a1", fontsize=10.5, weight="bold", ha="center")
    ax.text(1.9, 7.15, "TXT & PDF Documents\nDecoded & Cleaned", color="#0f172a", fontsize=9, ha="center")

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.5, 7.4), xytext=(3.0, 7.4), arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2))

    # Node 2: Passage Chunking
    n2 = patches.FancyBboxPatch((3.5, 6.7), 2.5, 1.4, boxstyle="round,pad=0.1", edgecolor="#0284c7", facecolor="#e0f2fe", lw=1.5)
    ax.add_patch(n2)
    ax.text(4.75, 7.55, "Passage Segmentation", color="#0369a1", fontsize=10.5, weight="bold", ha="center")
    ax.text(4.75, 7.15, "Sliding Window\nChunk: 150w, Overlap: 30w", color="#0f172a", fontsize=9, ha="center")

    # Arrow to Index 1 (BM25) and Index 2 (Dense)
    ax.annotate("", xy=(6.6, 7.8), xytext=(6.0, 7.5), arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
    ax.annotate("", xy=(6.6, 6.2), xytext=(6.0, 7.3), arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2))

    # Node 3: BM25 Inverted Index
    n3 = patches.FancyBboxPatch((6.6, 7.1), 3.2, 1.3, boxstyle="round,pad=0.1", edgecolor="#059669", facecolor="#d1fae5", lw=1.5)
    ax.add_patch(n3)
    ax.text(8.2, 7.85, "Okapi BM25 Index", color="#047857", fontsize=10.5, weight="bold", ha="center")
    ax.text(8.2, 7.4, "Lexical Inverted Index\nk1=1.5, b=0.75 Saturation", color="#065f46", fontsize=8.5, ha="center")

    # Node 4: Dense Vector Index
    n4 = patches.FancyBboxPatch((6.6, 5.5), 3.2, 1.3, boxstyle="round,pad=0.1", edgecolor="#7c3aed", facecolor="#ede9fe", lw=1.5)
    ax.add_patch(n4)
    ax.text(8.2, 6.25, "Dense Vector Index", color="#6d28d9", fontsize=10.5, weight="bold", ha="center")
    ax.text(8.2, 5.8, "all-MiniLM-L6-v2 Bi-Encoder\n384-dim L2-Normalized", color="#5b21b6", fontsize=8.5, ha="center")

    # Node 5: Repository Storage
    n5 = patches.FancyBboxPatch((10.4, 5.9), 2.8, 2.0, boxstyle="round,pad=0.1", edgecolor="#475569", facecolor="#f1f5f9", lw=1.5)
    ax.add_patch(n5)
    ax.text(11.8, 7.5, "Index Cache & Store", color="#334155", fontsize=10.5, weight="bold", ha="center")
    ax.text(11.8, 6.8, "In-Memory Passages\nTerm Frequencies\nVector Embeddings Cache", color="#475569", fontsize=8.5, ha="center")
    ax.annotate("", xy=(10.4, 7.6), xytext=(9.8, 7.6), arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
    ax.annotate("", xy=(10.4, 6.3), xytext=(9.8, 6.3), arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2))

    # Section 2: Online Query Execution, Fusion, Reranking & QA
    online_bg = patches.FancyBboxPatch(
        (0.5, 0.4), 13, 4.5,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        edgecolor="#cbd5e1", facecolor="#ffffff", linewidth=1.5
    )
    ax.add_patch(online_bg)
    ax.text(0.8, 4.55, "ONLINE STAGE: Query Processing, Candidate Fusion, Reranking & QA",
            color="#0f172a", fontsize=12, weight="bold", va="center")

    # Query Node
    nq = patches.FancyBboxPatch((0.8, 2.4), 2.1, 1.6, boxstyle="round,pad=0.1", edgecolor="#2563eb", facecolor="#dbeafe", lw=1.5)
    ax.add_patch(nq)
    ax.text(1.85, 3.5, "User Query", color="#1d4ed8", fontsize=11, weight="bold", ha="center")
    ax.text(1.85, 2.9, "Natural Language\nQuery Normalization", color="#1e3a8a", fontsize=9, ha="center")

    # Dual Retrieval Arrows
    ax.annotate("", xy=(3.4, 3.6), xytext=(2.9, 3.3), arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
    ax.annotate("", xy=(3.4, 2.5), xytext=(2.9, 2.9), arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2))

    # Stage 1 Retrieval Nodes
    nbm = patches.FancyBboxPatch((3.4, 3.2), 2.1, 1.0, boxstyle="round,pad=0.1", edgecolor="#059669", facecolor="#d1fae5", lw=1.5)
    ax.add_patch(nbm)
    ax.text(4.45, 3.8, "BM25 Scoring", color="#047857", fontsize=9.5, weight="bold", ha="center")
    ax.text(4.45, 3.45, "Lexical Top-k", color="#065f46", fontsize=8.5, ha="center")

    nde = patches.FancyBboxPatch((3.4, 1.8), 2.1, 1.0, boxstyle="round,pad=0.1", edgecolor="#7c3aed", facecolor="#ede9fe", lw=1.5)
    ax.add_patch(nde)
    ax.text(4.45, 2.4, "Dense Embedding", color="#6d28d9", fontsize=9.5, weight="bold", ha="center")
    ax.text(4.45, 2.05, "Cosine Top-k", color="#5b21b6", fontsize=8.5, ha="center")

    # Fusion Arrows
    ax.annotate("", xy=(5.9, 2.9), xytext=(5.5, 3.5), arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
    ax.annotate("", xy=(5.9, 2.7), xytext=(5.5, 2.3), arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2))

    # Candidate Fusion Node
    nfus = patches.FancyBboxPatch((5.9, 2.1), 2.1, 1.6, boxstyle="round,pad=0.1", edgecolor="#ea580c", facecolor="#ffedd5", lw=1.5)
    ax.add_patch(nfus)
    ax.text(6.95, 3.3, "Candidate Fusion", color="#c2410c", fontsize=10, weight="bold", ha="center")
    ax.text(6.95, 2.65, "Reciprocal Rank\nFusion (RRF k=60)\nSolves Scale Disparity", color="#9a3412", fontsize=8.5, ha="center")

    # Arrow Fusion -> Reranker
    ax.annotate("", xy=(8.4, 2.9), xytext=(8.0, 2.9), arrowprops=dict(arrowstyle="->", color="#ea580c", lw=2))

    # Stage 2 Cross-Encoder Reranker Node
    nrer = patches.FancyBboxPatch((8.4, 2.1), 2.3, 1.6, boxstyle="round,pad=0.1", edgecolor="#dc2626", facecolor="#fee2e2", lw=1.5)
    ax.add_patch(nrer)
    ax.text(9.55, 3.3, "Cross-Encoder Rerank", color="#b91c1c", fontsize=10, weight="bold", ha="center")
    ax.text(9.55, 2.65, "ms-marco-MiniLM-L-6-v2\nFull Cross-Attention\nPrecision Re-ordering", color="#991b1b", fontsize=8.5, ha="center")

    # Arrow Rerank -> QA
    ax.annotate("", xy=(11.1, 2.9), xytext=(10.7, 2.9), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))

    # Question Answering & Evidence Grounding Node
    nqa = patches.FancyBboxPatch((11.1, 1.2), 2.2, 3.0, boxstyle="round,pad=0.1", edgecolor="#16a34a", facecolor="#dcfce7", lw=1.5)
    ax.add_patch(nqa)
    ax.text(12.2, 3.8, "Evidence Grounding & QA", color="#15803d", fontsize=9.5, weight="bold", ha="center")
    ax.text(12.2, 2.8, "• Sentence Evidence\n• DistilBERT SQuAD\n• Fallback Heuristic\n• Abstention Check\n• Provenance Citation", color="#166534", fontsize=8.2, ha="center")
    ax.text(12.2, 1.5, "Final Answer Output", color="#14532d", fontsize=9, weight="bold", ha="center")

    plt.tight_layout()
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved diagram to: {path}")
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_paths = [
        os.path.join(base_dir, "reports", "architecture_diagram.png"),
        os.path.join(base_dir, "assets", "architecture_diagram.png")
    ]
    draw_architecture_diagram(out_paths)
