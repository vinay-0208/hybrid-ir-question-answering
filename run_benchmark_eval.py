"""
Script to execute the 10-query benchmark across all 4 retrieval modes,
calculating exact Precision@k, Recall@k, MRR, NDCG@k, Exact Match, Token F1, and Latency.
"""

import os
import json
import pandas as pd
from app.engine import HybridQAEngine
from app.metrics import EvaluationRunner

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_docs_dir = os.path.join(base_dir, "sample_docs")
    data_file = os.path.join(base_dir, "data", "evaluation_dataset.json")

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Initializing HybridQAEngine...")
    engine = HybridQAEngine(sample_docs_dir=sample_docs_dir)

    print(f"Running benchmark on {len(dataset)} queries across all 5 modes...")
    runner = EvaluationRunner(engine)
    results = runner.run_benchmark(dataset=dataset, top_k=5)

    summary_df = results["summary_df"]
    print("\nBenchmark Summary Results (k=5):")
    print(summary_df.to_string(index=False))

    # Save to reports
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    csv_path = os.path.join(reports_dir, "benchmark_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV to: {csv_path}")

    json_path = os.path.join(reports_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_df.to_dict(orient="records"),
            "details": results["detailed_results"]
        }, f, indent=2)
    print(f"Saved summary JSON to: {json_path}")

if __name__ == "__main__":
    main()
