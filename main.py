import argparse
import json
import sys

from src.data_loader import DataLoader
from src.analyzer import detect_anomalies_zscore, ClusterEngine
from src.llm_engine import RootCauseAnalyzer
from src.visualizer import generate_cluster_report


def cmd_serve(args):
    import uvicorn

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_analyze(args):
    csv_path = args.csv
    print(f"Loading {csv_path} ...")
    df = DataLoader(csv_path).load()
    print(f"  {len(df)} records loaded")

    df = detect_anomalies_zscore(df, threshold=args.threshold)
    anomaly_count = int(df["is_anomaly"].sum())
    print(f"  {anomaly_count} anomalies detected (threshold={args.threshold})")

    engine = ClusterEngine(df)
    engine.run_kmeans(k=args.k)
    engine.run_dbscan(eps=args.eps, min_samples=args.min_samples)

    db_url = "sqlite:///data/failures.db"
    engine.save_to_sqlite(db_url)
    print(f"  Results persisted to {db_url}")

    kmeans_summary = engine.cluster_summary("cluster_kmeans")
    print("\n=== K-Means Cluster Summary ===")
    for _, row in kmeans_summary.iterrows():
        print(
            f"  Cluster {int(row['cluster'])}: "
            f"size={int(row['size'])}, "
            f"dom_fc={int(row['dominant_failure_code'])}, "
            f"mean_dur={row['mean_duration_ms']}ms, "
            f"anom_rate={row['anomaly_rate_pct']}%"
        )
        msgs = row["top_error_messages"]
        if msgs:
            for msg in msgs[:3]:
                print(f"    - {msg}")

    chart_path = generate_cluster_report(
        engine.df, kmeans_summary, label_col="cluster_kmeans"
    )
    print(f"\n  Chart saved to {chart_path}")

    if not args.no_llm:
        print("\n=== Root Cause Analysis (LLM) ===")
        llm = RootCauseAnalyzer()
        for _, row in kmeans_summary.iterrows():
            cluster_data = row.to_dict()
            print(f"\n  Cluster {int(row['cluster'])}:")
            try:
                result = llm.analyze(cluster_data)
                for h in result.get("hypotheses", []):
                    print(
                        f"    [{h['confidence'].upper()}] Rank {h['rank']}: "
                        f"{h['root_cause']}"
                    )
                    print(f"      Triage: {h['triage_action']}")
            except Exception as exc:
                print(f"    LLM error: {exc}")

    if args.output_json:
        output = {
            "total_records": len(engine.df),
            "anomalies_detected": anomaly_count,
            "kmeans_summary": kmeans_summary.to_dict(orient="records"),
            "chart_path": chart_path,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  JSON summary written to {args.output_json}")

    print("\nDone.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfa",
        description="Manufacturing Failure Analyzer",
    )
    sub = parser.add_subparsers(title="commands", dest="command")

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", default=False)
    p_serve.set_defaults(func=cmd_serve)

    # ---- analyze ----
    p_analyze = sub.add_parser("analyze", help="Run analysis pipeline on a CSV file")
    p_analyze.add_argument("csv", help="Path to CSV hardware test log")
    p_analyze.add_argument("--threshold", type=float, default=2.5, help="Z-score anomaly threshold")
    p_analyze.add_argument("--k", type=int, default=4, help="Number of K-Means clusters")
    p_analyze.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon")
    p_analyze.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    p_analyze.add_argument("--no-llm", action="store_true", help="Skip LLM root cause analysis")
    p_analyze.add_argument("--output-json", default=None, help="Save summary to JSON file")
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


# ── FastAPI app (same reference for uvicorn when running "serve") ────────────

import uvicorn as _uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Manufacturing Failure Analyzer",
    description="API for analyzing hardware test logs and detecting failure patterns",
    version="0.1.0",
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    loader = DataLoader("data/sample_test_log.csv")
    df = loader.load()
    df = detect_anomalies_zscore(df, threshold=2.5)

    engine = ClusterEngine(df)
    engine.run_kmeans(k=4)
    engine.run_dbscan(eps=0.5, min_samples=5)
    engine.save_to_sqlite("sqlite:///data/failures.db")

    app.state.df = engine.df
    app.state.kmeans_summary = engine.cluster_summary("cluster_kmeans")
    app.state.dbscan_summary = engine.cluster_summary("cluster_dbscan")
    app.state.llm = RootCauseAnalyzer()

    generate_cluster_report(
        engine.df,
        app.state.kmeans_summary,
        label_col="cluster_kmeans",
        output_path="data/cluster_report.png",
    )


@app.get("/")
async def root():
    return {"message": "Manufacturing Failure Analyzer API"}


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)
