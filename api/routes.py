import io
from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text

from src.data_loader import DataLoader
from src.analyzer import detect_anomalies_zscore, ClusterEngine
from src.llm_engine import RootCauseAnalyzer
from src.visualizer import generate_cluster_report

router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    file_path: str | None = None
    inline_csv: str | None = None


# ── POST /analyze ────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(request: Request, body: AnalyzeRequest):
    import tempfile, os

    if body.file_path:
        df = DataLoader(body.file_path).load()
    elif body.inline_csv:
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        try:
            os.write(fd, body.inline_csv.encode())
        finally:
            os.close(fd)
        try:
            df = DataLoader(tmp).load()
        finally:
            os.remove(tmp)
    else:
        raise HTTPException(400, "Provide either 'file_path' or 'inline_csv'")

    df = detect_anomalies_zscore(df, threshold=2.5)

    engine = ClusterEngine(df)
    engine.run_kmeans(k=4)
    engine.run_dbscan(eps=0.5, min_samples=5)

    db_url = "sqlite:///data/failures.db"
    engine.save_to_sqlite(db_url)

    kmeans_summary = engine.cluster_summary("cluster_kmeans")
    dbscan_summary = engine.cluster_summary("cluster_dbscan")

    chart_path = generate_cluster_report(
        engine.df, kmeans_summary, label_col="cluster_kmeans"
    )

    llm = RootCauseAnalyzer()
    cluster_results = []
    for _, row in kmeans_summary.iterrows():
        cluster_data = row.to_dict()
        try:
            hypotheses = llm.analyze(cluster_data)
        except Exception as exc:
            hypotheses = {"error": str(exc), "hypotheses": []}
        hypotheses["cluster"] = int(row["cluster"])
        cluster_results.append(hypotheses)

    return {
        "total_records": len(engine.df),
        "anomalies_detected": int(engine.df["is_anomaly"].sum()) if "is_anomaly" in engine.df.columns else 0,
        "kmeans_summary": kmeans_summary.to_dict(orient="records"),
        "dbscan_summary": dbscan_summary.to_dict(orient="records"),
        "chart_path": chart_path,
        "root_cause_analysis": cluster_results,
    }


# ── GET /failures ────────────────────────────────────────────────────────────

@router.get("/failures")
async def get_failures(
    cluster_id: int = Query(None),
    is_anomaly: bool = Query(None),
    test_type: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    db_url = "sqlite:///data/failures.db"
    engine = create_engine(db_url)
    df = pd.read_sql_table("failure_records", engine)

    result = df[df["failure_code"] != 0].copy()

    if cluster_id is not None:
        if "cluster_kmeans" in result.columns:
            result = result[result["cluster_kmeans"] == cluster_id]
    if is_anomaly is not None:
        if "is_anomaly" in result.columns:
            result = result[result["is_anomaly"] == is_anomaly]
    if test_type:
        result = result[result["test_type"] == test_type]

    result = result.head(limit)
    return result.to_dict(orient="records")


# ── GET /health ──────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    db_url = "sqlite:///data/failures.db"
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM failure_records"))
            count = row.scalar()
        return {"status": "healthy", "record_count": count}
    except Exception:
        return {"status": "degraded", "record_count": 0}


# ── GET /summary ─────────────────────────────────────────────────────────────

@router.get("/summary")
async def get_summary(request: Request):
    df: pd.DataFrame = request.app.state.df
    total = len(df)
    failures = int((df["failure_code"] != 0).sum())
    pass_rate = (total - failures) / total * 100

    failure_by_type = (
        df[df["failure_code"] != 0]
        .groupby("failure_code")
        .size()
        .to_dict()
    )

    return {
        "total_tests": total,
        "failures": failures,
        "pass_rate_pct": round(pass_rate, 2),
        "failure_by_code": failure_by_type,
        "test_types": df["test_type"].unique().tolist(),
        "avg_duration_ms": round(df["duration_ms"].mean(), 2),
        "avg_temperature_c": round(df["temperature_c"].mean(), 2),
        "avg_voltage_v": round(df["voltage_v"].mean(), 2),
        "anomalies_detected": int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0,
    }


# ── GET /anomalies ───────────────────────────────────────────────────────────

@router.get("/anomalies")
async def get_anomalies(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
):
    df: pd.DataFrame = request.app.state.df
    if "is_anomaly" not in df.columns:
        return []
    result = df[df["is_anomaly"] == True].head(limit)
    return result.to_dict(orient="records")


# ── GET /clusters/{algorithm} ────────────────────────────────────────────────

@router.get("/clusters/{algorithm}")
async def get_clusters(
    request: Request,
    algorithm: str,
):
    if algorithm == "kmeans":
        summary_df = request.app.state.kmeans_summary
        label_col = "cluster_kmeans"
    elif algorithm == "dbscan":
        summary_df = request.app.state.dbscan_summary
        label_col = "cluster_dbscan"
    else:
        raise HTTPException(400, "Use 'kmeans' or 'dbscan'")

    df: pd.DataFrame = request.app.state.df
    result = summary_df.to_dict(orient="records")

    for entry in result:
        cluster_id = entry["cluster"]
        cluster_df = df[df[label_col] == cluster_id]
        entry["test_ids"] = cluster_df["test_id"].head(10).tolist()

    return result


# ── GET /clusters/{algorithm}/{cluster_id}/analyze ───────────────────────────

@router.get("/clusters/{algorithm}/{cluster_id}/analyze")
async def analyze_root_cause(
    request: Request,
    algorithm: str,
    cluster_id: int,
):
    if algorithm == "kmeans":
        summary_df = request.app.state.kmeans_summary
    elif algorithm == "dbscan":
        summary_df = request.app.state.dbscan_summary
    else:
        raise HTTPException(400, "Use 'kmeans' or 'dbscan'")

    match = summary_df[summary_df["cluster"] == cluster_id]
    if len(match) == 0:
        raise HTTPException(404, f"Cluster {cluster_id} not found")

    cluster_data = match.iloc[0].to_dict()
    analyzer: RootCauseAnalyzer = request.app.state.llm
    result = analyzer.analyze(cluster_data)
    result["cluster"] = cluster_id
    result["algorithm"] = algorithm
    return result
