from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel
import pandas as pd

router = APIRouter()


class ClusterAnalysisRequest(BaseModel):
    dominant_failure_code: int
    mean_duration_ms: float
    anomaly_rate_pct: float
    top_error_messages: list[str]
    size: int | None = None
    cluster: int | None = None


@router.get("/failures")
async def get_failures(
    request: Request,
    component_id: str = Query(None),
    test_type: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    df: pd.DataFrame = request.app.state.df
    result = df[df["failure_code"] != 0].copy()

    if component_id:
        result = result[result["component_id"] == component_id]
    if test_type:
        result = result[result["test_type"] == test_type]

    result = result.head(limit)
    return result.to_dict(orient="records")


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
        from fastapi import HTTPException
        raise HTTPException(400, "Use 'kmeans' or 'dbscan'")

    df: pd.DataFrame = request.app.state.df
    result = summary_df.to_dict(orient="records")

    for entry in result:
        cluster_id = entry["cluster"]
        cluster_df = df[df[label_col] == cluster_id]
        entry["test_ids"] = cluster_df["test_id"].head(10).tolist()

    return result


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

    analyzer: "RootCauseAnalyzer" = request.app.state.llm
    result = analyzer.analyze(cluster_data)
    result["cluster"] = cluster_id
    result["algorithm"] = algorithm
    return result


@router.post("/analyze")
async def analyze_custom(
    request: Request,
    body: ClusterAnalysisRequest,
):
    analyzer: "RootCauseAnalyzer" = request.app.state.llm
    return analyzer.analyze(body.model_dump())
