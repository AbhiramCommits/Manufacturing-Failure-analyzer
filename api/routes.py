from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
import pandas as pd

router = APIRouter()


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
    }
