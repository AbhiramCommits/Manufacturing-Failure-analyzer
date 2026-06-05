import uvicorn
from fastapi import FastAPI
from api.routes import router
from src.data_loader import DataLoader
from src.analyzer import detect_anomalies_zscore, ClusterEngine
from src.llm_engine import RootCauseAnalyzer

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


@app.get("/")
async def root():
    return {"message": "Manufacturing Failure Analyzer API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
