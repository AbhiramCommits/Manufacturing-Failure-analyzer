import uvicorn
from fastapi import FastAPI
from api.routes import router
from src.data_loader import DataLoader

app = FastAPI(
    title="Manufacturing Failure Analyzer",
    description="API for analyzing hardware test logs and detecting failure patterns",
    version="0.1.0",
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    loader = DataLoader("data/sample_test_log.csv")
    app.state.df = loader.load()


@app.get("/")
async def root():
    return {"message": "Manufacturing Failure Analyzer API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
