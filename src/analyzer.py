import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine


def detect_anomalies_zscore(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    df = df.copy()
    for col in ["duration_ms", "temperature_c"]:
        col_data = df[col].values
        mean = np.mean(col_data)
        std = np.std(col_data)
        z_scores = np.abs((col_data - mean) / std)
        df[f"{col}_zscore"] = z_scores

    df["is_anomaly"] = (
        (df["duration_ms_zscore"] > threshold)
        | (df["temperature_c_zscore"] > threshold)
    )
    return df


class ClusterEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._scaler = StandardScaler()

    def _encode_failure_code(self) -> np.ndarray:
        codes = self.df["failure_code"].values.reshape(-1, 1)
        return codes.astype(np.float64)

    def _build_feature_matrix(self) -> np.ndarray:
        fc_encoded = self._encode_failure_code()
        numeric = self.df[["duration_ms", "temperature_c", "voltage_v"]].values
        return np.hstack([fc_encoded, numeric])

    def run_kmeans(self, k: int = 4) -> pd.DataFrame:
        X = self._build_feature_matrix()
        X_scaled = self._scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
        self._kmeans = kmeans
        return self.df

    def run_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        X = self._build_feature_matrix()
        X_scaled = self._scaler.fit_transform(X)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.df["cluster_dbscan"] = dbscan.fit_predict(X_scaled)
        self._dbscan = dbscan
        return self.df

    def cluster_summary(self, label_col: str = "cluster_kmeans") -> pd.DataFrame:
        has_anomaly_col = "is_anomaly" in self.df.columns
        summaries = []

        for cluster_id in sorted(self.df[label_col].unique()):
            cluster_df = self.df[self.df[label_col] == cluster_id]
            dominant_fc = cluster_df["failure_code"].mode()
            dominant_failure_code = int(dominant_fc.iloc[0]) if len(dominant_fc) > 0 else -1

            non_empty = cluster_df[cluster_df["error_message"].notna() & (cluster_df["error_message"] != "")]
            top_msgs = non_empty["error_message"].value_counts().head(3).index.tolist()

            summary = {
                "cluster": int(cluster_id),
                "size": len(cluster_df),
                "dominant_failure_code": dominant_failure_code,
                "mean_duration_ms": round(cluster_df["duration_ms"].mean(), 2),
                "anomaly_rate_pct": round(cluster_df["is_anomaly"].mean() * 100, 1) if has_anomaly_col else 0.0,
                "top_error_messages": top_msgs,
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)

    def save_to_sqlite(self, db_url: str = "sqlite:///data/failures.db"):
        engine = create_engine(db_url)
        self.df.to_sql("failure_records", engine, if_exists="replace", index=False)
