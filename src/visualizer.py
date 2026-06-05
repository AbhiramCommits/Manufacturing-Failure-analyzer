import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_cluster_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    label_col: str = "cluster_kmeans",
    output_path: str = "data/cluster_report.png",
) -> str:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: PCA scatter of clusters ---
    feature_cols = ["failure_code", "duration_ms", "temperature_c", "voltage_v"]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    labels = df[label_col].values

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    for lbl in unique_labels:
        mask = labels == lbl
        ax_left.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=f"Cluster {lbl}",
            alpha=0.6,
            s=20,
            color=cmap(lbl),
        )
    ax_left.set_title(f"PCA of Failure Features by {label_col}")
    ax_left.set_xlabel("PC 1")
    ax_left.set_ylabel("PC 2")
    ax_left.legend(markerscale=2, fontsize=8)

    # --- Right panel: anomaly rate bar chart per cluster ---
    has_anomaly = "is_anomaly" in df.columns
    if has_anomaly:
        rates = []
        cluster_ids = []
        for _, row in summary_df.iterrows():
            cid = row["cluster"]
            mask = df[label_col] == cid
            if mask.any():
                rate = df.loc[mask, "is_anomaly"].mean() * 100
            else:
                rate = 0.0
            cluster_ids.append(int(cid))
            rates.append(rate)

        colors = [cmap(i) for i in range(len(cluster_ids))]
        bars = ax_right.bar(
            [str(c) for c in cluster_ids], rates, color=colors
        )
        ax_right.set_title("Anomaly Rate per Cluster")
        ax_right.set_xlabel("Cluster")
        ax_right.set_ylabel("Anomaly Rate (%)")
        for bar, rate in zip(bars, rates):
            ax_right.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax_right.text(0.5, 0.5, "No anomaly data", ha="center", va="center")
        ax_right.set_title("Anomaly Rate per Cluster")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10, wspace=0.25)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
