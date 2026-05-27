from pathlib import Path
import argparse
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IDX2LABEL = {i: c for i, c in enumerate(CLASSES)}


def batch_slices(n, batch_size):
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


def fit_scaler_memmap(X, batch_size=512):
    scaler = StandardScaler()

    n = X.shape[0]

    for start, end in batch_slices(n, batch_size):
        batch = np.asarray(X[start:end], dtype=np.float32)
        scaler.partial_fit(batch)

    return scaler


def transform_with_scaler_memmap(X, scaler, out_path, batch_size=512):
    n, d = X.shape

    X_scaled = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, d)
    )

    for start, end in batch_slices(n, batch_size):
        batch = np.asarray(X[start:end], dtype=np.float32)
        batch_scaled = scaler.transform(batch).astype(np.float32)
        X_scaled[start:end] = batch_scaled

    return X_scaled


def fit_incremental_pca(X_scaled, n_components=512, batch_size=512, seed=42):
    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size
    )

    n = X_scaled.shape[0]

    for start, end in batch_slices(n, batch_size):
        batch = np.asarray(X_scaled[start:end], dtype=np.float32)
        ipca.partial_fit(batch)

    return ipca


def transform_with_pca_memmap(X_scaled, pca, out_path, batch_size=512):
    n = X_scaled.shape[0]
    k = pca.n_components_

    X_pca = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, k)
    )

    for start, end in batch_slices(n, batch_size):
        batch = np.asarray(X_scaled[start:end], dtype=np.float32)
        batch_pca = pca.transform(batch).astype(np.float32)
        X_pca[start:end] = batch_pca

    return X_pca


def oversample_features(X, y, seed=42):
    rng = np.random.default_rng(seed)

    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_list = []
    y_list = []

    for cls in unique_classes:
        idx = np.where(y == cls)[0]

        sampled_idx = rng.choice(
            idx,
            size=max_count,
            replace=True
        )

        X_list.append(X[sampled_idx])
        y_list.append(y[sampled_idx])

    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)

    perm = rng.permutation(len(y_bal))

    return X_bal[perm], y_bal[perm]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def save_outputs(
    output_dir,
    y_true,
    y_pred,
    y_prob,
    val_image_ids,
    metrics,
    prefix="hbp_metadata_rf"
):
    output_dir = Path(output_dir)

    pd.DataFrame([metrics]).to_csv(
        output_dir / f"metrics_{prefix}.csv",
        index=False
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"per_class_metrics_{prefix}.csv"
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        output_dir / f"confusion_matrix_{prefix}.csv"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Fine-tuned EfficientNetV2-M HBP + Metadata + RF")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{prefix}.png", dpi=300)
    plt.close()

    pred_df = pd.DataFrame({
        "image_id": val_image_ids,
        "true_label": [IDX2LABEL[i] for i in y_true],
        "pred_label": [IDX2LABEL[i] for i in y_pred],
        "correct": y_true == y_pred
    })

    for i, cls in IDX2LABEL.items():
        if i < y_prob.shape[1]:
            pred_df[f"prob_{cls}"] = y_prob[:, i]

    pred_df.to_csv(
        output_dir / f"predictions_{prefix}.csv",
        index=False
    )


def main(args):
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading features with mmap...")
    print("Feature dir:", feature_dir)
    print("Output dir:", output_dir)
    print("=" * 80)

    X_train_hbp = np.load(feature_dir / "X_train_hbp.npy", mmap_mode="r")
    X_val_hbp = np.load(feature_dir / "X_val_hbp.npy", mmap_mode="r")

    M_train = np.load(feature_dir / "M_train.npy")
    M_val = np.load(feature_dir / "M_val.npy")

    y_train = np.load(feature_dir / "y_train.npy")
    y_val = np.load(feature_dir / "y_val.npy")

    train_ids_df = pd.read_csv(feature_dir / "train_feature_ids.csv")
    val_ids_df = pd.read_csv(feature_dir / "val_feature_ids.csv")

    val_image_ids = val_ids_df["image_id"].tolist()

    print("X_train_hbp:", X_train_hbp.shape, X_train_hbp.dtype)
    print("X_val_hbp:", X_val_hbp.shape, X_val_hbp.dtype)
    print("M_train:", M_train.shape, M_train.dtype)
    print("M_val:", M_val.shape, M_val.dtype)
    print("y_train:", Counter(y_train))
    print("y_val:", Counter(y_val))

    # ------------------------------------------------------------
    # 1. Scale HBP feature in batches
    # ------------------------------------------------------------
    print("\nFitting StandardScaler on HBP features...")
    scaler = fit_scaler_memmap(
        X_train_hbp,
        batch_size=args.batch_size
    )

    with open(output_dir / "scaler_hbp.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Transforming train HBP features...")
    X_train_scaled = transform_with_scaler_memmap(
        X_train_hbp,
        scaler,
        output_dir / "X_train_hbp_scaled.npy",
        batch_size=args.batch_size
    )

    print("Transforming val HBP features...")
    X_val_scaled = transform_with_scaler_memmap(
        X_val_hbp,
        scaler,
        output_dir / "X_val_hbp_scaled.npy",
        batch_size=args.batch_size
    )

    # ------------------------------------------------------------
    # 2. Incremental PCA on HBP only
    # ------------------------------------------------------------
    print("\nFitting IncrementalPCA...")
    pca = fit_incremental_pca(
        X_train_scaled,
        n_components=args.pca_dim,
        batch_size=args.batch_size,
        seed=args.seed
    )

    with open(output_dir / "pca_hbp.pkl", "wb") as f:
        pickle.dump(pca, f)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print("PCA explained variance:", explained)

    print("Transforming train PCA...")
    X_train_pca = transform_with_pca_memmap(
        X_train_scaled,
        pca,
        output_dir / "X_train_hbp_pca.npy",
        batch_size=args.batch_size
    )

    print("Transforming val PCA...")
    X_val_pca = transform_with_pca_memmap(
        X_val_scaled,
        pca,
        output_dir / "X_val_hbp_pca.npy",
        batch_size=args.batch_size
    )

    # ------------------------------------------------------------
    # 3. Concatenate metadata after PCA
    # ------------------------------------------------------------
    print("\nConcatenating PCA HBP features with metadata...")
    X_train_fused = np.concatenate(
        [
            np.asarray(X_train_pca, dtype=np.float32),
            M_train.astype(np.float32)
        ],
        axis=1
    )

    X_val_fused = np.concatenate(
        [
            np.asarray(X_val_pca, dtype=np.float32),
            M_val.astype(np.float32)
        ],
        axis=1
    )

    print("X_train_fused:", X_train_fused.shape)
    print("X_val_fused:", X_val_fused.shape)

    np.save(output_dir / "X_train_fused_pca_metadata.npy", X_train_fused)
    np.save(output_dir / "X_val_fused_pca_metadata.npy", X_val_fused)

    # ------------------------------------------------------------
    # 4. Oversampling
    # ------------------------------------------------------------
    print("\nOversampling training features...")
    X_train_bal, y_train_bal = oversample_features(
        X_train_fused,
        y_train,
        seed=args.seed
    )

    print("Before:", Counter(y_train))
    print("After:", Counter(y_train_bal))

    # ------------------------------------------------------------
    # 5. Random Forest
    # ------------------------------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=args.seed,
        n_jobs=-1,
        verbose=1
    )

    rf.fit(X_train_bal, y_train_bal)

    with open(output_dir / "rf_hbp_metadata.pkl", "wb") as f:
        pickle.dump(rf, f)

    # ------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------
    print("\nEvaluating...")
    y_pred = rf.predict(X_val_fused)
    y_prob = rf.predict_proba(X_val_fused)

    metrics = compute_metrics(y_val, y_pred)

    print("Metrics:")
    print(metrics)
    print("Prediction distribution:")
    print(Counter(y_pred))

    save_outputs(
        output_dir=output_dir,
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
        val_image_ids=val_image_ids,
        metrics=metrics,
        prefix="hbp_metadata_rf"
    )

    config = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "pca_dim": args.pca_dim,
        "batch_size": args.batch_size,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "pca_explained_variance": explained,
        "metrics": metrics
    }

    with open(output_dir / "config_hbp_metadata_rf.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Outputs saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        type=str,
        default="outputs_effnetv2_m_finetuned_hbp_metadata_rf",
        help="Directory containing X_train_hbp.npy, M_train.npy, etc."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_effnetv2_m_hbp_metadata_rf_stage2",
        help="Output directory."
    )

    parser.add_argument(
        "--pca_dim",
        type=int,
        default=512,
        help="PCA dimension for HBP image features."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for scaler/PCA transformation."
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Number of Random Forest trees."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()
    main(args)