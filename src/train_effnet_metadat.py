from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (SCRIPT_DIR / "../preprocess/preprocessing").resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_effnetv2_m_metadata_finetune_classifier"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL2IDX = {class_name: idx for idx, class_name in enumerate(CLASSES)}
IDX2LABEL = {idx: class_name for class_name, idx in LABEL2IDX.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-M with HAM10000 metadata features."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default="tf_efficientnetv2_m")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one forward/eval pass before training to check tensor shapes.",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_image_path(row: pd.Series, root: Path) -> str | None:
    image_path = root / row["dx"] / "enhanced" / f"{row['image_id']}.jpg"
    if image_path.exists():
        return str(image_path)
    return None


def load_dataframe(root: Path, metadata_path: Path) -> pd.DataFrame:
    print("root:", root)
    print("metadata:", metadata_path)
    print("root exists:", root.exists())
    print("metadata exists:", metadata_path.exists())

    df = pd.read_csv(metadata_path)
    required_cols = [
        "lesion_id",
        "image_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
        "dataset",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing metadata columns: {missing_cols}")

    df["image_path"] = df.apply(lambda row: get_image_path(row, root), axis=1)
    missing_images = int(df["image_path"].isna().sum())
    print("missing images:", missing_images)

    if missing_images > 0:
        missing_df = df.loc[df["image_path"].isna(), ["image_id", "dx"]].head()
        print(missing_df)
        raise FileNotFoundError(f"{missing_images} images missing. Check image paths.")

    df["label"] = df["dx"].map(LABEL2IDX).astype(int)

    print(df.head())
    print(df.columns)
    print(df.shape)
    print(df["dx"].value_counts())
    print(df[["image_id", "dx", "label", "image_path"]].head())
    print(df["label"].value_counts().sort_index())

    return df


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_tfms, val_tfms


def split_dataframe(
    df: pd.DataFrame,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["dx"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("train:", len(train_df))
    print(train_df["dx"].value_counts())
    print("\nval:", len(val_df))
    print(val_df["dx"].value_counts())

    return train_df, val_df


def build_metadata_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict, dict]:
    train_df_meta = train_df.copy()
    val_df_meta = val_df.copy()

    age_median = train_df_meta["age"].median()
    train_df_meta["age"] = train_df_meta["age"].fillna(age_median)
    val_df_meta["age"] = val_df_meta["age"].fillna(age_median)

    for col in ["sex", "localization"]:
        train_df_meta[col] = train_df_meta[col].fillna("unknown").astype(str)
        val_df_meta[col] = val_df_meta[col].fillna("unknown").astype(str)

    scaler = StandardScaler()
    train_age = scaler.fit_transform(train_df_meta[["age"]])
    val_age = scaler.transform(val_df_meta[["age"]])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    train_cat = encoder.fit_transform(train_df_meta[["sex", "localization"]])
    val_cat = encoder.transform(val_df_meta[["sex", "localization"]])

    train_meta = np.concatenate([train_age, train_cat], axis=1).astype("float32")
    val_meta = np.concatenate([val_age, val_cat], axis=1).astype("float32")

    metadata_info = {
        "metadata_dim": int(train_meta.shape[1]),
        "age_median": float(age_median),
        "age_scaler_mean": scaler.mean_.tolist(),
        "age_scaler_scale": scaler.scale_.tolist(),
        "sex_categories": encoder.categories_[0].tolist(),
        "localization_categories": encoder.categories_[1].tolist(),
    }

    metadata_preprocessor = {
        "age_median": age_median,
        "scaler": scaler,
        "encoder": encoder,
    }

    print("metadata_dim:", metadata_info["metadata_dim"])
    print("train_meta:", train_meta.shape)
    print("val_meta:", val_meta.shape)
    print("encoder categories:", encoder.categories_)

    return train_df_meta, val_df_meta, train_meta, val_meta, metadata_info, metadata_preprocessor


class HAMImageMetadataDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        metadata_array: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.metadata_array = metadata_array
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        metadata = torch.tensor(self.metadata_array[idx], dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        image_id = row["image_id"]

        return image, metadata, label, image_id


class EfficientNetV2MetadataClassifier(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        model_name: str = "tf_efficientnetv2_m",
        num_classes: int = 7,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        image_feature_dim = self.backbone.num_features

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_feat = self.backbone(image)
        meta_feat = self.metadata_mlp(metadata)
        features = torch.cat([image_feat, meta_feat], dim=1)
        return self.classifier(features)


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_meta: np.ndarray,
    val_meta: np.ndarray,
    train_tfms: transforms.Compose,
    val_tfms: transforms.Compose,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = HAMImageMetadataDataset(train_df, train_meta, transform=train_tfms)
    val_dataset = HAMImageMetadataDataset(val_df, val_meta, transform=val_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    images, metas, labels, image_ids = next(iter(train_loader))
    print("images:", images.shape)
    print("metas:", metas.shape)
    print("labels:", labels.shape)
    print("image_ids:", image_ids[:3])

    return train_loader, val_loader


def compute_class_weights(train_df: pd.DataFrame, output_dir: Path) -> torch.Tensor:
    class_counts = train_df["label"].value_counts().sort_index().values
    total_count = class_counts.sum()
    num_classes = len(CLASSES)

    class_weights = total_count / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    for class_name, count, weight in zip(CLASSES, class_counts, class_weights):
        print(f"{class_name:6s} count={count:5d}, weight={weight.item():.4f}")

    pd.DataFrame(
        {
            "class": CLASSES,
            "count": class_counts,
            "weight": class_weights.numpy(),
        }
    ).to_csv(output_dir / "class_weights.csv", index=False)

    return class_weights


def compute_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, metas, labels, _ in tqdm(loader, desc="Training EfficientNetV2 + Metadata"):
        images = images.to(device)
        metas = metas.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, metas)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_ids = []

    for images, metas, labels, image_ids in tqdm(
        loader,
        desc="Validating EfficientNetV2 + Metadata",
    ):
        images = images.to(device)
        metas = metas.to(device)
        labels = labels.to(device)

        logits = model(images, metas)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * images.size(0)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_image_ids.extend(list(image_ids))

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)

    return (
        metrics,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_image_ids,
    )


@torch.no_grad()
def smoke_test_batch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    model.eval()
    images, metas, labels, _ = next(iter(loader))
    images = images.to(device)
    metas = metas.to(device)
    labels = labels.to(device)

    logits = model(images, metas)
    loss = criterion(logits, labels)

    print("logits:", logits.shape)
    print("loss:", loss.item())


def save_best_outputs(
    output_dir: Path,
    model: EfficientNetV2MetadataClassifier,
    val_metrics: dict,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    val_image_ids: list[str],
    args: argparse.Namespace,
    metadata_info: dict,
    class_weights: torch.Tensor,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "classes": CLASSES,
            "label2idx": LABEL2IDX,
            "idx2label": IDX2LABEL,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "metadata_dim": metadata_info["metadata_dim"],
            "age_median": metadata_info["age_median"],
            "metadata_info": metadata_info,
            "class_weights": class_weights,
        },
        output_dir / "best_effnetv2_metadata_classifier.pth",
    )

    pd.DataFrame([val_metrics]).to_csv(output_dir / "metrics.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "image_id": val_image_ids,
            "true_label": [IDX2LABEL[int(i)] for i in y_val],
            "pred_label": [IDX2LABEL[int(i)] for i in y_pred],
            "correct": y_val == y_pred,
        }
    )

    for idx, class_name in IDX2LABEL.items():
        pred_df[f"prob_{class_name}"] = y_prob[:, idx]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    report = classification_report(
        y_val,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "per_class_metrics.csv")

    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(CLASSES))))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        output_dir / "confusion_matrix.csv"
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    root = args.root.resolve()
    metadata_path = args.metadata_path.resolve() if args.metadata_path else root / "metadata.csv"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASSES)

    print("device:", device)
    print("model_name:", args.model_name)

    df = load_dataframe(root, metadata_path)
    train_df, val_df = split_dataframe(df, args.val_size, args.seed)
    train_tfms, val_tfms = build_transforms(args.image_size)
    (
        train_df_meta,
        val_df_meta,
        train_meta,
        val_meta,
        metadata_info,
        metadata_preprocessor,
    ) = build_metadata_features(
        train_df,
        val_df,
    )

    with open(output_dir / "metadata_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2, ensure_ascii=False)

    with open(output_dir / "metadata_preprocessor.pkl", "wb") as f:
        pickle.dump(metadata_preprocessor, f)

    train_df_meta.to_csv(output_dir / "train_split.csv", index=False)
    val_df_meta.to_csv(output_dir / "val_split.csv", index=False)

    train_loader, val_loader = build_loaders(
        train_df_meta,
        val_df_meta,
        train_meta,
        val_meta,
        train_tfms,
        val_tfms,
        args.batch_size,
        args.num_workers,
    )

    class_weights = compute_class_weights(train_df_meta, output_dir)

    model = EfficientNetV2MetadataClassifier(
        metadata_dim=metadata_info["metadata_dim"],
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    print("backbone feature dim:", model.backbone.num_features)

    if args.smoke_test:
        smoke_test_batch(model, train_loader, criterion, device)

    best_f1 = -1.0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics, y_val, y_pred, y_prob, val_image_ids = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_metrics["f1_macro"])

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)
        print(row)
        print("pred distribution:", Counter(y_pred))

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            no_improve = 0
            save_best_outputs(
                output_dir,
                model,
                val_metrics,
                y_val,
                y_pred,
                y_prob,
                val_image_ids,
                args,
                metadata_info,
                class_weights,
            )
            print(f"Saved best EfficientNetV2 + Metadata model. val_f1_macro={best_f1:.4f}")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print("Early stopping triggered.")
            break


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
