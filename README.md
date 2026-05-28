# EfficientNetV2 Pipeline — HAM10000 Skin Lesion Classification

## Project Structure

```text
EFFNet/
├── README.md                                      # 项目说明、方法路线与结果对比
├── log.md                                         # 实验过程记录、debug 记录与阶段性结论
│
├── src/                                           # 可复现实验脚本
│   ├── train_SA.py                                # line1: EfficientNetV2-M + Soft Attention + Metadata 训练脚本
│   ├── train_effnet_metadat.py                    # stepA: EfficientNetV2-M + Metadata backbone 训练脚本
│   └── extract_hbp_rf.py                          # stepB/C: 提取 HBP/metadata 特征并训练 Random Forest
│
├── notebooks/                                     # notebook 探索与最小实验验证
│   ├── line1_SA/                                  # line1: 参考 Nguyen et al., 2022 的 Soft Attention 路线
│   │   └── minimum_set_with_finetune_metadata_soft-attention.ipynb
│   │                                               # EfficientNetV2-M + Soft Attention + Metadata 探索
│   │
│   ├── line2_effnet_rf/                           # line2: 参考 EFFNet 的 feature fusion + RF 路线
│   │   ├── minimum_set.ipynb                      # HBP + RF 最小流程验证
│   │   ├── minimum_set_branch_GAP.ipynb           # GAP 特征替代 HBP 的对照探索
│   │   ├── minimum_set_with_fine_tune.ipynb       # EfficientNetV2-M 纯图像微调
│   │   └── minimum_set_with_finetune_metadata.ipynb
│   │                                               # EfficientNetV2-M + Metadata 微调，后续作为 RF 特征提取 backbone
│   │
│   └── debug/                                     # 数据检查与训练框架调试
│       ├── Basic_training_framework.ipynb         # 基础训练框架探索
│       └── check.ipynb                            # 数据、路径、类别分布等检查
│
├── experiments/                                   # 训练输出、特征文件、模型权重与评估结果
│   ├── comparisons_5epoch_before_train/           # 正式长训练前的 5-epoch 快速对比实验
│   │   ├── outputs_effnetv2_m_finetune_classifier/
│   │   │                                           # EfficientNetV2-M 纯图像微调结果
│   │   ├── outputs_effnetv2_m_metadata_finetune_classifier/
│   │   │                                           # EfficientNetV2-M + Metadata 微调结果
│   │   └── outputs_effnetv2_m_softattention_metadata_finetune_classifier/
│   │                                               # EfficientNetV2-M + Soft Attention + Metadata 微调结果
│   │
│   ├── line1_nguyen2022_soft_attention/           # line1 输出目录：EfficientNetV2-M + SA + Metadata 完整训练结果
│   │
│   ├── line2_effnet_feature_fusion_rf/            # line2 输出目录：EfficientNet feature fusion + RF
│   │   ├── stepA_effnetv2_m_metadata_backbone/    # EfficientNetV2-M + Metadata backbone 权重与评估结果
│   │   ├── stepB_hbp_metadata_features/           # 从 fine-tuned backbone 提取的 HBP 特征与 metadata 特征
│   │   └── stepB_hbp_metadata_rf/                 # HBP + Metadata + Random Forest 的 stage-2 分类结果
│   │
│   └── negative_results/                          # 未达到预期或失败实验，保留用于分析
│       └── failed_outputs_effnet_hbp_rf_minimal/  # 早期 HBP + RF minimal 版本失败输出
│
└── __pycache__/                                   # Python 运行产生的缓存文件，可忽略
```


---

## line 1
- reference: Nguyen et al., 2022, Skin Lesion Classification on Imbalanced Data Using Deep Learning with Soft Attention
- current implementation: EfficientNetV2-M + Soft Attention + Metadata

```
preprocess.ipynb
  图像裁剪、增强质量 → 输出 {dx}/enhanced/{image_id}.jpg
      │
      ▼
src/train_SA.py
  │
  ├── 数据准备
  │   [1] 读取 metadata.csv
  │   [2] 构建 enhanced 图像路径，检查缺失
  │   [3] 标签编码 → 7 类
  │   [4] 元数据处理：年龄标准化 + 性别/部位 one-hot → 19 维向量
  │   [5] 按频率倒数计算类别权重
  │   [6] 90/10 分层划分 → DataLoader
  │
  ├── 模型构建
  │   [7] EfficientNetV2-M(features-only, ImageNet 预训练)
  │       + Soft-Attention(16头, γ=0 初始化)
  │       + Metadata MLP(19→64)
  │       → 拼接 → FC(2C+64 → 512 → 7)
  │
  ├── 验证单轮
  │   [8] 可选 --smoke-test 做一次 forward/eval，确认 tensor shape 无误
  │
  ├── 正式训练
  │   [9] 60 epochs
  │       · 每轮：train → validate → ReduceLROnPlateau（按 val F1-macro）
  │       · F1 提升 → 保存 best_effnetv2_softattention_metadata_classifier.pth
  │       · 每轮   → 保存 last_checkpoint.pth（含 optimizer/scheduler）
  │       · patience=12 无提升 → early stopping（按 best_epoch 统计）
  │
  └── 最终输出
      [10] 保存当前最优 epoch 的验证集评估（best epoch = 57）
            → metrics.csv / confusion_matrix.png
            → per_class_metrics.csv / predictions.csv
            → loss_curve.png / metric_curve.png
            → training_history.csv
```

当前结果：
- best epoch: 57 / 60
- validation F1-macro: 0.8784
- validation accuracy: 0.9112
- validation balanced accuracy: 0.8957
- validation precision macro: 0.8664
- validation recall macro: 0.8957

## line 2
- reference: EFFNet: A skin cancer classification model based on feature fusion and random forests
- current implementation: EfficientNetV2-M + Metadata backbone, followed by HBP/Metadata feature fusion and Random Forest

```
preprocess.ipynb
  图像裁剪、增强质量 → 输出 {dx}/enhanced/{image_id}.jpg
      │
      ▼
src/train_effnet_metadat.py
  stepA: EfficientNetV2-M + Metadata backbone
  │
  ├── 数据准备
  │   [1] 读取 metadata.csv
  │   [2] 构建 enhanced 图像路径，检查缺失
  │   [3] 标签编码 → 7 类
  │   [4] 元数据处理：年龄标准化 + 性别/部位 one-hot → 19 维向量
  │   [5] 按频率倒数计算类别权重
  │   [6] 90/10 分层划分 → DataLoader
  │
  ├── 模型构建
  │   [7] EfficientNetV2-M (ImageNet 预训练, num_classes=0)
  │       + Metadata MLP(19→128→64)
  │       → 拼接 → FC(1280+64 → 512 → 7)
  │
  ├── 正式训练
  │   [8] 先做 5 epochs 快速对比，再进行 80 epochs 完整训练
  │       · 每轮：train → validate → ReduceLROnPlateau（按 val F1-macro）
  │       · F1 提升 → 保存 best_effnetv2_metadata_classifier.pth
  │       · patience 无提升 → early stopping
  │
  └── stepA 输出
      [9] best_effnetv2_metadata_classifier.pth
          metrics.csv / predictions.csv / training_history.csv
          class_weights.csv / metadata_info.json / metadata_preprocessor.pkl

      │
      ▼
src/extract_hbp_rf.py
  stepB: feature extraction and fusion
  │
  ├── 加载 stepA 的 best backbone
  ├── 从 EfficientNetV2-M 中间层提取 multi-layer feature maps
  ├── 使用 hierarchical bilinear pooling (HBP) 构造跨层交互特征
  ├── 读取并对齐 metadata 特征
  └── 保存 HBP 特征、metadata 特征、标签与 image_id

      │
      ▼
src/extract_hbp_rf.py
  stepC: Random Forest classification
  │
  ├── 对 HBP 特征做标准化 / PCA
  ├── 拼接 PCA 后的 HBP 特征与 metadata 特征
  ├── 训练 Random Forest 作为最终分类器
  └── 输出 metrics / per-class metrics / predictions / confusion matrix
```

当前观察：
- EfficientNetV2-M + Metadata 的 stepA backbone 指标高于 line1 的早期对比结果。
- 加入 HBP + Metadata + RF 后，预测分布变得更保守，stage-2 RF 暂未带来进一步提升。
- 可能原因包括：当前 HBP/RF 并非 EFFNet 原论文的完全复现、类别均衡增强策略未完全对齐、RF 对高维融合特征更倾向稳定多数类预测，以及 stepA 中 metadata 已经提供了主要增益。

## stepA
### backbone Training Results Comparison (Best of First 5 Epochs)

对比当前三个变体在 **前 5 个 epoch 中的最佳验证集表现**：

| 指标 | EfficientNetV2-M + Metadata | **EfficientNetV2-M + SA + Metadata** | InceptionResNetV2 + SA + Metadata |
|:---|:---:|:---:|:---:|
| **Accuracy** | **0.8423** (E5) | 0.8293 (E4) | 0.8234 (E5) |
| **Balanced Accuracy** | **0.8449** (E5) | 0.8241 (E3) | 0.8441 (E5) |
| **Precision (Macro)** | **0.7086** (E5) | 0.7112 (E4) | ~0.6870 (E5) |
| **Recall (Macro)** | **0.8449** (E5) | 0.8167 (E4) | 0.8441 (E5) |
| **F1 (Macro)** | **0.7574** (E5) | 0.7520 (E4) | 0.7461 (E5) |
| **Val Loss** | 0.5491 (E5) | **0.5234** (E5) | 0.5671 (E5) |

### stepA的策略
- 用EfficientNetV2-M + Metadata的backbone做eff论文的主线
- EfficientNetV2-M + SA + Metadata做对比实验，预期指标高于InceptionResNetV2 + SA + Metadata

## Final Results Comparison

| Line | Model / Stage | Best Epoch | Accuracy | Balanced Accuracy | Precision Macro | Recall Macro | F1 Macro | Val Loss | Notes |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| line1 | EfficientNetV2-M + SA + Metadata | 57 | **0.9112** | **0.8957** | 0.8664 | **0.8957** | 0.8784 | 0.6180 | Nguyen et al. inspired Soft Attention route |
| line2 stepA | EfficientNetV2-M + Metadata backbone | 74 | 0.9062 | 0.8947 | 0.8719 | 0.8947 | **0.8825** | **0.5972** | EFFNet route backbone before RF |
| line2 stepB | HBP + Metadata + Random Forest | - | 0.8792 | 0.7129 | **0.9252** | 0.7129 | 0.7958 | - | RF stage becomes more conservative |
| baseline | InceptionResNetV2 + SA + Metadata | 18 | 0.8902 | 0.8828 | 0.8384 | 0.8828 | 0.8593 | 0.6231 | Output from `Deep learning/outputs_inceptionresnetv2_softattention_metadata_weighted` |


