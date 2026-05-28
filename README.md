# EfficientNetV2 Pipeline — HAM10000 Skin Lesion Classification

## 最优模型pipeline

最优模型来自对line1与line2 stepA，做**TTA + Ensemble** 的方案。

- line1: `EfficientNetV2-M + Soft Attention + Metadata`
- line2 stepA: `EfficientNetV2-M + Metadata`

指标结果：
- Accuracy: **0.9261**
- Balanced Accuracy: **0.9143**
- F1-macro: **0.9016**

```text
preprocess.ipynb
      │
      ▼
metadata.csv
  │
  ├── 读取csv表头字段
  ├── 构建图像路径
  ├── 标签编码为 7 类：
  │   akiec / bcc / bkl / df / mel / nv / vasc
  └── metadata 预处理：
      age 缺失值填训练集 median
      age 使用 StandardScaler 标准化
      sex / localization 使用 OneHotEncoder 编码
      最终得到 19 维 metadata feature

      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
line1: EfficientNetV2-M + Soft Attention + Metadata  line2 stepA: EfficientNetV2-M + Metadata
  │                                                    │
  ├── EfficientNetV2-M features-only backbone           ├── EfficientNetV2-M backbone
  ├── Soft Attention 模块                               ├── Image feature: global average pooling
  ├── Image feature pooling                             ├── Metadata MLP: 19 → 128 → 64
  ├── Metadata MLP: 19 → 128 → 64                       └── Classifier: image feature + metadata → 7 classes
  └── Classifier: image feature + attention feature
      + metadata → 7 classes

      │                                              │
      │                                              │
      ▼                                              ▼
best_effnetv2_softattention_metadata_classifier.pth  best_effnetv2_metadata_classifier.pth
  best epoch = 57                                      best epoch = 74
  F1-macro = 0.8784                                    F1-macro = 0.8825

      │                                              │
      └──────────────────────┬───────────────────────┘
                             ▼
TTA prediction
  对每张验证图像做 4 种测试时增强：
  orig / hflip / vflip / hvflip

  每个模型分别输出 4 组概率：
  prob_orig, prob_hflip, prob_vflip, prob_hvflip

  对同一模型的 4 组概率取平均：
  prob_line1_tta = mean(prob_line1_4tta)
  prob_line2_tta = mean(prob_line2_4tta)

                             │
                             ▼
Probability Ensemble
  在验证集上搜索融合权重 alpha：

  final_prob = alpha * prob_line1_tta
             + (1 - alpha) * prob_line2_tta

  最佳权重：
  alpha = 0.50

  即：
  final_prob = 0.5 * prob_line1_tta
             + 0.5 * prob_line2_tta

                             │
                             ▼
Final prediction
  pred_label = argmax(final_prob)

                             │
                             ▼
Final outputs
  experiments/ensemble_line1_line2_stepA/
  ├── tta_ensemble_alpha_0.50_metrics.csv
  ├── tta_ensemble_alpha_0.50_predictions.csv
  ├── tta_ensemble_alpha_0.50_per_class_metrics.csv
  ├── tta_ensemble_alpha_0.50_confusion_matrix.csv
  └── tta_ensemble_search_results.json
```

选择该方案的原因：
- 两个单模型都已经取得较高性能，但结构不同，错误模式存在互补。
  - line1 引入 Soft Attention，更关注病灶区域的空间特征。
  - line2 stepA 去掉 Soft Attention，但保留 EfficientNetV2-M + metadata 的强 backbone 表达。
  - TTA 可以降低单次图像方向变化带来的预测波动。
- 集成学习可以综合两个模型的置信度，比单独取某一个模型更稳定，提升泛化能力与推理性能。


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
│   ├── ensemble/                                  # line1 与 line2 stepA 的概率融合和 TTA 实验
│   │   └── ensemble+TTA.ipynb
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
│   ├── ensemble_line1_line2_stepA/                # line1 与 line2 stepA 的概率融合 / TTA 融合结果
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

### stepA
#### backbone Training Results Comparison (Best of First 5 Epochs)

对比当前三个变体在 **前 5 个 epoch 中的最佳验证集表现**：

| 指标 | EfficientNetV2-M + Metadata | **EfficientNetV2-M + SA + Metadata** | InceptionResNetV2 + SA + Metadata |
|:---|:---:|:---:|:---:|
| **Accuracy** | **0.8423** (E5) | 0.8293 (E4) | 0.8234 (E5) |
| **Balanced Accuracy** | **0.8449** (E5) | 0.8241 (E3) | 0.8441 (E5) |
| **Precision (Macro)** | **0.7086** (E5) | 0.7112 (E4) | ~0.6870 (E5) |
| **Recall (Macro)** | **0.8449** (E5) | 0.8167 (E4) | 0.8441 (E5) |
| **F1 (Macro)** | **0.7574** (E5) | 0.7520 (E4) | 0.7461 (E5) |
| **Val Loss** | 0.5491 (E5) | **0.5234** (E5) | 0.5671 (E5) |

#### stepA的策略
- 用EfficientNetV2-M + Metadata的backbone做eff论文的主线
- EfficientNetV2-M + SA + Metadata做对比实验，预期指标高于InceptionResNetV2 + SA + Metadata

## Ensemble + TTA

在完成 line1 和 line2 stepA 后，进一步尝试对两个强模型做集成学习：

```text
line1: EfficientNetV2-M + Soft Attention + Metadata
line2 stepA: EfficientNetV2-M + Metadata backbone

final_prob = alpha * prob_line1 + (1 - alpha) * prob_line2_stepA
```

实验分为两步：
- **普通概率融合**：直接读取两个模型在验证集上的 `predictions.csv`，搜索融合权重 `alpha`。
- **TTA + 概率融合**：先对每张验证图像做 `orig / hflip / vflip / hvflip` 四种测试时增强，分别平均两个模型的预测概率，再做模型间概率融合。

结果：
- 普通概率融合的最佳权重为 `alpha=0.70`，即 70% line1 + 30% line2 stepA，F1-macro 提升到 **0.8889**。
- TTA 后最佳权重为 `alpha=0.50`，即 line1 和 line2 stepA 等权融合，F1-macro 进一步提升到 **0.9016**。
- TTA + Ensemble 是当前所有实验中表现最好的方案。



## Final Results Comparison

| 实验线 | 模型 / 阶段 | Best Epoch | Accuracy | Balanced Accuracy | Precision Macro | Recall Macro | F1 Macro | Val Loss | 说明 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| Ensemble + TTA | EfficientNetV2-M + SA + Metadata 与 EfficientNetV2-M + Metadata 等权融合 | - | **0.9261** | **0.9143** | 0.8938 | **0.9143** | **0.9016** | - | 当前最佳结果；TTA modes: orig / hflip / vflip / hvflip |
| Ensemble | 70% line1 + 30% line2 stepA 概率融合 | - | 0.9182 | 0.9045 | 0.8780 | 0.9045 | 0.8889 | - | 不重新训练，只融合两个模型的验证集概率 |
| line2 stepA | EfficientNetV2-M + Metadata backbone | 74 | 0.9062 | 0.8947 | 0.8719 | 0.8947 | 0.8825 | **0.5972** | EFFNet 路线的 backbone，单模型 F1-macro 最好 |
| line2 stepB | HBP + Metadata + Random Forest | - | 0.8792 | 0.7129 | **0.9252** | 0.7129 | 0.7958 | - | RF 阶段 precision 高，但 recall 明显下降，预测更保守 |
| line1 | EfficientNetV2-M + SA + Metadata | 57 | 0.9112 | 0.8957 | 0.8664 | 0.8957 | 0.8784 | 0.6180 | 参考 Nguyen et al. 的 Soft Attention 路线 |
| baseline | InceptionResNetV2 + SA + Metadata | 18 | 0.8902 | 0.8828 | 0.8384 | 0.8828 | 0.8593 | 0.6231 | 来自 `Deep learning/outputs_inceptionresnetv2_softattention_metadata_weighted` |


### 结果分析
- line2的解释：
  - EfficientNetV2-M + Metadata 的 stepA backbone 指标高于 line1 的早期对比结果。
  - 加入 HBP + Metadata + RF 后，预测分布变得更保守，即precision 很高，但 recall 明显低，说明 RF 预测得很“收缩”，比如 
    - per-class 里：
      - mel recall = 0.5676
      - df recall  = 0.5833
      - bcc recall = 0.6275
    - stage-2 RF 暂未带来进一步提升。可能原因包括：当前 HBP/RF 并非 EFFNet 原论文的完全复现、类别均衡增强策略未完全对齐（导致少数类别的性能下降）、RF 对高维融合特征更倾向稳定多数类预测，以及 stepA 中 metadata 已经提供了主要增益。
- line2 stepB 的 Random Forest 虽然 precision macro 最高，但 recall macro 下降明显，因此更适合作为 feature-fusion/RF 路线的分析结果，而不是最终最佳模型。
  - RF 只能在固定特征层面过采样，在特征维度高，样本数量不均衡时学习到了保守的边界策略。
- **最终最佳方案是 TTA + Ensemble**：Accuracy = **0.9261**，Balanced Accuracy = **0.9143**，F1-macro = **0.9016**。
- 与最佳单模型 line2 stepA 相比，TTA + Ensemble 的 F1-macro 从 **0.8825** 提升到 **0.9016**，说明 line1 和 line2 stepA 的错误模式存在互补。

