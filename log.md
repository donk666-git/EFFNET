## reference paper
1. 对 HAM10000 做类别均衡增强
2. 用 ImageNet 预训练 EfficientNetV2-M 在 HAM10000 上微调
3. 从多个中间层提取 multi-layer feature maps
4. 用 hierarchical bilinear pooling 捕获跨层特征交互
5. 对融合特征做归一化 / 降维
6. 用 Random Forest 做最终分类

## 我先尝试实现 EFFNet-reproduction-lite
包含：
multi-layer features
bilinear pooling / hierarchical bilinear pooling
feature normalization
Random Forest
+
metadata（从另一篇论文借鉴，如果5epoch的效果更好，就加入？）

## 流程
```
1. stepA：train.py（_effnet_backbone）
   训练 EfficientNetV2-M 七分类 backbone

```
# backbone
Best Epoch: 74 / 80 (saved as best EfficientNetV2 + Metadata model)

Metric	Train	Val
Accuracy	0.9900	0.9062
Balanced Accuracy	0.9960	0.8947
Precision (macro)	-	0.8719
Recall (macro)	-	0.8947
F1 (macro)	-	0.8825
Loss	0.0167	0.5972
Learning Rate	-	9.77e-08
Pred distribution (val): class 5: 666, class 4: 117, class 2: 105, class 1: 52, class 0: 34, class 6: 16, class 3: 12
```

2. stepB：extract_hbp_rf.py
   加载 backbone
   提取多层 features
   做 hierarchical bilinear pooling
   normalization
   训练 Random Forest
   输出指标
```

## log
1.模型
"tf_efficientnetv2_m"
2.先训一个普通cnn
image → EfficientNetV2-M → Linear(7)
3.fine-tune
- without metadata
```
# 1 epoch
Training EfficientNetV2: 100%|██████████| 1127/1127 [07:04<00:00,  2.66it/s]
Validating EfficientNetV2: 100%|██████████| 126/126 [00:27<00:00,  4.59it/s]train: {'accuracy': 0.6327526905580828, 'balanced_accuracy': np.float64(0.559640713558637), 'precision_macro': 0.4104619015702342, 'recall_macro': 0.559640713558637, 'f1_macro': 0.457353676735524, 'loss': 1.1685189728822585}
val: {'accuracy': 0.7524950099800399, 'balanced_accuracy': np.float64(0.7518140151698492), 'precision_macro': 0.5458144954013824, 'recall_macro': 0.7518140151698492, 'f1_macro': 0.6023746323106918, 'loss': 0.8339342664055407}
pred distribution: Counter({np.int64(5): 572, np.int64(2): 160, np.int64(1): 84, np.int64(4): 67, np.int64(0): 64, np.int64(6): 37, np.int64(3): 18})
```

```
# 5 epochs
Epoch 3/5
Training EfficientNetV2: 100%|██████████| 1127/1127 [07:13<00:00,  2.60it/s]
Validating EfficientNetV2: 100%|██████████| 126/126 [00:26<00:00,  4.77it/s]
{'epoch': 3, 'train_accuracy': 0.7756573837789859, 'train_balanced_accuracy': np.float64(0.7925621111554568), 'train_precision_macro': 0.6231230768145125, 'train_recall_macro': 0.7925621111554568, 'train_f1_macro': 0.6888465173847657, 'train_loss': 0.647771138952057, 'val_accuracy': 0.8243512974051896, 'val_balanced_accuracy': np.float64(0.8395298152289473), 'val_precision_macro': 0.7035198085055592, 'val_recall_macro': 0.8395298152289473, 'val_f1_macro': 0.7593356407904588, 'val_loss': 0.5504567567221889, 'lr': 0.0001}
pred distribution: Counter({np.int64(5): 620, np.int64(2): 133, np.int64(4): 99, np.int64(1): 64, np.int64(0): 52, np.int64(6): 19, np.int64(3): 15})
Saved best EfficientNetV2 classifier. val_f1_macro=0.7593
```

- with metadata
```
# 1 epoch
Training EfficientNetV2 + Metadata: 100%|██████████| 1127/1127 [07:11<00:00,  2.61it/s]
Validating EfficientNetV2 + Metadata: 100%|██████████| 126/126 [00:25<00:00,  4.88it/s]train: {'accuracy': 0.6005769444136247, 'balanced_accuracy': np.float64(0.5004845408307118), 'precision_macro': 0.3556023773022759, 'recall_macro': 0.5004845408307118, 'f1_macro': 0.3923679435244228, 'loss': 1.292560859299468}
val: {'accuracy': 0.6097804391217565, 'balanced_accuracy': np.float64(0.6790857689190793), 'precision_macro': 0.4272652584859512, 'recall_macro': 0.6790857689190793, 'f1_macro': 0.4583758730186399, 'loss': 0.991992104835734}
pred distribution: Counter({np.int64(5): 454, np.int64(0): 133, np.int64(2): 116, np.int64(4): 104, np.int64(3): 96, np.int64(1): 59, np.int64(6): 40})
```

```
# 5 epochs

```

## debug
- 模型把所有样本都预测成了nv
   - 去掉PCA降维
      - 无效
   - 加入fine tune
      - 有效
- stepB的Python 进程占用内存超过系统可用上限，内核直接 kill 了它。在保存以下文件后该用py文件
   - effnetv2_m + metadata + RF
      - 失败了，指标相对stepA没有提升，预测更保守了
         - 原因
            - 原论文做了类别均衡增强
            - 原论文的 feature fusion 模块与我的“随机 1×1 projection + HBP”不同
            - 原文没有metadata classifier
            - 评估协议不一定完全对齐
         - 接下来怎么办
            - 不走原文，effnetv2_m + softattention + metadata
               - 预期比InceptionResNetV2 + SA + Metadata更好，
            - 完全复现原文
            

```
outputs_effnetv2_m_finetuned_hbp_metadata_rf/
├── X_train_hbp.npy
├── X_val_hbp.npy
├── M_train.npy
├── M_val.npy
├── y_train.npy
├── y_val.npy
├── train_feature_ids.csv
└── val_feature_ids.csv
```
