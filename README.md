# 多模态情感分类实验 - Multimodal Sentiment Classification

**作者**: 秦鑫成 (Student ID: 10235501453)  
**项目**: 当代人工智能实验五 - 多模态情感分类

本项目实现了一个基于 PyTorch 的多模态情感分类系统，融合文本和图像信息进行三分类（positive/neutral/negative）。

---

## 目录

1. [运行环境](#运行环境)
2. [代码文件结构](#代码文件结构)
3. [完整运行流程](#完整运行流程)
4. [实验设置与超参数](#实验设置与超参数)
5. [模型设计](#模型设计)
6. [实验结果](#实验结果)
7. [参考资料](#参考资料)

---

## 运行环境

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖

- Python >= 3.10
- PyTorch >= 2.0.0
- Transformers >= 4.30.0 (BERT)
- torchvision >= 0.15.0 (ResNet)
- Pillow, pandas, numpy, scikit-learn
- matplotlib, seaborn (可视化)

---

## 代码文件结构

```
AI_project5_qinxincheng10235501453/
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖
├── main.py                     # 主入口脚本
├── train.txt                   # 训练数据标签
├── test_without_label.txt      # 测试数据（待预测）
├── data/                       # 数据目录
│   ├── 1.jpg, 1.txt           # 图像和文本文件（按guid配对）
│   ├── 2.jpg, 2.txt
│   └── ...
├── configs/                    # 配置文件
│   └── config.yaml            # 主配置文件（超参数、路径等）
├── src/                        # 源代码
│   ├── __init__.py
│   ├── dataset.py             # 数据加载与预处理
│   ├── model.py               # 多模态模型定义
│   ├── train.py               # 训练脚本
│   ├── predict.py             # 预测脚本
│   └── ablation_study.py      # 消融实验脚本
├── outputs/                    # 输出目录
│   ├── best_model.pth         # 最佳模型权重
│   ├── predictions.txt        # 测试集预测结果
│   ├── confusion_matrix.png   # 混淆矩阵
│   └── training_history.png   # 训练曲线
└── reports/                    # 实验报告
    └── ablation_study.md      # 消融实验结果
```

---

## 完整运行流程

### 1. 训练模型

训练多模态融合模型：

```bash
python main.py --mode train --config configs/config.yaml
```

模型会自动：
- 将训练数据按 80:20 划分为训练集和验证集
- 在训练过程中保存最佳模型（基于验证集F1分数）
- 输出训练曲线和混淆矩阵
- 保存验证集结果到 `outputs/results.txt`

### 2. 预测测试集

使用训练好的模型预测 `test_without_label.txt`：

```bash
python main.py --mode predict --config configs/config.yaml
```

预测结果会保存到 `outputs/predictions.txt`，格式为：
```
guid,tag
8,positive
1576,neutral
...
```

### 3. 消融实验

运行三种模式的对比实验（文本、图像、多模态）：

```bash
python main.py --mode ablation
```

该脚本会依次训练：
1. **Text-only**: 仅使用文本特征
2. **Image-only**: 仅使用图像特征
3. **Multimodal**: 融合文本+图像特征

结果保存在 `outputs/text_only/`, `outputs/image_only/`, `outputs/multimodal/`，并生成对比报告 `reports/ablation_study.md`。

---

## 实验设置与超参数

为保证实验可复现，所有关键设置已固定在 `configs/config.yaml` 中：

### 随机种子
```yaml
seed: 42  # 固定随机种子确保可复现
```

### 模型配置
```yaml
model:
  text_model: "bert-base-uncased"    # 文本编码器
  image_model: "resnet50"            # 图像编码器
  hidden_dim: 256                    # 融合层隐藏维度
  num_classes: 3                     # 三分类
  dropout: 0.3                       # Dropout率
```

### 训练配置
```yaml
training:
  batch_size: 16                     # 批大小
  num_epochs: 20                     # 最大训练轮数
  learning_rate: 2.0e-5              # 学习率（AdamW）
  weight_decay: 0.01                 # 权重衰减
  val_split: 0.2                     # 验证集比例（20%）
  early_stopping_patience: 5         # 早停耐心值
  gradient_clip: 1.0                 # 梯度裁剪
```

### 数据处理
```yaml
data:
  max_text_length: 128               # 文本最大长度
  image_size: 224                    # 图像尺寸（ResNet标准）
```

---

## 模型设计

### 整体架构

本项目采用**双编码器+融合层**的多模态架构：

```
┌─────────────┐      ┌──────────────┐
│   Text      │      │   Image      │
│ (guid.txt)  │      │ (guid.jpg)   │
└──────┬──────┘      └──────┬───────┘
       │                    │
       ▼                    ▼
┌─────────────┐      ┌──────────────┐
│    BERT     │      │  ResNet-50   │
│  Encoder    │      │   Encoder    │
└──────┬──────┘      └──────┬───────┘
       │                    │
       │ (768-dim)         │ (2048-dim)
       ▼                    ▼
┌─────────────┐      ┌──────────────┐
│ Projection  │      │  Projection  │
│  (256-dim)  │      │  (256-dim)   │
└──────┬──────┘      └──────┬───────┘
       │                    │
       └──────────┬─────────┘
                  │
            Concatenation
                  │
                  ▼
         ┌────────────────┐
         │  Fusion Layer  │
         │   (MLP + ReLU) │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Classifier    │
         │  (3 classes)   │
         └────────────────┘
```

### 关键设计

#### 1. **文本编码器 (BERT)**
- 使用预训练的 `bert-base-uncased` 模型
- 提取 `[CLS]` token 作为文本表示（768维）
- 通过全连接层投影到 256 维

#### 2. **图像编码器 (ResNet-50)**
- 使用预训练的 ResNet-50（ImageNet权重）
- 移除最后的分类层，保留特征提取部分
- 输出 2048 维特征向量
- 冻结部分早期层以加速训练
- 通过全连接层投影到 256 维

#### 3. **数据增强**
- **文本**: BERT自带的tokenization和padding
- **图像**:
  - 训练集: 随机水平翻转、随机旋转(±10°)、颜色抖动
  - 验证/测试集: 仅resize和标准化

#### 4. **融合策略**
- **特征拼接**: 将文本和图像的投影特征拼接 (256+256=512维)
- **MLP分类器**: 
  ```
  Linear(512 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 3)
  ```

#### 5. **训练策略**
- **损失函数**: CrossEntropyLoss
- **优化器**: AdamW (学习率 2e-5, 权重衰减 0.01)
- **早停**: 验证F1连续5轮不提升则停止
- **梯度裁剪**: 最大范数为1.0

### 设计亮点

1. **模块化设计**: 支持灵活切换单模态/多模态模式
2. **预训练优势**: 利用BERT和ResNet的预训练权重，提升小样本学习能力
3. **消融实验**: 通过统一接口支持 text-only、image-only、multimodal 三种模式
4. **可复现性**: 固定随机种子，确保实验结果可重现
5. **数据增强**: 针对图像的增强策略提升泛化能力

---

## 实验结果

### 1. 验证集结果

本节展示在验证集（20% 训练数据）上的性能：

#### 多模态融合模型（Multimodal）

```
Validation Accuracy: 0.XXXX0000000
Validation F1 Score: 0.XXXX

Classification Report:
              precision    recall  f1-score   support
    negative       0.XX      0.XX      0.XX       XXX
     neutral       0.XX      0.XX      0.XX       XXX
    positive       0.XX      0.XX      0.XX       XXX
    accuracy                           0.XX       XXX
   macro avg       0.XX      0.XX      0.XX       XXX
weighted avg       0.XX      0.XX      0.XX       XXX
```

*注: 实际结果将在运行实验后自动填充*

### 2. 消融实验结果

对比三种模式的性能：

| 模式 | 验证准确率 | 验证F1 | 说明 |
|------|----------|--------|------|
| Text-only | XX.X% | 0.XXX | 仅使用文本信息 |
| Image-only | XX.X% | 0.XXX | 仅使用图像信息 |
| Multimodal | XX.X% | 0.XXX | 融合文本+图像 |

**关键发现**:
- 多模态融合模型预期优于单模态模型
- 文本信息对情感分类的贡献 vs 图像信息
- 具体结果见 `reports/ablation_study.md`

### 3. 可视化结果

训练过程中会生成以下可视化文件（保存在 `outputs/` 目录）：

1. **training_history.png**: 训练/验证的损失、准确率、F1曲线
2. **confusion_matrix.png**: 验证集混淆矩阵

---

## 遇到的问题与解决方案

### Bug 1: 图像/文本文件缺失

**问题**: 部分 guid 对应的图像或文本文件不存在，导致加载失败。

**解决方案**: 
- 在 `dataset.py` 中添加异常处理
- 图像缺失时创建空白图像 (224x224 白色)
- 文本缺失时使用空字符串
- 保证数据加载的鲁棒性

```python
try:
    image = Image.open(image_path).convert('RGB')
except:
    image = Image.new('RGB', (224, 224), color='white')
```

### Bug 2: GPU内存不足

**问题**: 批大小过大导致 CUDA out of memory。

**解决方案**:
- 将 batch_size 从 32 降至 16
- 冻结 ResNet 早期层减少显存占用
- 使用梯度累积（可选）

### Bug 3: BERT下载慢或失败

**问题**: 首次运行时下载 BERT 模型耗时长。

**解决方案**:
- 使用镜像源或提前下载模型
- 或者改用更小的模型如 `distilbert-base-uncased`

### Bug 4: 标签不平衡

**问题**: 训练数据中三类标签分布可能不均衡。

**解决方案**:
- 监控各类别的 precision/recall
- 使用 F1 score 作为主要评估指标（对不平衡数据更鲁棒）
- **NEW**: 现已实现完整的类别不平衡解决方案（见下方"类别不平衡处理"章节）

---

## 类别不平衡处理与监控改进

### 概述

针对数据集中的类别不平衡问题（positive: 59.7%, negative: 29.8%, neutral: 10.5%），本项目实现了多种解决策略：

### 1. 数据层面：过采样策略

在 `configs/config.yaml` 中配置：

```yaml
class_imbalance:
  # 过采样策略: null, "random", 或 "smote"
  oversample_strategy: "random"  # RandomOverSampler
  sampling_strategy: "auto"       # 自动平衡少数类
```

**支持的策略**:
- `random`: RandomOverSampler - 随机复制少数类样本
- `smote`: SMOTE - 合成少数类样本（部分支持）
- `null`: 不使用过采样

**示例**:
```yaml
# 轻度过采样：将 neutral 提升到 500 样本
sampling_strategy: {0: 1193, 1: 500, 2: 2388}
```

### 2. 损失函数层面：类别权重与 Focal Loss

#### 类别权重
```yaml
class_imbalance:
  class_weights: [1.0, 2.5, 1.0]  # [negative, neutral, positive]
  loss_function: "ce"              # CrossEntropyLoss with weights
```

#### Focal Loss
针对难分类样本，使用 Focal Loss：

```yaml
class_imbalance:
  loss_function: "focal"
  focal_gamma: 2.0                 # 聚焦参数
  focal_alpha: [1.0, 2.5, 1.0]    # 可选的类别权重
```

### 3. 预测层面：阈值调整与温度缩放

**阈值调整**: 降低 positive 类的过预测

```yaml
prediction:
  positive_threshold: 0.6  # >0.5 表示提高 positive 阈值
  temperature: 1.0         # 概率校准 (1.0 = 不缩放)
```

**温度缩放**: 校准预测概率

```yaml
prediction:
  temperature: 1.5  # >1.0 = 更平滑的概率分布
```

### 4. 训练稳定性：多种子与 K 折交叉验证

#### 多种子训练
```yaml
stability:
  multi_seed: [42, 123, 456]  # 训练 3 次并输出平均指标
```

输出：
- 每个种子的详细结果
- 平均 F1、Accuracy、Recall（带标准差）
- 保存到 `outputs/multiseed_results.txt`

#### K 折交叉验证
```yaml
stability:
  k_fold: 5  # 5 折分层交叉验证
```

输出：
- 每折的详细指标
- 平均指标与标准差
- 保存到 `outputs/kfold_results.txt`

### 5. 监控与指标改进

#### 详细指标输出

每个 epoch 都会输出：
- Macro-F1（主要评估指标）
- Negative Recall
- Neutral Recall
- Positive Recall

#### Neutral 错误分析

```yaml
monitoring:
  save_neutral_errors: true
```

自动保存被误分类的 neutral 样本到 `outputs/neutral_errors.txt`，包括：
- 样本 GUID
- 预测标签
- 各类别概率

#### 增强的混淆矩阵

混淆矩阵图表现在包含：
- 每个类别的 Recall 值
- 高分辨率输出 (150 DPI)

#### 训练曲线

自动生成 6 个子图：
1. Loss (训练 vs 验证)
2. Accuracy (训练 vs 验证)
3. Macro F1 Score (训练 vs 验证)
4. Negative Recall (验证)
5. Neutral Recall (验证)
6. Positive Recall (验证)

### 配置示例

#### 推荐配置 1: 轻度过采样 + 类别权重
```yaml
class_imbalance:
  oversample_strategy: "random"
  sampling_strategy: "auto"
  class_weights: [1.0, 2.5, 1.0]
  loss_function: "ce"

prediction:
  positive_threshold: 0.55
  temperature: 1.0

monitoring:
  save_neutral_errors: true
  verbose_metrics: true
```

#### 推荐配置 2: Focal Loss + K 折验证
```yaml
class_imbalance:
  oversample_strategy: null
  class_weights: null
  loss_function: "focal"
  focal_gamma: 2.0
  focal_alpha: [1.0, 3.0, 1.0]

stability:
  k_fold: 5

monitoring:
  save_neutral_errors: true
```

#### 推荐配置 3: 多种子鲁棒性测试
```yaml
class_imbalance:
  oversample_strategy: "random"
  class_weights: [1.0, 2.5, 1.0]
  loss_function: "ce"

stability:
  multi_seed: [42, 123, 456, 789, 2024]
```

### 输出文件说明

训练完成后，`outputs/` 目录包含：

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳模型权重 |
| `training_history.png` | 训练曲线（6 个子图） |
| `confusion_matrix.png` | 混淆矩阵（带 Recall 标注） |
| `results.txt` | 验证集详细结果 |
| `detailed_metrics.txt` | 详细指标汇总 |
| `neutral_errors.txt` | Neutral 错误分析 |
| `multiseed_results.txt` | 多种子训练结果（如启用） |
| `kfold_results.txt` | K 折验证结果（如启用） |

---

## 参考资料

### 论文

1. **BERT**: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL.

2. **ResNet**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

3. **Multimodal Learning**: Baltrusaitis, T., et al. (2019). "Multimodal Machine Learning: A Survey and Taxonomy." TPAMI.

4. **Focal Loss**: Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

5. **Imbalanced Learning**: He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." IEEE TKDE.

### 代码参考

- PyTorch官方教程: https://pytorch.org/tutorials/
- Transformers库文档: https://huggingface.co/docs/transformers/
- torchvision模型: https://pytorch.org/vision/stable/models.html
- imbalanced-learn: https://imbalanced-learn.org/

### 数据集

本项目使用的多模态情感数据集包含：
- 图像: JPEG格式，社交媒体风格
- 文本: 短文本，通常包含话题标签
- 标签: positive / neutral / negative

---

## 许可与引用

本项目为课程作业，仅供学习参考。

**作者**: 秦鑫成 (Student ID: 10235501453)  
**课程**: 当代人工智能实验五  
**完成时间**: 2024

如有问题，请联系: qinxincheng@example.com
