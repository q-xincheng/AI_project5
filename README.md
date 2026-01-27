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

#### 结果合理性说明（示例）

当出现如下分类报告时：

```
Classification Report:
              precision    recall  f1-score   support
    negative       0.64      0.56      0.60       244
     neutral       0.43      0.35      0.39        79
    positive       0.75      0.83      0.79       477
    accuracy                           0.70       800
   macro avg       0.61      0.58      0.59       800
weighted avg       0.69      0.70      0.69       800
```

这是**合理的中等水平结果**，主要原因是：

- **类别不平衡明显**：positive 占比高（477/800），neutral 样本最少（79/800），因此模型在 positive 上表现更好，而 neutral 的精确率/召回率偏低是常见现象。
- **宏平均指标偏低但符合预期**：macro avg 受小类影响更大，0.59 的 F1 表明模型还有提升空间，但并非异常值。
- **情感边界本身模糊**：neutral 与 negative/positive 的语义边界较模糊，导致混淆和召回率下降。

因此，该报告可以作为课程实验的**合理基线**，若需提升可尝试更强的数据增强或更细致的类别权重设置。

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
- 可考虑使用加权损失函数（未在当前版本实现）

---

## 参考资料

### 论文

1. **BERT**: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL.

2. **ResNet**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

3. **Multimodal Learning**: Baltrusaitis, T., et al. (2019). "Multimodal Machine Learning: A Survey and Taxonomy." TPAMI.

### 代码参考

- PyTorch官方教程: https://pytorch.org/tutorials/
- Transformers库文档: https://huggingface.co/docs/transformers/
- torchvision模型: https://pytorch.org/vision/stable/models.html

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
