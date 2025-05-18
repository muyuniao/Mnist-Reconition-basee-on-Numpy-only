# MNIST手写数字识别 - 基于纯NumPy实现

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)                                                                                                                                                                     
本项目通过仅使用NumPy库实现了一个简单的两层神经网络，用于MNIST手写数字识别。代码完全从零开始编写，旨在帮助深入理解神经网络的前向传播、反向传播、激活函数和损失函数的核心原理。

## 📁 文件结构

```
├── Activation_Function.py      # 激活函数（ReLU/Sigmoid/Softmax）及其导数
├── Loss_Function.py            # 损失函数（交叉熵误差、均方误差）
├── Gradient_Calculating.py     # 数值梯度计算工具
├── MNIST-Training.ipynb        # 神经网络训练流程（Jupyter Notebook）
└── README.md
```

## 🛠️ 依赖项
- Python 3.8+
- NumPy
- TensorFlow（仅用于加载MNIST数据集）

安装依赖：
```bash
pip install numpy tensorflow
```

## 🚀 快速开始

**1.克隆仓库**：

```git
git clone git@github.com:muyuniao/Mnist-Reconition-basee-on-Numpy-only.git
```

**2.运行训练**：

- 直接执行 `MNIST-Training.ipynb`（使用Jupyter Notebook）。
- 或导出为Python脚本运行：

```
jupyter nbconvert --to script MNIST-Training.ipynb
python MNIST-Training.py
```

## 🧠 模型架构

- **网络结构**：输入层（784）→ 隐藏层（50神经元，ReLU）→ 输出层（10神经元，Softmax）
- **损失函数**：交叉熵误差
- **优化器**：随机梯度下降（SGD，学习率0.05）
- **训练配置**：
  - 批量大小：512
  - 迭代次数：10,000
  - 数据集：MNIST（60,000训练样本 + 10,000测试样本）
