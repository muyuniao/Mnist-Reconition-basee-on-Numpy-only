#!/usr/bin/env python
# coding: utf-8

# # 定义损失函数

# In[3]:


import numpy as np


# ## 均方误差函数

# In[4]:


def mse(y,t):
    return 0.5 * np.sum((y - t) ** 2)


# ## 交叉熵误差函数

# In[1]:


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

