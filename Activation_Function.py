#!/usr/bin/env python
# coding: utf-8

# # 函数定义文件

# In[1]:


import numpy as np


# ## sigmoid函数

# In[2]:


def sigmoid(x):
    return 1/np.exp(-x)


# ## sigmoid导函数

# In[ ]:


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# ## softmax函数

# In[5]:


def softmax(x):
    x_max=np.max(x)
    exp_x=np.exp(x-x_max)#防止溢出
    sum_exp_x=np.sum(exp_x)
    y=exp_x/sum_exp_x
    return y


# ## softmax损失函数

# In[5]:


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


# ## relu函数

# In[6]:


def relu(x):
    return np.maximum(x,0)


# ## relu导函数

# In[7]:


def relu_grad(x):
    #grad = np.zeros(x)
    #grad[x>=0] = 1
    x = np.where(x>=0,1,0)
    return x

