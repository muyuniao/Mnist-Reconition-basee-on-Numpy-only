#!/usr/bin/env python
# coding: utf-8

# # 梯度计算

# ## 数值差分计算函数

#  数值差分计算梯度,本质就是高数中$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$,理论上来说h值需要趋近于0,但是由于python中数据精度有限,因此h取到1e-4即0.0001(并且这样取值的误差非常小,一般可以直接忽略)

# In[2]:


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad

