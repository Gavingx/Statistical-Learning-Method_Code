# -*- encoding: utf-8 -*-
# 感知机模型
# y = sign(w*x + b)（sign是示性函数)
# 模型性质：二分类模型
# 原理：线性可分的数据集，用一个w * x + b = 0的超平面（其中w是超平面的法向量，b是超平面的截距）可将数据集分为两类，目的就是求出这个超平面的参数w和b
# 策略：使分错的点到超平面的距离最小，极小化损失函数
# 损失函数：L(w, b) = - Σyi (w * xi + b)（其中xi属于分错的点，即分错的点到超平面的函数间隔求和）
    # 因为误分类的点一定满足 -yi(w * xi + b) >= 0
# 原始形式算法：
    # 1. 输入：训练数据集 T={(x1, y1), (x2, y2), ... , (xN, yN)}, 其中xi∈ R^n, yi∈{-1, 1}, i = 1， 2， 3，... ，N; 步长（即学习率) 0 ≤ η ≤ 1；
    # 2. 输出：w，b
    # 3. 感知机模型：y = sign(w*x + b)（sign是示性函数)
        # 1. 选取初值w0，b0
        # 2. 在训练集上任取一个数据(xi, yi)
        # 3. 如果 -yi(w * xi + b) 》 0 ，更新w和b
            # w = w + η * yi * xi
            # b = b + η * yi
        # 4. 否则回到第2步，测试下一个数据
        # 5. 直到没有错误分类点或迭代点
# 性质：
    # 感知机是判别模型
    # 只有线性可分数据集才可采用感知机，线性不可分数据集一般不采用感知机算法
    # 线性可分数据集上，感知机算法的原始形式是收敛的，即经过有限次迭代可以得到一个将训练集完全正确划分的分离超平面及感知机模型，k≤(R/r)^2
    # 线性可分数据集上，感知机模型不是唯一的（无论是原始形式还是对偶形式）
    # 对偶形式可以减少计算量，因为对偶形式中有训练实例的内积，而内积可以事先计算好（即所谓的Gram矩阵),因此可减少计算量
    # 原始形式可以任意指定初值w0，b0，对偶形式初值一定要为0，即w0=0， b=0


import numpy as np
import timeit
import os, sys


def load_data(path):
    """加载文件

    :param path:
    :return:
    """
    print("-" * 20 + "加载数据" + "-" * 20)
    start = timeit.default_timer()
    dataArr, dataLabel = [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        label = line.strip().split(',')[0]
        pixel_num = line.strip().split(",")[1:]
        if int(label) >= 5:
            dataLabel.append(1)
        else:
            dataLabel.append(-1)
        dataArr.append([int(i.strip())/255 for i in pixel_num])
    print("加载数据耗时：%ss"%(timeit.default_timer()-start))
    return dataArr, dataLabel


def perceptron(dataArr, dataLabel, epochs=30):
    """感知机算法

    :param dataArr:
    :param dataLabel:
    :param epochs:
    :return:
    """
    print("-" * 20 + "训练" + "-" * 20)
    start = timeit.default_timer()
    dataMat = np.mat(dataArr)
    dataLabel = np.mat(dataLabel).T
    m, n = np.shape(dataMat)
    w = np.zeros((1, n))
    yita = 0.0001
    b = 0
    for epoch in range(epochs):
        for i in range(m):
            x = dataMat[i]
            y = dataLabel[i]
            if -1 * y * (x * w.T + b) >= 0:
                w = w + yita * y * x
                b = b + yita * y
        print("Round %d: %d training"%(epoch+1, epochs))
    print("训练耗时：%ss"%(timeit.default_timer()-start))
    return w, b


def model_test(dataArr, labelArr, w, b):
    """测试模型

    :param dataArr:
    :param labelArr:
    :param w:
    :param b:
    :return:
    """
    print("-" * 20 + "开始测试" + "-" * 20)
    start = timeit.default_timer()
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    errorCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (xi * w.T + b)
        if result >= 0:
            errorCnt += 1
    accruRate = 1 - (errorCnt / m)
    print("测试耗时：%ss"%(timeit.default_timer()-start))
    return accruRate


if __name__ == "__main__":
    train_data, train_label = load_data(os.path.join(os.path.join(os.getcwd(), "../"),'Mnist/mnist_train.csv'))
    test_data, test_label = load_data(os.path.join(os.path.join(os.getcwd(), "../"), 'Mnist/mnist_test.csv'))
    w, b = perceptron(train_data, train_label, epochs=50)
    accuracy = model_test(test_data, test_label, w, b)
    print('accuracy rate is:', accuracy)



