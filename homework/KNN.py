# -*- encoding: utf-8 -*-
# K近邻
# 定义：是一种基本的分类和回归模型
# 原理：对新的实例，根据其k个最近邻的训练实例的类别，通过多数表决等方式进行预测；因此，k近邻不具有显示的学习过程
# 三要素：
#     1.K值的选择
    # 2.距离度量
    # 3.分类决策规则
# 距离度量：
#     1.欧式距离
    # 2.曼哈顿距离（也叫城市街区距离）
    # 3.P范数（也叫Lp距离或闵式距离）
    # 4.切比雪夫距离（也叫棋盘距离）
# k值的选择：
#   较小的k值预测结果会对周围的实例点非常敏感，如果临近的实例点恰恰是噪声，预测就会出错
#   较大的k值对于输入实例较远的样本点也会对预测额起作用，使预测发生错误。
#   在应用中，先选择较小的k值，再通过交叉验证选择最优的k值
# 分类决策规则：多数表决法（分类），回归（平均值法）
# 原始算法：
#     输入：训练数据集T={(x1, y1), (x2, y2), ... , (xN, yN)}, 其中xi∈X∈R为实例的特征向量，yi∈y={c1, c2, ... , ck}为实例的类别，i=1, 2, ... , N;实例特征向量为x
#     输出：实例x所属的类别y
#     1.根据给定的距离度量，在训练集T中找出与x最近邻的k个点，涵盖这k个点的邻域记作Nk(x)
#     2.在Nk(x)中根据分类决策规则决定x的类别y；y=argmax(ΣI(yi=cj)，i=1，2，3，...，k；I为指示函数：yi=cj时，函数值为1，否则为0
# k近邻的特殊情况是k=1的情形，称为最近邻算法


import numpy as np
import time as time
from tqdm import tqdm


def loadData(path):
    print("-"*20 + "loading data" + "-"*20)
    dataArr = []
    labelArr = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        curLine = line.strip().split(",")
        dataArr.append([float(num) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    return dataArr, labelArr


def calc_dis(x, y, distance):
    if distance == "曼哈顿距离":
        return np.sum(np.abs(x - y))
    elif distance == "欧氏距离":
        return np.sqrt(np.sum(np.square(x - y)))
    elif distance == "切比雪夫距离":
        return np.max(np.abs(x - y))


def getClosest(trainDataMat, trainLabelMat, x, topK, distance):
    distList = [calc_dis(np.array(x), np.array(trainDataMat[i]), distance) for i in range(len(trainDataMat))]
    # argsort函数返回的是数组值从小到大的索引值
    topK_index = np.argsort(distList)[: topK]
    predict = np.argmax(np.bincount(np.array(trainLabelMat)[topK_index]))
    return predict


def model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK, sample_nums, distance):
    print("-"*20 + "starting test" + "-"*20)
    # trainDataMat = np.mat(trainDataArr)
    # trainLabelMat = np.mat(trainLabelArr)
    # testDataMat = np.mat(testDataArr)
    # testLabelMat = np.mat(testLabelArr)
    error_count = 0
    for i in range(sample_nums):
        print('test %d:%d' % (i, sample_nums))
        x = testDataArr[i]
        y = getClosest(trainDataArr, trainLabelArr, x, topK, distance)
        if y != testLabelArr[i]:
            error_count += 1
    return 1 - (error_count / sample_nums)


if __name__ == "__main__":
    start = time.time()

    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')
    acc = model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25, 200, "欧氏距离")
    print('accur is: {:.2%}'.format(acc))