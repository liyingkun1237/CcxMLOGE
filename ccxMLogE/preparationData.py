"""
准备模型所需的数据
1.one-hot处理
2.切分训练集和测试集
3.二期的对字符型变量的处理，对多分类变量的处理
"""

from ccxmodel.modelutil import ModelUtil
import pandas as pd


def f_splitdata(dummyAfterdf, target_name):
    '''
    切分数据集为训练集和测试集
    :param rawdata:
    :param target_name:
    :return:
    '''
    x_columns = [x for x in dummyAfterdf.columns if x not in [target_name]]
    y_columns = target_name
    tr, te = ModelUtil.splitdata(dummyAfterdf, x_columns, y_columns)
    return tr, te


def f_dummyOld(train, dummyList):
    '''
    对训练集进行one-hot操作
    :param train: 训练集
    :param dummyList: 需要dummy的列 list(set(res[4]) - set(res[5])) 一期：需要one-hot的减去多分类的
    :return:
    隐藏bug dummylist为空的情况，或者是为None的情况
    '''
    dummyAfterdf = pd.get_dummies(train, columns=dummyList, dummy_na=True)
    return dummyAfterdf


def f_genAllcol(dummyAfterdf):
    res = dummyAfterdf.head(1)
    res.index = ['All']
    # 要保存下来 后续将会用到 用户预测评分时 需要做相同的处理
    return res


def f_dummyNew(test, dummyList, Allcol):
    '''

    :param test: 需要被编码的数据集
    :param dummyList: 需要one-hot的列
    :param Allcol: 全部的列
    :return: dummy后的数据集
    '''
    # 1.dummy
    dummy1df = pd.get_dummies(test, columns=dummyList, dummy_na=True)
    # 2. 补全列
    dummy2df = pd.concat([Allcol, dummy1df]).drop('All')
    # 3.填补上0
    # 找出one-hot 之后的列
    ls = []
    for i in dummyList:  # 隐藏bug 需要one-hot的列 不能重名
        ls += list(Allcol.filter(regex=i).columns.values)
    fill_dict = {}
    for x in ls:
        fill_dict[x] = 0
    # print(fill_dict)
    var = dummy2df.fillna(fill_dict)
    return var

# dummyList = list(set(res[4]) - set(res[5]))
# dummyAfterdf = f_dummyOld(tr, dummyList)
# f_dummyNew(te.head(3), dummyList, f_genAllcol(dummyAfterdf))
