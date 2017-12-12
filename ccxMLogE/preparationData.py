"""
准备模型所需的数据
1.one-hot处理
2.切分训练集和测试集
3.二期的对字符型变量的处理，对多分类变量的处理
"""

from ccxmodel.modelutil import ModelUtil
import pandas as pd
import os
import pickle


# 定义一个类 记录下需要one-hot 的列，需要填补的缺失值等前期数据训练时做的数据处理
class processData(object):
    def __init__(self, modelname, dummyList, Allcol, bstmodelpath):
        self.modelname = modelname
        self.dummyList = dummyList  # 这个也会获取到
        self.Allcol = Allcol  # f_genAllcol(dummyAfterdf) 可以由这个函数获取到
        self.bstmodelpath = bstmodelpath
        self.bstmodel = ModelUtil.load_bstmodel(bstmodelpath)

    def getdata(self, test):
        '''
        获得和训练模型时 对数据集相同的处理方法
        :param test: 需要被编码的数据集
        :return:
        '''
        if self.modelname == 'ccxrf':
            var = f_dummyNew(test, self.dummyList, self.Allcol)
            var = var.fillna(-999)
            return var
        else:
            return f_dummyNew(test, self.dummyList, self.Allcol)

    def __getstate__(self):
        '''
        pickel 时 将只pickle return返回的值
        :return:
        (self.modelname, self.dummyList, self.Allcol, self.bstmodel, self.bstmodelpath)
        '''

        return self.modelname, self.dummyList, self.Allcol, self.bstmodelpath, self.bstmodel

    def __setstate__(self, state):
        '''
        unpickle 时 解析出pickle 中的对象信息
        :param state:
        :return:
        '''
        self.modelname, self.dummyList, self.Allcol, self.bstmodelpath, self.bstmodel = state

    def getmodelname(self):
        return self.modelname

    def getbstmodel(self):
        return self.bstmodel

    def getdummyList(self):
        return self.dummyList

    def getAllcol(self):
        return self.Allcol

    def getbstmodelpath(self):
        return self.bstmodelpath


def f_saveprocessData(self, reqId, userPath):
    '''
    将processData对象保存 并返回保存的地址
    :param reqId: 为了把一次请求和另一次请求区分开
    :param userPath:
    :return:
    '''
    path = os.path.join(userPath, 'predict')
    if os.path.exists(path):
        prepath = os.path.join(path, self.modelname + '_' + reqId + '.model')
    else:
        os.mkdir(path)
        prepath = os.path.join(path, self.modelname + '_' + reqId + '.model')
    with open(prepath, 'wb') as f:
        pickle.dump(self, f)
    return prepath


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
    for i in dummyList:  # 隐藏bug 需要one-hot的列 不能重名 #bug 需要onehot的列是V1 却找到了V12等
        regex = str(i) + '_'  # 有下划线的存在 才可能是one-hot后的变量 ，思考隐藏的bug点
        ls += list(Allcol.filter(regex=regex).columns.values)
    fill_dict = {}
    for x in ls:
        fill_dict[x] = 0
    # print(fill_dict)
    var = dummy2df.fillna(fill_dict)
    return var


# dummyList = list(set(res[4]) - set(res[5]))
# dummyAfterdf = f_dummyOld(tr, dummyList)
# f_dummyNew(te.head(3), dummyList, f_genAllcol(dummyAfterdf))




if __name__ == '__main__':
    # 测试一下 保存模型处理数据的类


    modelpath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxboost\model20171211190055\modeltxt\model_ccxboost_2017-12-11.txt'
    bst = ModelUtil.load_bstmodel(modelpath)
    Allcol = bst.feature_names
    import numpy as np

    dd = pd.DataFrame(np.ones(len(Allcol))).T
    dd.columns = Allcol
    dd.index = ['All']
    dummyList = list(set([i.split('_')[0] for i in Allcol if '_' in i]))
    psd = processData('ccxboost', dummyList, dd, modelpath)
    psd.modelname
    psd.save('111', r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit')

    path = 'C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/predict/ccxboost111.model'
    with open(path, 'rb') as f:
        psd_1 = pickle.load(f)

    #####
    import pandas as pd

    df = pd.read_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\tn_3000_2017_12_08_V2.csv')
    psd.getdata(df.head(1)).isnull().sum()
    len(Allcol)

    # 测试成功 明天来开发接口
    psd.bstmodel

    #### 生成一个测试gbm的psd对象
    modelpath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxgbm\model20171212120101\modeltxt\model_ccxgbm_2017-12-12.txt'
    bst = ModelUtil.load_bstmodel(modelpath)
    Allcol = bst.feature_name()
    import numpy as np

    dd = pd.DataFrame(np.ones(len(Allcol))).T
    dd.columns = Allcol
    dd.index = ['All']
    dummyList = list(set([i.split('_')[0] for i in Allcol if '_' in i]))
    psd_gbm = processData('ccxgbm', dummyList, dd, modelpath)

    # 生成一个随机森林的测试
    modelpath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxrf\model20171212140335\modeltxt\model_ccxrf_2017-12-12.txt'
    bst = ModelUtil.load_bstmodel(modelpath)

    psd_rf_nostable = processData('ccxrf', dummyList, dd, modelpath)

    modelpath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxrf\model20171212140414\modeltxt\model_ccxrf_2017-12-12.txt'
    bst = ModelUtil.load_bstmodel(modelpath)

    psd_rf = processData('ccxrf', dummyList, dd, modelpath)

    ####
    with open(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\predict\ccxgbm_Xgboost-128-721742.model', 'rb') as f:
        psd_1 = pickle.load(f)

    psd_1.bstmodel.bstmodel.bstmodel
