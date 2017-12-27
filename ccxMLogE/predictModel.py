"""
对新样本提供预测功能
1.对数据集做和train一致的处理
    为了做和用于建模的数据一样的处理，需要对新来的样本数据进行one-hot处理，使用随机森林的话，还要进行缺失值填补
2.利用训练好的模型进行预测
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import os
from datetime import datetime


def predictmodel(processData, test, indexName, targetName=None):
    '''
    传入的是保存下来的processData对象,内部含有的属性是
    self.modelname = modelname
    self.dummyList = dummyList  # 这个也会获取到
    self.Allcol = Allcol  # f_genAllcol(dummyAfterdf) 可以由这个函数获取到
    self.bstmodelpath = bstmodelpath
    self.bstmodel = ModelUtil.load_bstmodel(bstmodelpath)

    :param processData:
    :param test: 预测的数据集
    :return:
    此时发现了随机森林的一个bug 模型输入了哪些变量，模型没有记录下来
    '''

    bst = processData.getbstmodel()  # 取到最优模型
    modelname = processData.getmodelname()  # 取到模型名字
    vartest = processData.getdata(test)  # 获取到和训练模型时相同处理的预测数据集
    if modelname == 'ccxboost':
        re = bst.predict(xgb.DMatrix(vartest[bst.feature_names], missing=np.nan))
    elif modelname == 'ccxgbm':
        re = bst.predict(vartest[bst.feature_name()])
    elif modelname == 'ccxrf':
        # 随机森林这里 需要去修改一下底层源码了 ccxmodel 里面
        re = bst[0].predict_proba(vartest[bst[1]])[:, 1]  # [x_col] 怎么保证和入模时的是一样的 严重的bug 思考一下，随机森林这里 是不是可以从Alldata入手
    else:
        print('模型预测出错的bug')

    if targetName:
        # 有监督预测
        res = pd.DataFrame({indexName: test[indexName], targetName: test[targetName], "P_value": re})
        res = res[[indexName, targetName, "P_value"]]
    else:
        # 无监督预测
        res = pd.DataFrame({indexName: test[indexName], "predictProb": re})

    return res


def f_save_predictRes(res, modelPath):
    '''
    将模型预测结果保存至指定路径 并返回保存的位置地址
    :param res:
    :param modelPath: 模型路径 主要是为了获取到结果要保存的路径
    :return:
    '''
    path = os.path.dirname(modelPath)
    reqTime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    filename = os.path.join(path, 'predictProb_' + reqTime + '.csv')
    res.to_csv(filename, index=False)
    return filename


##测试一下 尤其是随机森林
'''


if __name__ == '__main__':

# predictmodel(psd, df[df.V0 == 2767], 'V0')  # xgboost 成功
# predictmodel(psd_gbm, df[df.V0 == 2767], 'V0')  # gbm 成功
# predictmodel(psd_rf_nostable, df[df.V0 == 2767], 'V0')  # rf 成功
# predictmodel(psd_rf, df[df.V0 == 2767], 'V0')  # rf 成功

'''
# modelPath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\predict\ccxboost111.model'
# res = predictmodel(psd_rf, df[df.V0 == 2767], 'V0')
# f_save_predictRes(res, modelPath)
