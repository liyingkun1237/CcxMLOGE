"""
输出的json格式 定义 转换 等
"""
import pandas as pd
import os
from datetime import datetime
import simplejson
import pandas_profiling
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
# type 1 接口的输出
# part2 = {
#     "datasetInfo": {"数据集名称": "data_demo", "样本量": 20000, "维度": 200},
#     "varSummary": {"cateVar": [{'IV': 0.01,
#                                 'Type': 'category',
#                                 'missingN': 0,
#                                 'missing_pct': 0,
#                                 'nunique': 2,
#                                 'top1': 1,
#                                 'top1_pct': 95.24,
#                                 'top2': 0,
#                                 'top2_pct': 4.76,
#                                 'top3': None,
#                                 'top3_pct': None,
#                                 'vList': '[0,1]',
#                                 'varName': 'regprov_isequal_liveprov'},
#                                {'IV': 0.01,
#                                 'Type': 'category',
#                                 'missingN': 0,
#                                 'missing_pct': 0,
#                                 'nunique': 2,
#                                 'top1': 1,
#                                 'top1_pct': 95.24,
#                                 'top2': 0,
#                                 'top2_pct': 4.76,
#                                 'top3': None,
#                                 'top3_pct': None,
#                                 'vList': '[0,1]',
#                                 'varName': 'regprov_isequal_liveprov'}
#
#                                ],
#                    "numVar": [{'IV': 0.01,
#                                'Type': 'numric',
#                                'max': 172179.8,
#                                'mean': 37562.1559456181,
#                                'median': 32294.8891325,
#                                'min': 590,
#                                'missingN': 100334,
#                                'missing_pct': 99.33,
#                                'quantile1': 15143.94023275,
#                                'quantile3': 50249.75,
#                                'range': '[590,172180]',
#                                'std': 30375.4095940718,
#                                'varName': 'overdue_cuiqian_meanAmount'}
#                        , {'IV': 0.01,
#                           'Type': 'numric',
#                           'max': 172179.8,
#                           'mean': 37562.1559456181,
#                           'median': 32294.8891325,
#                           'min': 590,
#                           'missingN': 100334,
#                           'missing_pct': 99.33,
#                           'quantile1': 15143.94023275,
#                           'quantile3': 50249.75,
#                           'range': '[590,172180]',
#                           'std': 30375.4095940718,
#                           'varName': 'overdue_cuiqian_meanAmount'}
#
#                               ]},
#     "detailVarPath": {"path": "ccc/ccc/sdcfef"}
#  }
# Type1 = {
#     "reqId": "请求ID",
#     "reqTime": "请求时间戳",
#     "type": 1,
#     "dataDescription": part2,
#     "variableAnalysis": None,
#     "modelOutput": None,
#     "otherOutput": None
#
# }
from ccxmodel.modelutil import ModelUtil
from sklearn.metrics import roc_curve

from ccxMLogE.trainModel import f_getAucKs
from ccxMLogE.varDescSummary import f_rawbins, IV, f_VardescWriter


def f_detailVarhtml(df, userPath):
    '''
    详细变量的输出情况
    :param df:
    :return:
    '''

    profile = pandas_profiling.ProfileReport(df)
    path = os.path.join(userPath, 'detailVarhtml.html')
    profile.to_file(outputfile=path)
    return path


def f_part2Output(resdesc, userPath, df):
    '''
    数据集的描述性分析结果
    :param resdesc: 元组 numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol
    :param userPath 用户路径
    :return:
    '''
    # numVardesc, cateVardesc 转字典
    # detailVarIV 写入csv中，返回路径
    if resdesc:  # 过滤None
        numVardesc = resdesc[0]
        cateVardesc = resdesc[1]
        detailVarIV = resdesc[2]
    else:
        numVardesc = {}
        cateVardesc = {}
        detailVarIV = pd.DataFrame()

    numVar = numVardesc.to_dict(orient='records')
    cateVar = cateVardesc.to_dict(orient='records')

    # 第二部分这里的是返回html文件路径
    # 发现 同步异步都要计算这个的话 会很麻烦
    path_ = f_detailVarhtml(df, userPath)

    # 这是第三部分的事情了
    path = os.path.join(userPath, 'detailVarIV.csv')
    if len(detailVarIV) > 1:
        detailVarIV.to_csv(path, index=False)
    else:
        path = None

    return {"varSummary": {"cateVar": cateVar, "numVar": numVar}, "detailVarPath": {"path": path_}}, path


def f_part2Output4yibu(resdesc, userPath):
    '''
    数据集的描述性分析结果 为了异步服务 异步时 不会计算html 页面
    :param resdesc: 元组 numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol
    :param userPath 用户路径
    :return:
    '''
    # numVardesc, cateVardesc 转字典
    # detailVarIV 写入csv中，返回路径
    if resdesc:  # 过滤None
        numVardesc = resdesc[0]
        cateVardesc = resdesc[1]
        detailVarIV = resdesc[2]
    else:
        numVardesc = {}
        cateVardesc = {}
        detailVarIV = pd.DataFrame()

    numVar = numVardesc.to_dict(orient='records')
    cateVar = cateVardesc.to_dict(orient='records')

    # 第二部分这里的是返回html文件路径
    # 发现 同步异步都要计算这个的话 会很麻烦 异步将不会计算了
    # path_ = f_detailVarhtml(df, userPath)

    # 这是第三部分的事情了
    path = os.path.join(userPath, 'detailVarIV.csv')
    if len(detailVarIV) > 1:
        detailVarIV.to_csv(path, index=False)
    else:
        path = None

    return {"varSummary": {"cateVar": cateVar, "numVar": numVar}, "detailVarPath": {"path": None}}, path


def f_type1Output(reqId, datasetInfo, descout, path):
    part2 = dict({"datasetInfo": datasetInfo}, **descout)
    part3_1 = {"impVar": None, "topNpath": path}
    reqTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    dict1 = {"reqId": reqId, "reqTime": reqTime, "type": 1,
             "dataDescription": part2,
             "variableAnalysis": part3_1,
             "modelOutput": None,
             "otherOutput": None
             }
    return simplejson.dumps(dict1, ensure_ascii=False, ignore_nan=True)


##

# part3 = {
#     "impVar": [{'Feature_Name': 'asset_grad_E',
#                 'gain': 16.8522672727272,
#                 'pct_importance': 0.045},
#                {'Feature_Name': 'asset_grad_D', 'gain': 14.697476, 'pct_importance': 0.0393},
#                {'Feature_Name': '_age_5.0',
#                 'gain': 11.5207033333333,
#                 'pct_importance': 0.0308}]
#     ,
#     "topNpath": "xxx/xxx/xxx.csv"
# }
#

def f_part3Output(imppath, topNpath, dfcolnames):
    '''
    变量重要性分析
    :param imppath: 重要变量结果路径
    :param topNpath: 第二部分提前计算好的所有变量的IV详情
    :param dfcolnames 原始数据的字段
    :return: 结果字典 {}
    '''
    impVar = pd.read_csv(imppath)
    # topN这里可以优化 往回追到one-hot之前的变量列表 在只显示全部的重要变量
    # 开发一个函数 依据one-hot后的变量 找到没有one-hot之前的变量名
    topNames = f_getRawcolnames(impVar.Feature_Name, dfcolnames)
    topNames.sort_values(by=['rank'], inplace=True)
    AllIv = pd.read_csv(topNpath)
    topNIV = pd.merge(topNames, AllIv, left_on='varname', right_on='varname', how='left')
    topNpath = os.path.join(os.path.dirname(topNpath), 'TopNVarIV.csv')
    topNIV.to_csv(topNpath, index=False)
    print('重要变量IV值已经保存成功： ', topNpath)

    return {"impVar": impVar.to_dict(orient='records'), "topNpath": topNpath}


def f_getRawcolnames(impVarcol, dfcolnames):
    '''
    依据模型选出的重要变量 找到原始的变量名
    :param impVarcol: 列表
    :param dfcolnames: 列表
    :return:
    '''
    ls = []
    rankls = []
    for i in dfcolnames:
        # 拿到原始的字段名
        for j, rank in zip(impVarcol, np.arange(len(impVarcol)) + 1):
            if str(i) in str(j):
                ls.append(i)
                rankls.append(rank)
                break

    return pd.DataFrame({'varname': ls, 'rank': rankls})


# 测一下topN
# imppath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxboost\model20171201200539\modeldata\d_2017-12-01_importance_var.csv'
# topNpath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\detailVarIV.csv'
# dfcolnames = pd.read_csv(r'C:\\Users\\liyin\\Desktop\\20170620_tn\\0620_base\\train_base14.csv').columns
#
# zz = f_part3Output(imppath, topNpath, dfcolnames)


# part4 = {
#     "modeldataInfo": [{'入模维度': 61.0,
#                        '正负样本比': 5.9500000000000002,
#                        '总维度': 3145.0,
#                        '样本量': 9661.0,
#                        '正样本比例': 0.14380000000000001,
#                        '重要变量': 58.0},
#                       {'入模维度': 61.0,
#                        '正负样本比': 5.9500000000000002,
#                        '总维度': 3145.0,
#                        '样本量': 4141.0,
#                        '正样本比例': 0.14380000000000001,
#                        '重要变量': 58.0}],
#
#     "modelreport": [{'AUC': 0.69367500000000004,
#                      'KS': 0.29683399999999999,
#                      'f1-score': 0.80000000000000004,
#                      'gini': 0.38735000000000008,
#                      'precision': 0.84999999999999998,
#                      'recall': 0.85999999999999999,
#                      'support': 14422.0},
#                     {'AUC': 0.64169900000000002,
#                      'KS': 0.221996,
#                      'f1-score': 0.79000000000000004,
#                      'gini': 0.28339800000000004,
#                      'precision': 0.82999999999999996,
#                      'recall': 0.85999999999999999,
#                      'support': 6182.0}]
#     ,
#     "aucksPlot": {'trainKSpath': 'xxxx.csv',
#                   'trainAUCpath': 'xxxx.csv',
#                   'testKSpath': 'xxxx.csv',
#                   'testAUCpath': 'xxxx.csv',
#                   }
#     ,
#     "pvalueReport": [{'IV': 0.412686951544129,
#                       'bad': 6,
#                       'bad_per': 0.75,
#                       'bins_score': '(590, 600]',
#                       'factor_per': 0.000579626141138965,
#                       'good': 2,
#                       'model_pvalue': 0.527831077575683,
#                       'total': 8},
#                      {'IV': 0.412686951544129,
#                       'bad': 20,
#                       'bad_per': 0.571428571428571,
#                       'bins_score': '(600, 610]',
#                       'factor_per': 0.00253586436748297,
#                       'good': 15,
#                       'model_pvalue': 0.442838847637176,
#                       'total': 35},
#                      {'IV': 0.412686951544129,
#                       'bad': 33,
#                       'bad_per': 0.407407407407407,
#                       'bins_score': '(610, 620]',
#                       'factor_per': 0.00586871467903202,
#                       'good': 48,
#                       'model_pvalue': 0.362247616052627,
#                       'total': 81},
#                      {'IV': 0.412686951544129,
#                       'bad': 147,
#                       'bad_per': 0.402739726027397,
#                       'bins_score': '(620, 630]',
#                       'factor_per': 0.0264454426894652,
#                       'good': 218,
#                       'model_pvalue': 0.285614490509033,
#                       'total': 365},
#                      {'IV': 0.412686951544129,
#                       'bad': 329,
#                       'bad_per': 0.285590277777777,
#                       'bins_score': '(630, 640]',
#                       'factor_per': 0.083466164324011,
#                       'good': 823,
#                       'model_pvalue': 0.220752000808715,
#                       'total': 1152},
#                      {'IV': 0.412686951544129,
#                       'bad': 713,
#                       'bad_per': 0.183149242229642,
#                       'bins_score': '(640, 650]',
#                       'factor_per': 0.282060570931749,
#                       'good': 3180,
#                       'model_pvalue': 0.168598234653472,
#                       'total': 3893},
#                      {'IV': 0.412686951544129,
#                       'bad': 498,
#                       'bad_per': 0.106798198584602,
#                       'bins_score': '(650, 660]',
#                       'factor_per': 0.337849587016374,
#                       'good': 4165,
#                       'model_pvalue': 0.128025189042091,
#                       'total': 4663},
#                      {'IV': 0.412686951544129,
#                       'bad': 217,
#                       'bad_per': 0.0693512304250559,
#                       'bins_score': '(660, 670]',
#                       'factor_per': 0.226706274452977,
#                       'good': 2912,
#                       'model_pvalue': 0.0971796214580535,
#                       'total': 3129},
#                      {'IV': 0.412686951544129,
#                       'bad': 21,
#                       'bad_per': 0.0443974630021141,
#                       'bins_score': '(670, 680]',
#                       'factor_per': 0.0342703955948413,
#                       'good': 452,
#                       'model_pvalue': 0.0732611268758773,
#                       'total': 473},
#                      {'IV': 0.412686951544129,
#                       'bad': 1,
#                       'bad_per': 0.333333333333333,
#                       'bins_score': '(680, 690]',
#                       'factor_per': 0.000217359802927112,
#                       'good': 2,
#                       'model_pvalue': 0.0569750107824802,
#                       'total': 3},
#                      {'IV': 0.412686951544129,
#                       'bad': 1985,
#                       'bad_per': 0.143819736270105,
#                       'bins_score': 'All',
#                       'factor_per': 1.0,
#                       'good': 11817,
#                       'model_pvalue': None,
#                       'total': 13802}]
#
# }


def f_part4Output(repathlist, train, test, target_name, userPath):
    # modeldataInfo
    modeldataInfo = f_modeldataInfo(repathlist, train, test, target_name)

    # modelreport
    trpredpath = repathlist[3]
    tepredpath = repathlist[4]
    trpred = pd.read_csv(trpredpath)
    tepred = pd.read_csv(tepredpath)

    trp = precision_score(trpred.iloc[:, 1], trpred.P_value > 0.5, average='binary')
    trr = recall_score(trpred.iloc[:, 1], trpred.P_value > 0.5, average='binary')
    trf1score = f1_score(trpred.iloc[:, 1], trpred.P_value > 0.5, average='binary')
    trauc, trks = f_getAucKs(trpredpath)
    trgini = trauc * 2 - 1
    trsupport = trpred.shape[0]

    tep = precision_score(tepred.iloc[:, 1], tepred.P_value > 0.5, average='binary')
    ter = recall_score(tepred.iloc[:, 1], tepred.P_value > 0.5, average='binary')
    tef1score = f1_score(tepred.iloc[:, 1], tepred.P_value > 0.5, average='binary')
    teauc, teks = f_getAucKs(tepredpath)
    tegini = teauc * 2 - 1
    tesupport = tepred.shape[0]

    re = {'AUC': [trauc, teauc],
          'KS': [trks, teks],
          'f1-score': [trf1score, tef1score],
          'gini': [trgini, tegini],
          'precision': [trp, tep],
          'recall': [trr, ter],
          'support': [trsupport, tesupport]}

    modelreport = pd.DataFrame(re).to_dict(orient='records')

    # aucksPlot
    fpr, tpr, thresholds = roc_curve(trpred.iloc[:, 1], trpred.P_value)

    trks = pd.DataFrame([range(len(tpr)), tpr, fpr, tpr - fpr], index=['x', 'tpr', 'fpr', 'ks']).T
    trauc = pd.DataFrame([fpr, tpr], index=['x', 'y']).T

    fpr, tpr, thresholds = roc_curve(tepred.iloc[:, 1], tepred.P_value)
    teks = pd.DataFrame([range(len(tpr)), tpr, fpr, tpr - fpr], index=['x', 'tpr', 'fpr', 'ks']).T
    teauc = pd.DataFrame([fpr, tpr], index=['x', 'y']).T

    trkspath = os.path.join(userPath, 'trainKSpath.csv')
    traucpath = os.path.join(userPath, 'trainAUCpath.csv')
    tekspath = os.path.join(userPath, 'testKSpath.csv')
    teaucpath = os.path.join(userPath, 'testAUCpath.csv')

    trks.to_csv(trkspath, index=False)
    trauc.to_csv(traucpath, index=False)

    teks.to_csv(tekspath, index=False)
    teauc.to_csv(teaucpath, index=False)

    aucksPlot = {'trainKSpath': trkspath,
                 'trainAUCpath': traucpath,
                 'testKSpath': tekspath,
                 'testAUCpath': teaucpath,
                 }

    # pvalueReport
    # 这次先显示测试集的，后续的将测试集的加上
    # 明天来优化一下概率分箱的函数
    # pvalueReport = f_NumVarIV(tepred.P_value * 100, tepred[target_name])
    # pvalueReport['bins'] = pvalueReport['bins'].apply(str)
    trpvalueReport = f_pvalueReport(trpred)
    trpvalueReport['modelDataName'] = 'train'
    tepvalueReport = f_pvalueReport(tepred)
    tepvalueReport['modelDataName'] = 'test'
    pvalueReport = pd.concat([trpvalueReport, tepvalueReport])
    pvalueReport = pvalueReport.to_dict(orient='records')

    return {"modeldataInfo": modeldataInfo, "modelreport": modelreport,
            "aucksPlot": aucksPlot, "pvalueReport": pvalueReport
            }


def f_zcPvalue(x):
    '''
    离一个数最近的整数 且 能被10, 5等整除整除,且离原始数据更近
    :return:
    '''
    if np.ceil(x) % 10 == 0 or np.ceil(x) % 5 == 0:
        return np.ceil(x)
    else:
        a = np.round(x / 5, 0) * 5
        b = np.round(x / 10, 0) * 10
        if np.abs(a - x) < np.abs(b - x):
            return a
        else:
            return b


def f_pvalueReport(tepred):
    x, y = tepred.P_value * 100, tepred.iloc[:, 1]
    rawbins = f_rawbins(x, y)  # 原始的分箱
    print('===原始分箱=====', rawbins)
    mhbins = [f_zcPvalue(i) for i in rawbins]
    mhbins = pd.Series(mhbins).unique().tolist()  # 1130发现 出现0的情况 解决方法
    mhbins[0] = 0
    mhbins[-1] = f_zcPvalue(max(rawbins) + 3.5)  # 加3.5的原因为 让71这种能corver到做大值
    bestbins = pd.Series(mhbins).unique().tolist()
    print('===调整后的分箱=====', bestbins)
    # 1201 开发一个方法 加上百分号
    bestbinslabel = f_addbfh(bestbins)
    tepred['scoreBins'] = pd.cut(x, bestbins, labels=bestbinslabel, right=False)
    iv = IV(pd.cut(x, bestbins, right=False), y)
    iv['bins'] = iv['bins'].apply(str)
    model_pvalue = tepred.groupby('scoreBins')['P_value'].mean().tolist()
    model_pvalue.append(np.nan)
    iv['model_pvalue'] = model_pvalue
    # print(tepred.head(100))
    return iv


def f_addbfh(bins):
    ls = []
    for idx in range(len(bins) - 1):
        _, __ = str(bins[idx]), str(bins[idx + 1])
        ls.append('[' + _ + '% , ' + __ + '%)')
    return ls


# 测试一下
# bins = [20, 21, 34, 56, 78]
# f_addbfh(bins)


#
# 测试一下这个函数
# dd = pd.read_csv(
#     r'C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/ccxboost/model20171130203843/modeldata/d_2017-11-30_train_predict.csv')
# bins = [0.0, 10.0, 15.0, 20.0, 25.0, 35.0, 70.0]
# dd['scoreBins']=pd.cut(dd.P_value * 100, bins)
# dd.groupby('scoreBins')['P_value'].mean().tolist()


def f_modeldataInfo(repathlist, train, test, target_name):
    '''
    对切分出来的训练集和测试集 进行描述性的分析
    :param repathlist:
    :param trian:
    :param test:
    :return:
    {'入模维度': 61.0,
     '正负样本比': 5.9500000000000002,
     '总维度': 3145.0,
     '样本量': 9661.0,
     '正样本比例': 0.14380000000000001,
     '重要变量': 58.0}
    '''
    # 1.加载最终模型 计算出入模所需的变量数
    modelen = f_getmodelen(repathlist[1])
    # 2.总维度
    trrow, trcol = train.shape
    terow, tecol = test.shape
    # 3.重要变量个数
    implen = pd.read_csv(repathlist[2]).shape[0]
    # 4.正负样本的比例
    x = train[target_name].value_counts().values.tolist()
    y = train[target_name].value_counts().values.tolist()
    trpostivePct = x[1] / sum(x)
    tepostivePct = y[1] / sum(x)

    trnegdivpos = x[0] / x[1]
    tenegdivpos = y[0] / y[1]

    re = {'入模维度': [modelen, modelen],
          '正负样本比': [trnegdivpos, tenegdivpos],
          '总维度': [trcol, tecol],
          '样本量': [trrow, terow],
          '正样本比例': [trpostivePct, tepostivePct],
          '重要变量': [implen, implen]}
    return pd.DataFrame(re).to_dict(orient='records')


def f_getmodelen(model_path):
    '''
    依据模型路径 给出需要输入模型的变量个数
    :param model_path: 模型路径
    :param implen: 重要变量长度
    :return:
    '''
    x = ModelUtil.load_bstmodel(model_path)
    try:
        # xgboost 获取变量的方法
        x = x.feature_names
        modellen = len(x)
    except:
        try:
            # 随机森林的获取方法
            modellen = x.n_features_
        except:
            # gbm 获取入模变量的方法
            modellen = len(x.feature_name())

    return modellen


def f_part5Output(repathlist, userPath, desres, modelres, impres):
    '''
    第五部分的输出
    :param repathlist: 算法返回的列表
    :param userPath: 用户路径
    :param desres: 描述性分析的结果
    :param modelres: 模型输出结果
    :param impres: 模型输出的重要变量 part3计算出来
    :return:
    '''
    creattime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    filename = 'analysisReport' + '_' + creattime + '_Ccx' + '.xlsx'
    path = os.path.join(userPath, filename)
    f_analysisReportMain(path, desres, modelres, impres)  # 保存结果至Excel
    part5 = {
        "predictResPath": [repathlist[3], repathlist[4]],  # 这个地方改为list
        "logPath": None,
        "analysisReport": path
    }
    return part5


def f_type2Output(reqId, datasetInfo, descout, path, repathlist, rawdatacol, train, test, target_name, userPath,
                  desres):
    part2 = dict({"datasetInfo": datasetInfo}, **descout)
    part3_2 = f_part3Output(repathlist[2], path, rawdatacol)  # bug 不能为train 因为train已经one-hot过了
    reqTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    part_4 = f_part4Output(repathlist, train, test, target_name, userPath)
    part_5 = f_part5Output(repathlist, userPath, desres, part_4, part3_2)
    dict2 = {"reqId": reqId, "reqTime": reqTime, "type": 2,
             "dataDescription": part2,
             "variableAnalysis": part3_2,
             "modelOutput": part_4,
             "otherOutput": part_5
             }

    return simplejson.dumps(dict2, ensure_ascii=False, ignore_nan=True)


def f_modelOutputWriter(writer, res, res_part3):
    '''
    将模型计算出的结果写出到Excel中
    :param write: 要写出的Excel writer 对象 接着变量描述的写入
    :param res: f_part4Output 函数计算出来的字典
    {"modeldataInfo": modeldataInfo, "modelreport": modelreport,
            "aucksPlot": aucksPlot, "pvalueReport": pvalueReport
            }
    :param res_part3: f_part3Output 函数计算出的结果
    {"impVar": impVar.to_dict(orient='records'), "topNpath": topNpath}
    :return: 保存成Excel
    '''
    modeldataInfo = pd.DataFrame(res['modeldataInfo'], index=['训练集', '测试集'])
    modelreport = pd.DataFrame(res['modelreport'], index=['训练集', '测试集'])
    pvalueReport = pd.DataFrame(res['pvalueReport'])

    PlottrainKS = pd.read_csv(res['aucksPlot']['trainKSpath'])
    PlottrainAUC = pd.read_csv(res['aucksPlot']['trainAUCpath'])
    PlottestKS = pd.read_csv(res['aucksPlot']['testKSpath'])
    PlottestAUC = pd.read_csv(res['aucksPlot']['testAUCpath'])

    # writer = pd.ExcelWriter(path)
    modeldataInfo.to_excel(writer, 'modeldataInfo')
    # 重要变量
    modelreport.to_excel(writer, 'modelreport')
    pvalueReport.to_excel(writer, 'pvalueReport', index=False)
    pd.DataFrame(res_part3['impVar']).to_excel(writer, 'ImpVar')
    pd.read_csv(res_part3['topNpath']).to_excel(writer, 'ImpVarIVdetail', index=False)

    PlottrainKS.to_excel(writer, 'PlottrainKS', index=False)
    PlottrainAUC.to_excel(writer, 'PlottrainAUC', index=False)
    PlottestKS.to_excel(writer, 'PlottestKS', index=False)
    PlottestAUC.to_excel(writer, 'PlottestAUC', index=False)

    writer.save()
    # writer.colse()
    print('模型计算结果保存完成')


# 测试一下
# res = {
#     "modeldataInfo": [
#         {
#             "入模维度": 87.0,
#             "总维度": 90.0,
#             "样本量": 17132.0,
#             "正样本比例": 0.18118141489610087,
#             "正负样本比": 4.519329896907217,
#             "重要变量": 52.0
#         },
#         {
#             "入模维度": 87.0,
#             "总维度": 90.0,
#             "样本量": 7343.0,
#             "正样本比例": 0.18118141489610087,
#             "正负样本比": 4.519329896907217,
#             "重要变量": 52.0
#         }
#     ],
#     "modelreport": [
#         {
#             "AUC": 0.7703199776808679,
#             "KS": 0.38654493296176423,
#             "f1-score": 0.17039343334276819,
#             "gini": 0.5406399553617358,
#             "precision": 0.7016317016317016,
#             "recall": 0.09697164948453608,
#             "support": 17132.0
#         },
#         {
#             "AUC": 0.7340033311284198,
#             "KS": 0.33224417271350665,
#             "f1-score": 0.17039343334276819,
#             "gini": 0.4680066622568395,
#             "precision": 0.7016317016317016,
#             "recall": 0.09697164948453608,
#             "support": 7343.0
#         }
#     ],
#     "aucksPlot": {
#         "trainKSpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/trainKSpath.csv",
#         "trainAUCpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/trainAUCpath.csv",
#         "testKSpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/testKSpath.csv",
#         "testAUCpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/testAUCpath.csv"
#     },
#     "pvalueReport": [
#         {
#             "bins": "[4.4, 9.956)",
#             "good": 1573,
#             "bad": 56,
#             "total": 1629,
#             "factor_per": 0.2218439329974125,
#             "bad_per": 0.03437691835481891,
#             "p": 0.042105263157894736,
#             "q": 0.261599866954931,
#             "woe": -1.8266433624993594,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[9.956, 16.0)",
#             "good": 1321,
#             "bad": 169,
#             "total": 1490,
#             "factor_per": 0.20291434018793408,
#             "bad_per": 0.11342281879194631,
#             "p": 0.12706766917293233,
#             "q": 0.21969067021453517,
#             "woe": -0.5475007397754827,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[16.0, 20.0)",
#             "good": 954,
#             "bad": 190,
#             "total": 1144,
#             "factor_per": 0.15579463434563529,
#             "bad_per": 0.1660839160839161,
#             "p": 0.14285714285714285,
#             "q": 0.158656244802927,
#             "woe": -0.1048947494640313,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[20.0, 29.0)",
#             "good": 1392,
#             "bad": 393,
#             "total": 1785,
#             "factor_per": 0.2430886558627264,
#             "bad_per": 0.22016806722689075,
#             "p": 0.2954887218045113,
#             "q": 0.23149842008980542,
#             "woe": 0.24405762079866558,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[29.0, 76.0)",
#             "good": 773,
#             "bad": 522,
#             "total": 1295,
#             "factor_per": 0.17635843660629172,
#             "bad_per": 0.40308880308880307,
#             "p": 0.3924812030075188,
#             "q": 0.12855479793780142,
#             "woe": 1.1161333891189862,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "All",
#             "good": 6013,
#             "bad": 1330,
#             "total": 7343,
#             "factor_per": 1.0,
#             "bad_per": 0.1811248808388942,
#             "p": 1.0,
#             "q": 1.0,
#             "woe": 0.0,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         }
#     ]
# }
#
# f_modelOutputWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\modelOutput.xlsx', res)


def f_analysisReportMain(path, desres, modelres, impres):
    '''
    存储计算结果的主函数
    :param path: 写入的Excel的路径 xxx.xlsx
    :param desres: 描述性分析的结果 f_mainDesc 计算出来
    :param modelres: 模型计算出的结果 f_part4Output 函数计算
    :return: 写入指定路径下的Excel文件中
    '''

    writer = f_VardescWriter(path, desres)
    f_modelOutputWriter(writer, modelres, impres)
    print('项目计算结果已保存至：', path)

# 测试一下 Excel 会被覆盖的问题
# r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\analysisReport_2017-11-30_175927_Ccx.xlsx'
