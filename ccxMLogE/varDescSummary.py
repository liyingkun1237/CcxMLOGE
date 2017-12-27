"""
变量的描述统计汇总分析
1.连续型变量 分类型变量的汇总分析 以及IV的计算
2.html的生成
"""
from ccxMLogE.logModel import ABS_log

"""
对dataframe按照分类型变量 还是 连续性变量 给予描述性分析

#分类型变量
变量名	数据类型	缺失个数	缺失率	取值个数	值出现最多个数	值出现最多比例	值出现第二多个数	值出现第二多比例	值出现第三多个数	值出现第三多比例	取值列表	IV值
varName	Type	missingN	missing_pct	nunique	top1	top1_pct	top2	top2_pct	top3	top3_pct	vList	IV
# regprov_isequal_liveprov	category	0	0	2	1	95.24	0	4.76			[0,1]	0.01

#连续型变量
变量名	数据类型	缺失个数	缺失率	取值范围	均值	标准差	最小值	四分之一分位数	中位数	四分之三分位数	最大值	IV值
varName	Type	missingN	missing_pct	range	mean	std	min	quantile1	median	quantile3	max	IV
# overdue_cuiqian_meanAmount	numeric	100334	99.33	[590,172180]	37562.15595	30375.40959	590	15143.94023	32294.88913	50249.75	172179.8	0.01

"""

import pandas as pd
import numpy as np
from sklearn import tree
from inspect import getmembers
import json


def missingN(s):
    return np.sum(pd.isnull(s))


def miss_pct(s):
    re = np.round((np.sum(pd.isnull(s)) / len(s) * 100), 2)
    return re


def unique_(x):
    return x.nunique()


def freq(s, n=3):
    op = pd.value_counts(s)
    op = pd.concat([pd.Series(op.index[:n]).rename(lambda x: "top{}".format(x + 1)),
                    pd.Series(np.round(op.values[:n] / len(s) * 100, 2)).rename(lambda x: "top{}_pct".format(x + 1)).T])
    return op


def f_vlist(x):
    '''

    :param x: 分类型变量
    :return:
    '''
    ll = x.unique().tolist()
    if len(ll) > 5:
        t = [str(i) for i in ll[0:5]]
        t.append('...')
        return t
    else:
        return ll


def f_catevardesc(df):
    '''
    将分类型变量筛选出来 进行描述性分析
    :param df:
    :return:
    '''
    op = pd.concat([df.dtypes.rename("Type"),
                    df.apply(missingN).T.rename("missingN"),
                    df.apply(miss_pct).T.rename("missing_pct"),
                    df.apply(unique_).T.rename("nunique"),
                    df.apply(freq).T,
                    df.apply(f_vlist).T.rename("vList")],
                   axis=1).loc[df.columns]
    return op


def f_range(s):
    '''
    连续变量的取值范围
    :param s:
    :return:
    '''
    return [s.min(), s.max()]


def f_numVardesc(df):
    '''
    连续性变量的统计描述
    :param df:
    :return:
    '''
    op = pd.concat([df.dtypes.rename("Type"),
                    df.apply(missingN).T.rename("missingN"),
                    df.apply(miss_pct).T.rename("missing_pct"),
                    df.apply(f_range).T.rename("range"),
                    df.describe().iloc[1:].T],
                   axis=1).loc[df.columns]
    rename_dict = {"25%": "quantile1",
                   "75%": "quantile3",
                   "50%": "median"}
    return op.rename(columns=rename_dict)


# Type和IV留着和算IV的时候一起处理

# 数据概览的函数开发

def f_viewdata(df, name):
    '''
    数据概览函数
    :param df: 数据集
    :param name: 数据集名称
    :return: 字典
    '''
    row, col = df.shape
    return {"数据集名称": name, "样本量": row, "维度": col}


#########计算IV############################################


def f_zc(x, f=1):
    '''
    离一个数最近的整数 且 能被10 或 100 1000等整除整除
    :return:
    '''
    n = len(str(int(x)))
    if n > 6:
        f = n - 6
        if np.ceil(x) % 10 ** n == 0:
            return x
        else:
            return round(x / 10 ** n, f) * 10 ** n
    else:
        if np.ceil(x) % 10 ** n == 0:
            return x
        else:
            return round(x / 10 ** n, f) * 10 ** n


def f_xiaoshu(x):
    '''
    对小数进行美化
    :param x:
    :return:
    '''
    if x < 1:
        x = round(x, 4)  # 取四位小数
        return f_zc(x * 10000, 2) / 10000
    else:
        return f_zc(x, 2)


def f_mdqujian(x, bins):
    '''
    ###注意一个隐藏bug,区间点出现重复值的问题
    ###注意一个隐藏bug,区间点出现不在所有范围的问题
    ###注意一个隐藏bug,区间点为缺失或者无穷大，无穷小的情况
    :param x: 待分箱的原始数据,且经过无穷值转nan值处理
    :param bins: 决策树给出的分箱，且经过经过美化
    :return: 最终符合逻辑的分箱bins 和 用来填补缺失值的 值
    '''
    min_, max_ = np.min(x), np.max(x)
    if min_ < bins[0]:  # 即美化后的分箱不能取到最小值时
        bins[0] = np.floor(min_)
    if max_ >= bins[-1]:  # 即美化后的分箱不能取到最大值时,取等的原因为左闭右开区间
        bins[-1] = np.ceil(max_ + 1)
    bins = pd.Series(bins).unique().tolist()  # 去除重复值 (损失了最优分箱的效果，如果出现了这个情况)

    bins = [round(i, 2) for i in bins]  # 2017-12-19 新增 将np.round 替换为了round

    if -99 not in x and -99 < bins[0] and any(pd.isnull(x)):  # 限制了bins里面只能为可比较的数字
        bins.insert(0, -99)
        return bins, -99
    elif -999 not in x and -999 < bins[0] and any(pd.isnull(x)):
        bins.insert(0, -999)
        return bins, -999
    elif any(pd.isnull(x)):
        bins.insert(0, f_xiaoshu(x.min()) - 1)  # 要告知用户才对，二期吧#1127发现一个bug
        '''原始的决策树下的最优分箱==含缺失 [-197141.0, 0.5, 157.0]
        Traceback (most recent call last):
        优化后的分箱==含缺失 [-197142.0, -200000.0, 0.5, 160.0]
        解决办法，取整后小于1
        '''
        return bins, f_xiaoshu(x.min()) - 1
    else:
        return bins, None


def f_fillInf(x):
    '''
    填补无穷大，无穷小值，填补思路，这一列的最大，最小值加减1
    但是考虑到这样的情况一般不会太多，忍痛割爱，作为缺失值处理
    :param x:series
    :return:
    '''
    if any(np.isinf(x)) or any(np.isneginf(x)):  # 含有一个无穷值
        # x = pd.Series([np.inf, -np.inf, 1, 2, 3])
        x = x.copy()
        x[(np.isinf(x)) | (np.isneginf(x))] = np.nan
    return x


def f_rawbins(x, y):
    '''
    使用决策树进行分箱的函数
    :param x:
    :param y:
    :return:
    '''
    vary = y.tolist()
    varx = [[i] for i in x]

    clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)
    model = clf.fit(varx, vary)

    v_tree = getmembers(model.tree_)
    v_tree_thres = dict(v_tree)['threshold']
    v_tree_thres = sorted(list(v_tree_thres[v_tree_thres != -2]))
    split_p = [min(x)] + v_tree_thres + [max(x)]
    return split_p


# 最优分箱的主函数
def f_mainBestBins(x, y):
    '''
    #步骤：
    0.判断一个变量只有一个取值的时候，IV为0
    1.判断是否有缺失值和无穷值
    2.没有缺失值和无穷值的，直接进行决策树分箱
    3.有无穷值和缺失值的，先过滤掉无穷值和缺失值，对没有缺失的部分进行最优分箱，然后加入缺失值的分箱

    :param x: 自变量 即待分箱的变量 Series
    :param y: 目标变量 Series
    :return: 最优的分箱区间bins list 和 要使用来填补缺失值的填补值
    '''
    if x.nunique() == 1:
        # 一列只有一个取值时的情况
        return None, None
    if any(pd.isnull(x)) or any(np.isinf(x)) or any(np.isneginf(x)):
        # 过滤掉含有无穷值和缺失值的数据
        x = x.copy()
        x = f_fillInf(x)  # 将无穷值替换为Nan
        dd = pd.DataFrame([x, y]).T
        ddd = dd.dropna()
        # 分箱
        rawbins = f_rawbins(ddd.iloc[:, 0], ddd.iloc[:, 1])
        print('变量名', x.name, '原始的决策树下的最优分箱==含缺失', rawbins)
        # 美化分箱
        mhbins = [f_xiaoshu(i) for i in rawbins]
        bestbins, nan = f_mdqujian(x, mhbins)
        print('优化后的分箱==含缺失', bestbins)
        # # 将缺失部分加入进行分析
        # x = x.fillna(nan)
        # # 计算出IV
        return bestbins, nan
    else:
        # 即不含有无穷值和缺失值的情况
        # 分箱
        rawbins = f_rawbins(x, y)
        print('变量名', x.name, '原始的决策树下的最优分箱', rawbins)
        # 美化分箱
        mhbins = [f_xiaoshu(i) for i in rawbins]
        bestbins, nan = f_mdqujian(x, mhbins)
        print('优化后的分箱', bestbins)
        return bestbins, nan


# 最优分箱的主函数，主要为多类目服务
def f_mainBestBins4multi(x, y):
    '''
    #步骤：
    0.判断一个变量只有一个取值的时候，IV为0
    1.判断是否有缺失值和无穷值
    2.没有缺失值和无穷值的，直接进行决策树分箱
    3.有无穷值和缺失值的，先过滤掉无穷值和缺失值，对没有缺失的部分进行最优分箱，然后加入缺失值的分箱

    :param x: 自变量 即待分箱的变量 Series
    :param y: 目标变量 Series
    :return: 最优的分箱区间bins list 和 要使用来填补缺失值的填补值
    '''
    if x.nunique() == 1:
        # 一列只有一个取值时的情况
        return None, None
    if any(pd.isnull(x)) or any(np.isinf(x)) or any(np.isneginf(x)):
        # 过滤掉含有无穷值和缺失值的数据
        x = x.copy()
        x = f_fillInf(x)  # 将无穷值替换为Nan
        dd = pd.DataFrame([x, y]).T
        ddd = dd.dropna()
        # 分箱
        rawbins = f_rawbins(ddd.iloc[:, 0], ddd.iloc[:, 1])
        # 美化分箱
        # mhbins = [f_xiaoshu(i) for i in rawbins]
        bestbins, nan = f_mdqujian(x, rawbins)
        # # 将缺失部分加入进行分析
        # x = x.fillna(nan)
        # # 计算出IV
        return bestbins, nan
    else:
        # 即不含有无穷值和缺失值的情况
        # 分箱
        rawbins = f_rawbins(x, y)
        # 美化分箱
        # mhbins = [f_xiaoshu(i) for i in rawbins]
        bestbins, nan = f_mdqujian(x, rawbins)
        return bestbins, nan


@ABS_log('MLogEDebug')
def IV(x, y):
    '''

    :param x: 需要计算IV值的自变量
    :param y: 目标变量，target 0,或 1
    :return: 计算出的详细IV值table
    '''
    crtab = pd.crosstab(x, y, margins=True)
    crtab.columns = ['good', 'bad', 'total']
    crtab['factor_per'] = crtab['total'] / len(y)
    crtab['bad_per'] = crtab['bad'] / crtab['total']
    crtab['p'] = crtab['bad'] / crtab.loc['All', 'bad']
    crtab['q'] = crtab['good'] / crtab.loc['All', 'good']
    crtab['woe'] = np.log(crtab['p'] / crtab['q'])
    iv = (crtab['p'] - crtab['q']) * np.log(crtab['p'] / crtab['q'])
    crtab['IV'] = np.round(sum(iv[(iv != np.inf) & (iv != -np.inf)]), 4)
    crtab['varname'] = crtab.index.name
    crtab.index.name = 'bins'
    crtab = crtab.reset_index()
    return crtab


def f_divide(x, y):
    '''
    除法 消除掉被除数为0的情况
    :param x:
    :param y:
    :return:
    '''
    ls = []
    for i, j in zip(x, y):
        if j == 0:
            ls.append(-99)
        else:
            ls.append(x / y)
    print(ls)
    return ls


# 如何根据bins来生成lable
def f_genlabels(bins, nan):
    '''
    依据bins 来生成 labels
    #明确的第一点，labels比bins的长度少一
    :param bins:
    :return: 生成的label
    '''
    if nan:
        # 有缺失值了 才这么处理
        ls = []
        for idx in range(len(bins) - 1):
            if idx == 0:
                ls.append('missValue')
            else:
                _, __ = str(bins[idx]), str(bins[idx + 1])
                ls.append('[' + _ + ',' + __ + ')')
        return ls
    else:
        return None


def f_NumVarIV(x, y):
    '''
    计算连续型变量的IV值，依据自动分箱的结果
    :param x: 自变量 数据结构为Series
    :param y: 因变量 数据结构为Series
    :return: IV值表
    '''
    bins, nan = f_mainBestBins(x, y)
    if nan:
        iv = IV(pd.cut(x.fillna(nan), bins, right=False, labels=f_genlabels(bins, nan)), y)
    else:
        # 原始数据没有缺失值的情况
        iv = IV(pd.cut(x, bins, right=False), y)
    return iv


####多类目归约后计算IV 分类变量的类目数大于15 认为为多分类。这个定义有待商榷
def f_transDict(df):
    '''
    将类目的违约率与类目对应起来 做成字典
    :param df: 只有两列 第一列为bins 第二列为bad_per
    :return:
    '''
    dd = {}
    for i, j in zip(df.bins, df.bad_per):
        # 数据的类型会不会导致问题 需考虑
        dd[i] = j
    return dd


def f_quantiCate(col, dict_):
    '''
    量化多类目的列
    :param col: 多类目的列
    :param dict_: 类目与违约率的映射字典
    :return:
    '''

    def _tras(x):
        '''

        :param x: 列中的元素
        :return: 对应的违约率
        '''
        try:
            return dict_[x]
        except:
            # x的值不在字典中时
            return np.nan

    col_ = col.apply(_tras)
    res = pd.concat([col, col_], axis=1)
    return res


def f_CatVarIV(x, y):
    '''
    计算分类型变量的IV
    1.判断自变量中是否有缺失值
    2.缺失值填补
    3.计算IV
    :param x: 自变量
    :param y: 因变量
    :return: IV值table
    '''
    if any(pd.isnull(x)):
        x = x.copy()
        x = x.fillna('missValue')
        return IV(x, y)
    else:
        return IV(x, y)


def sumCate(x):
    ls = [i for i in x]
    return pd.Series(ls).unique().tolist()


def f_getCateDict(s):
    '''
    将array处理成字典
    :param s:
    :return:
    '''
    j = 1
    re = {}
    for i in s:
        re[str(j)] = i
        j = j + 1
    return re


def f_Cate2Num(df_, dd):
    '''
    将多分类型变量归约为少类目的分类型变量
    :param df_:
                  department  department_1  department_2
    0               执法大队      0.181818   [0.09, 0.2)
    1                放射科      0.212766    [0.2, 0.3)

    :param dd: 字典
    :return: new_df 多了一列类目
    '''

    def f_ddtrans(dd_):
        '''
        将dd 字典中的key和value调换一下
        :param dd:
        :return:
        '''
        re = {}
        for k in dd_.keys():
            if dd_[k]:
                for v in dd_[k]:
                    if v:
                        re[v] = k
                    else:
                        pass  # v 为None时的情况
            else:
                pass  # dd_[k] 为None
        return re

    # def f_(s):
    #     '''
    #
    #     :param s: 要归约的原始列中的元素
    #     :return:
    #     '''
    #     try:
    #         return f_ddtrans(dd)[s] #这样实现的话太慢，产生了笛卡尔积
    #     except:
    #         return np.nan

    df = pd.DataFrame(f_ddtrans(dd), index=['_cate_']).T.reset_index()
    df.columns = [df_.columns[0], '_cate_']  # 隐藏bug，df_.columns[0]=='_cate_'

    return pd.merge(df_, df, how='left')['_cate_']


#############################
# 多类目归约的主函数 计算IV和返回类目对应字典
def f_multiCateReduce(x, y):
    x_ = IV(x, y)[['bins', 'bad_per']]
    dd = f_transDict(x_)
    xx = f_quantiCate(x, dd)
    bins_, nan = f_mainBestBins4multi(xx.iloc[:, 1], y)
    if nan:
        yy = pd.cut(xx.iloc[:, 1].fillna(nan), bins=bins_, right=False)
    else:
        yy = pd.cut(xx.iloc[:, 1], bins=bins_, right=False)
    xxx = pd.concat([xx, yy], axis=1)
    oriname = xxx.columns.unique().values[0]
    xxx.columns = [oriname, oriname + '_1', oriname + '_2']
    s = xxx.groupby([oriname + '_2'])[oriname].agg(sumCate)
    dd = f_getCateDict(s)
    x_new = f_Cate2Num(xxx, dd)
    x_new.name = x.name

    return f_CatVarIV(x_new, y), dd
    # f_CatVarIV(app.Work_unit, All_data.TargetBad_P12)


def f_multiCateReduce4genNewCol(x, y):
    x_ = IV(x, y)[['bins', 'bad_per']]
    dd = f_transDict(x_)
    xx = f_quantiCate(x, dd)
    bins_, nan = f_mainBestBins4multi(xx.iloc[:, 1], y)
    if nan:
        yy = pd.cut(xx.iloc[:, 1].fillna(nan), bins=bins_, right=False)
    else:
        yy = pd.cut(xx.iloc[:, 1], bins=bins_, right=False)
    xxx = pd.concat([xx, yy], axis=1)
    oriname = xxx.columns.unique().values[0]
    xxx.columns = [oriname, oriname + '_1', oriname + '_2']
    s = xxx.groupby([oriname + '_2'])[oriname].agg(sumCate)
    dd = f_getCateDict(s)
    x_new = f_Cate2Num(xxx, dd)
    x_new.name = x.name

    return x_new, dd
    # f_CatVarIV(app.Work_unit, All_data.TargetBad_P12)


##############################数据的变量类型的确定
def f_VarTypeClassfiy(df, cateList):
    '''

    :param df: 数据集
    :param cateList: 用户指定的分类型变量 列表
    :return: 连续型变量的列表 少分类型变量的列表 多分类型变量的列表 需要one-hot处理的变量列表
    # 1 分类型和取值个数小于10的连续型变量  2 多分类型  0 连续型
    '''
    cate_ = df.select_dtypes(include=[object, bool]).columns.tolist()  # 一定是分类型变量 bool为1211新增
    num_ = df.select_dtypes(include=[int, float, 'int64', 'float64']).columns.tolist()  # 连续型变量备选
    # 不在这两个list的变量类型有 时间类型，unit64类型等等
    aa = df.apply(unique_)
    nunique10ls = aa[aa >= 10].index.tolist()  # 取值个数大于10的变量列表
    nunique15ls = aa[aa >= 15].index.tolist()  # 取值个数大于15的变量列表
    nunique2ls = aa[aa <= 2].index.tolist()  # 取值个数小于2的变量列表，主要用于判断是否需要one-hot

    cate_2 = list(set(cate_) & set(nunique15ls))  # 判断出的多分类，隐藏bug没判断出的多分类，如用户自定义了 set(cate_) | set(cateList)
    cate_0 = list(set(num_) & set(nunique10ls))  # 连续型变量
    # 多分类变量确认
    cate_2_ = list(set(cateList) & set(nunique15ls))  # 用户输入分类型变量 且 个数大于15
    cate_2 = list(set(cate_2) | set(cate_2_))  # 多分类变量
    # 连续型变量
    cate_0 = list(set(cate_0) - set(cateList))
    # 分类型
    cate_1 = list(set(cate_) - set(cate_2) | (set(num_) - set(cate_0)))
    # 1127晚 发现一个多分类变量 即在cate_2里 也在 cate_1里 原因是(set(num_) - set(cate_0))带来的
    if (set(cate_2) & set(cate_1)):
        x = set(cate_2) & set(cate_1)
        cate_1 = list(set(cate_1) - x)
    # 二值型的分类变量，需要用one-hot而且是能够直接one-hot的
    one_hot = list(set(cate_1) - set(nunique2ls))
    return cate_0, cate_1, cate_2, one_hot


#############################对变量的详细汇总 desc+IV
@ABS_log('MLogEDebug')
def f_mainDesc(df, index_name, target_name, cateList):
    '''
    变量描述性分析的总函数
    :param df: 数据集
    :param index_name: 唯一索引的列名
    :param target_name: 目标变量的列名
    :param cateList: 用户输入的分类变量列名
    :return:
        numVardesc 连续型变量的描述性分析结果,
        cateVardesc 分类型变量的描述性分析结果,
        detailVarIV 详细的IV值结果,
        dd 多类目分箱时的字典,
        one_hot 后续需要被one_hot的列
        cate_2 多类目的列
    '''
    cate_0, cate_1, cate_2, one_hot = f_VarTypeClassfiy(df, cateList)

    cate_0 = [i for i in cate_0 if i not in [index_name]]
    cate_1 = [i for i in cate_1 if i not in [target_name]]
    cate_2 = [i for i in cate_2 if i not in [index_name]]
    one_hot = one_hot + cate_2

    numVardesc = f_numVardesc(df[cate_0]).reset_index().rename(columns={'index': 'varName'}).drop('Type', axis=1)
    cateVardesc = f_catevardesc(df[cate_1 + cate_2]).reset_index().rename(columns={'index': 'varName'}).drop('Type',
                                                                                                             axis=1)

    # 连续型变量的IV计算
    # 1128发现bug cate_0/1/2有可能出现空列表的情况
    ls = []
    if len(cate_0) > 0:
        for i in cate_0:
            try:
                re = f_NumVarIV(df[i], df[target_name])
                re['Type'] = 'Numeric'
                ls.append(re)
            except:
                pass

        NumVarIV = pd.concat(ls)
        print('++' * 20, '连续变量IV计算', '++' * 20)
    else:
        NumVarIV = pd.DataFrame()
    # 少分类型 + 假连续型
    ls = []
    if len(cate_1) > 0:
        for i in cate_1:
            try:
                re = f_CatVarIV(df[i], df[target_name])
                re['Type'] = 'Category/Numeric'
                ls.append(re)
            except:
                pass
        CatVarIV = pd.concat(ls)
        print('==' * 20, '分类变量IV计算', '==' * 20)
    else:
        CatVarIV = pd.DataFrame()
    # 多分类型变量
    ls = []
    dd = {}  # 类目对应的字典
    if len(cate_2) > 0:
        for i in cate_2:
            try:
                re, _ = f_multiCateReduce(df[i], df[target_name])
                re['Type'] = 'multi-Category'
                ls.append(re)
                dd[i] = _
            except:
                pass

        multiCateVarIv = pd.concat(ls)
        print('ΓΓ' * 20, '多分类变量IV计算', 'ΓΓ' * 20)
    else:
        multiCateVarIv = pd.DataFrame()

    detailVarIV = pd.concat([NumVarIV, CatVarIV, multiCateVarIv])
    detailVarIV['bins'] = detailVarIV['bins'].apply(str)

    simpleVarIV = detailVarIV[['varname', 'IV', 'Type']].drop_duplicates()
    dropcol = simpleVarIV.varname[(simpleVarIV.IV == 0) | (simpleVarIV.IV >= 1)].values.tolist()

    numVardesc = pd.merge(numVardesc, simpleVarIV, how='left', left_on='varName', right_on='varname')
    cateVardesc = pd.merge(cateVardesc, simpleVarIV, how='left', left_on='varName', right_on='varname')

    numVardesc.drop('varname', axis=1, inplace=True)
    cateVardesc.drop('varname', axis=1, inplace=True)

    numVardesc = numVardesc
    cateVardesc = cateVardesc

    return numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol


def f_mdmultiDict(dd, dropcol):
    '''
    修正多分类的字典 删除dropcol中的键
    :param dd:
    :param dropcol:
    :return: Dataframe
    '''
    if dd:  # dd=={} 的情况 会报错
        dd = dd.copy()
        for i in dropcol:
            try:
                del dd[i]
            except:
                pass
        for j in dd.keys():
            dd[j] = [json.dumps(dd[j], ensure_ascii=False)]
        re = pd.DataFrame(dd).T.reset_index()
        re.columns = ['变量名', '分类字典']
        return re
    else:
        return pd.DataFrame({'变量名': ['无'], '分类字典': ["无"]})


def f_VardescWriter(path, res):
    '''

    :param path: 写出的Excel路径
    :param res: 由f_mainDesc 计算出来返回的结果
    :return:
    '''
    numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol = res
    multiCateDict = f_mdmultiDict(dd, dropcol)

    writer = pd.ExcelWriter(path)
    numVardesc.to_excel(writer, 'numVardesc', index=False)
    cateVardesc.to_excel(writer, 'cateVardesc', index=False)
    detailVarIV.to_excel(writer, 'detailVarIV', index=False)
    multiCateDict.to_excel(writer, 'multiCateDict', index=False)
    # writer.save()
    # writer.close()
    print(one_hot, cate_2)
    return writer


r'''
if __name__ == '__main__':
    FP_RawData = pd.read_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\lyk_A\1019_API_xuqiu\FP_VAR_ALL_1020.csv',
                             encoding='gbk')

    index_name, target_name = 'lend_request_id', 'TargetBad_P12'
    cateList = FP_RawData.select_dtypes(include=[object]).columns
    numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol = f_mainDesc(FP_RawData.iloc[:20601], index_name,
                                                                                    target_name, cateList)

    # print(numVardesc)
    # print(cateVardesc)
    # detailVarIV.to_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\detailVarIV.csv', index=False)
    print('++==++' * 50)
    print(dd.keys())

    writer = pd.ExcelWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\out.xlsx')
    numVardesc.to_excel(writer, 'numVardesc', index=False)
    cateVardesc.to_excel(writer, 'cateVardesc', index=False)
    detailVarIV.to_excel(writer, 'detailVarIV', index=False)
    writer.save()
    type(detailVarIV)

    dd['Work_unit'].keys()
    dd['department'].keys()
    dd['job_title']
    dd['live_prov']

    newdata = pd.read_excel(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\model_data(1).xlsx')
    index_name, target_name = '合同号', 'Y'
    sel = [i for i in newdata.columns if i not in ['编号/申请号', '姓名', '身份证号', '手机号', '申请时间']]

    cateList = newdata.select_dtypes(include=[object]).columns

    newdata = newdata[sel].select_dtypes(exclude=['datetime64[ns]'])
    f_VardescWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\NewdescVar.xlsx',
                    f_mainDesc(newdata, index_name, target_name, cateList))

    numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol = f_mainDesc(newdata[sel], index_name,
                                                                                    target_name, cateList)

    multiCateDict = f_mdmultiDict(dd, dropcol)

    writer = pd.ExcelWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\modelbinsout.xlsx')
    numVardesc.to_excel(writer, 'numVardesc', index=False)
    cateVardesc.to_excel(writer, 'cateVardesc', index=False)
    detailVarIV.to_excel(writer, 'detailVarIV', index=False)
    multiCateDict.to_excel(writer, 'multiCateDict', index=False)
    writer.save()

    #################################
    newdata = pd.read_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\data\train.csv', encoding='gbk')
    index_name, target_name = 'id', 'target'

    cateList = newdata.select_dtypes(include=[object]).columns
    numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol = f_mainDesc(newdata, index_name,
                                                                                    target_name, cateList)

    multiCateDict = f_mdmultiDict(dd, dropcol)

    writer = pd.ExcelWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\out_dasai.xlsx')
    numVardesc.to_excel(writer, 'numVardesc', index=False)
    cateVardesc.to_excel(writer, 'cateVardesc', index=False)
    detailVarIV.to_excel(writer, 'detailVarIV', index=False)
    multiCateDict.to_excel(writer, 'multiCateDict', index=False)
    writer.save()

    ##################
    f_VardescWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\data\xxx.xlsx',
                    f_mainDesc(newdata, index_name, target_name, cateList))

'''
r'''
df = pd.read_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\Bugkun.csv')

f_mainDesc(df, 'new_var', 'target', [])
f_mainDesc(df, 'old_var', 'target', [])

df = pd.read_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\base14_1130.csv')

f_mainDesc(df, 'contract_id', 'target', [])

'''
