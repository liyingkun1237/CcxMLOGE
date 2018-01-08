"""
输入json格式的解析 和 对应的配置函数
"""
import pandas as pd

from ccxMLogE.logModel import ABS_log


def f_getCateList(fields):
    '''
    依据用户输入的字段列表 获取到用户指定的分类型变量
    :param fields: [{}] 0-离散型变量，1-连续型变量
    [{'fileName': 'name', 'fieldType': 1}, {'fileName': 'age', 'fieldType': 0}]
    :return:
    '''
    re = pd.DataFrame(fields).query('fieldType==0').fileName.values.tolist()
    return re


def f_readData(base):
    '''
    读取数据
    很关键 1208 发现60.17上存在严重的编码问题 试图解决
    :param fileUrl: 文件路径
    :param fileType: 文件类型
    :param codeType: 文件编码
    :param fielDelimiter: 文件分隔符
    :param nullValue: 缺失值说明
    :return: 读入的数据 DataFrame
    '''
    fileUrl, fileType, codeType = base['fileUrl'], base['fileType'], base['codeType']
    fielDelimiter, nullValue = base['fielDelimiter'], base['nullValue']
    nullValues = ['', '  # N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A',
                  'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
    nullValues = nullValues + [nullValue]
    nullValues = list(flat(nullValues))
    try:
        if fileType == 'csv':
            data = pd.read_csv(fileUrl, sep=fielDelimiter, na_values=nullValues, encoding=codeType,
                               error_bad_lines=False)
            return data
        elif fileType == 'txt':
            data = pd.read_table(fileUrl, sep=fielDelimiter, na_values=nullValue, encoding=codeType,
                                 error_bad_lines=False)
            return data

    except OSError:
        if fileType == 'csv':
            data = pd.read_csv(fileUrl, sep=fielDelimiter, na_values=nullValue, encoding=codeType, engine='python',
                               error_bad_lines=False)
            return data
        elif fileType == 'txt':
            data = pd.read_table(fileUrl, sep=fielDelimiter, na_values=nullValue, encoding=codeType, engine='python',
                                 error_bad_lines=False)
            return data

    except Exception as e:
        raise ValueError("read data bug by lyk {}".format(str(e)))


def flat(l):
    '''
    将多层嵌套的列表 弄成一维的
    :param l:
    :return:
    '''
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def f_ReadData(base):
    '''
    这个函数是在f_readData基础上进行的封装 即数据读取失败了 重新读取
    2018-01-08新增的 为了解决编码的错误
    :param base:
    :return:
    '''
    try:
        return f_readData(base)
    except ValueError:
        if base['codeType'] == 'GBK':
            base['codeType'] = 'utf-8'
            print('前端编码获取为gbk不正确')
            return f_readData(base)
        else:
            base['codeType'] = 'gbk'
            print('前端编码获取为uff-8不正确')
            return f_readData(base)
