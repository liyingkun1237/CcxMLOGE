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
        print('数据读取bug', str(e))
