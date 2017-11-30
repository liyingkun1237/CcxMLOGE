"""
输入json格式的解析 和 对应的配置函数
"""
import pandas as pd


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
    :param fileUrl: 文件路径
    :param fileType: 文件类型
    :param codeType: 文件编码
    :param fielDelimiter: 文件分隔符
    :param nullValue: 缺失值说明
    :return: 读入的数据 DataFrame
    '''
    fileUrl, fileType, codeType = base['fileUrl'], base['fileType'], base['codeType']
    fielDelimiter, nullValue = base['fielDelimiter'], base['nullValue']
    try:
        if fileType == 'csv':
            data = pd.read_csv(fileUrl, sep=fielDelimiter, na_values=nullValue, encoding=codeType,
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
        print(str(e))
