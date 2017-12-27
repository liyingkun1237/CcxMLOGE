"""
模型日志模块 主要为日志格式定义
"""

import logging
import datetime
import os
from ccxMLogE.config import LOGFILEPATH
import sys


# 创建一个info级别的日志文件，按天创建
def ml_infologger(username, reqId):
    if os.path.exists(LOGFILEPATH):
        os.chdir(LOGFILEPATH)
    else:
        os.mkdir(LOGFILEPATH)
        os.chdir(LOGFILEPATH)

    format = '[%(asctime)s - line:%(lineno)d - %(levelname)s]:{}--> %(message)s'.format(reqId)
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    infoLogName = r'%s_info_%s.log' % (username, curDate)

    formatter = logging.Formatter(format)

    infoLogger = logging.getLogger('%s_info_%s.log' % (username, curDate))
    logpath = os.path.join(LOGFILEPATH, infoLogger.name)
    #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not infoLogger.handlers:
        infoLogger.setLevel(logging.INFO)

        infoHandler = logging.FileHandler(infoLogName, 'a', encoding='utf-8')
        infoHandler.setLevel(logging.INFO)
        infoHandler.setFormatter(formatter)
        infoLogger.addHandler(infoHandler)

    os.chdir(os.pardir)

    return infoLogger, logpath


# 创建一个DEBUG级别的日志文件，按天创建
def tn_debuglogger(message):
    if os.path.exists(LOGFILEPATH):
        os.chdir(LOGFILEPATH)
    else:
        os.mkdir(LOGFILEPATH)
        os.chdir(LOGFILEPATH)

    format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s:\n %(message)s'
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    DebugLogName = r'%s_Debug_%s.log' % (message, curDate)

    formatter = logging.Formatter(format)

    DebugLogger = logging.getLogger('%s_Debug_%s.log' % (message, curDate))

    #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not DebugLogger.handlers:
        DebugLogger.setLevel(logging.DEBUG)

        DebugHandler = logging.FileHandler(DebugLogName, 'a')
        DebugHandler.setLevel(logging.DEBUG)
        DebugHandler.setFormatter(formatter)
        DebugLogger.addHandler(DebugHandler)

    os.chdir(os.pardir)

    return DebugLogger


# 装饰器函数 实现try except ,并且将异常输出至DEBUG级别的日志中
def ABS_log(message):
    def handle_func(func):
        def handle_args(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                # 初始化info日志,后续将注释掉，不用对每个函数都建立日志文件
                # infoLogger = tn_infologger(message)
                # infoLogger.info("[" + func.__name__ + "] -> " + 'run succeed')
            except:
                # 初始化debug日志
                DebugLogger = tn_debuglogger(message)
                DebugLogger.exception("[function name:" + func.__name__ + "] -> ")
                # raise ValueError

        return handle_args

    return handle_func


def f_stdout2log(logPath, func, *args, **kwargs):
    '''
    将控制台的输出重定向到日志
    :param logPath:
    :param func:
    :param args:
    :param kwargs:
    :return:
    '''

    temp = sys.stdout  # 记录当前输出指向，默认是consle ,12-25 行缓冲模式 buffering=1,
    with open(logPath, "a+", buffering=1, encoding='utf-8') as f:
        sys.stdout = f  # 输出指向txt文件
        re = func(*args, **kwargs)
        sys.stdout = temp  # 输出重定向回consle
        # print(f.readlines())
    return re


if __name__ == '__main__':
    log_ = ml_infologger('liyingkun7')
    log_.info('log test info2')

    logPath = r'C:\Users\liyin\Desktop\CcxMLOGE\liyingkun_info_2017-12-20.log'


    def test(x):
        print('ssss  test  consolcdfd dfvfvrf 李映坤')


    test(1)
    f_stdout2log(logPath, test, 1)
    print('11111')
