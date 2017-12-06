"""
模型日志模块 主要为日志格式定义
"""

import logging
import datetime
import os
from ccxMLogE.config import LOGFILEPATH

# 创建一个info级别的日志文件，按天创建
def tn_infologger(message):
    if os.path.exists(LOGFILEPATH):
        os.chdir(LOGFILEPATH)
    else:
        os.mkdir(LOGFILEPATH)
        os.chdir(LOGFILEPATH)

    format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s:\n %(message)s'
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    infoLogName = r'%s_info_%s.log' % (message, curDate)

    formatter = logging.Formatter(format)

    infoLogger = logging.getLogger('%s_info_%s.log' % (message, curDate))

    #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not infoLogger.handlers:
        infoLogger.setLevel(logging.INFO)

        infoHandler = logging.FileHandler(infoLogName, 'a')
        infoHandler.setLevel(logging.INFO)
        infoHandler.setFormatter(formatter)
        infoLogger.addHandler(infoHandler)

    os.chdir(os.pardir)

    return infoLogger


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
