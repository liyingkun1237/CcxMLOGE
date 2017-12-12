"""
机器学习平台MLogE的封装接口函数
"""
import pickle
import threading

import flask
import requests
from flask import request
import time
import json
from ccxMLogE.config import f_mdAllconf
from ccxMLogE.inputTransform import f_getCateList, f_readData
from ccxMLogE.outputTransform import f_part2Output, f_type1Output, f_type2Output, f_part2Output4yibu, \
    f_modelPredictOutput
from ccxMLogE.predictModel import predictmodel, f_save_predictRes
from ccxMLogE.preparationData import f_dummyOld, f_splitdata, processData, f_genAllcol, f_saveprocessData
from ccxMLogE.trainModel import f_trainModelMain
from ccxMLogE.varDescSummary import f_mainDesc, f_viewdata, f_VarTypeClassfiy
import os

server = flask.Flask(__name__)


@server.route('/ccxModelApi', methods=['post'])
def ccxModelApi():
    # try:
    st = time.time()
    # 1.解析数据
    Input = json.loads(request.data.decode())
    reqId = Input.get('reqId')
    type = Input.get('type')
    userPath = Input.get('userPath')
    base = Input.get('base')
    fields = Input.get('fields')
    print('前端请求接口数据', Input)
    # 2.修改用户的超参数配置文件
    flag = f_mdAllconf(userPath)
    # flag 为True说明了 用户目录下有配置文件且路径配置完成
    # 3.数据预处理
    # 解析用户自定义的离散型变量
    cateList = f_getCateList(fields)
    # 读取数据
    rawdata = f_readData(base)
    # 数据概览
    datasetInfo = f_viewdata(rawdata, (base['programName'] + str(base['pId'])))

    # 1208 遇到文件多? 的bug 先自己处理一下 后续交由李龙处理
    col0 = rawdata.columns[0]
    rawdata = rawdata.rename(columns={col0: col0.split('?')[-1]})
    if type == 0:
        print('变量统计')
        # 数据的描述性分析
        resdesc = f_mainDesc(rawdata, base['indexName'], base['targetName'], cateList)
        descout, path3 = f_part2Output(resdesc, userPath, rawdata)
        res = f_type1Output(reqId, datasetInfo, descout, path3)
    elif type == 1:
        # 数据的描述性分析,在计算一遍不是很明智，但是可以想个办法
        # 主要为了对付用户调整了变量的类型之后 需要重新计算的问题
        # 起一个异步线程去跑模型
        with server.app_context():
            t = threading.Thread(target=f_threadModelTrain,
                                 args=(rawdata, base, cateList, reqId, datasetInfo, userPath))
            t.start()
        res = json.dumps({"code": 200, "msg": '模型正在运行 请耐心等待'}, ensure_ascii=False)

    print('计算用时', time.time() - st)
    return res
    # except Exception as e:
    #     return json.dumps({"code": 500, "msg": str(e)})


def f_threadModelTrain(rawdata, base, cateList, reqId, datasetInfo, userPath):
    # 会了前端计时方便 计算错误也要回调
    # try:
    print('变量统计加模型返回')
    # dummyList = f_VarTypeClassfiy(rawdata, cateList)
    resdesc = f_mainDesc(rawdata, base['indexName'], base['targetName'], cateList)
    descout, path3 = f_part2Output4yibu(resdesc, userPath)  # path3 即为所有变量的IV值计算
    # res = f_type1Output(reqId, datasetInfo, descout, path3)
    print('开始跑模型 ' * 20)
    # 模型数据的准备
    dummyList = list(set(resdesc[4]) - set(resdesc[5]))  # 需要one-hot - 多分类
    dummyAfterdf = f_dummyOld(rawdata, dummyList)
    train_path, test_path = f_splitdata(dummyAfterdf, base['targetName'])
    # 模型训练
    modeltype = f_getmodelType(base)
    train_path.index = range(len(train_path))  # 必须加 1129 发现的bug
    test_path.index = range(len(test_path))
    repathlist = f_trainModelMain(train_path, test_path, base['indexName'], base['targetName'], userPath,
                                  modeltype,
                                  base['arithmetic'])
    # 保存模型对象 供后续预测使用 1212
    modelname = modeltype.split('_')[0]
    psd = processData(modelname, dummyList, f_genAllcol(dummyAfterdf), repathlist[1])
    modelPath = f_saveprocessData(psd, reqId, userPath)

    # 模型输出结果
    res = f_type2Output(reqId, datasetInfo, descout, path3, repathlist, rawdata.columns, train_path, test_path,
                        base['targetName'], userPath, resdesc, modelPath)

    # 回调输出接口
    header_dict = {"Content-Type": "application/json"}
    # url = 'http://10.0.5.136:9999/output/api'  # 开发环境请求接口
    # url = 'http://192.168.100.175:8080/ccx-models/output/api'  # 线上测试环境请求接口
    url = 'http://127.0.0.1:8081/ccx-models/output/api'  # 线上生产环境请求接口
    res_ = res.encode('utf-8')
    r = requests.post(url, data=res_, headers=header_dict)
    # print('用时' * 20, (time.time() - st()))
    print(r.text)
    print('回调内容:正常===\n', res)
    return res
    # except Exception as e:
    #     header_dict = {"Content-Type": "application/json"}
    #     # url = 'http://10.0.5.136:9999/output/api'  # 开发环境请求接口
    #     # url = 'http://192.168.100.175:8080/ccx-models/output/api'  # 线上测试环境请求接口
    #     url = 'http://127.0.0.1:8081/ccx-models/output/api'  # 线上生产环境请求接口
    #     res = json.dumps({"code": 502, "reqId": reqId, "msg": str(e)}, ensure_ascii=False)
    #     res_ = res.encode('utf-8')
    #     r = requests.post(url, data=res_, headers=header_dict)
    #     # print('用时' * 20, (time.time() - st()))
    #     print(r.text)
    #     print('回调内容：异常===\n', res)
    #     return res


def f_getmodelType(base):
    '''
    依据前端输入的base信息 判断出用户想要跑的模型类型 12种
    :param base:
    :return:
    '''
    MODELDICT = {'Xgboost': {'demo': 'ccxboost_demo',
                             'speed': 'ccxboost_speed',
                             'accuracy': 'ccxboost_accuracy',
                             'stable': 'ccxboost_stable'
                             },
                 'GBM': {'demo': 'ccxgbm_demo',
                         'speed': 'ccxgbm_speed',
                         'accuracy': 'ccxgbm_accuracy',
                         'stable': 'ccxgbm_stable'},
                 'RF': {'demo': 'ccxrf_demo',
                        'speed': 'ccxrf_speed',
                        'accuracy': 'ccxrf_accuracy',
                        'stable': 'ccxrf_stable'}
                 }
    # base['arithmetic']  # 大的模型方向 Xgboost GBM RF
    # base['modelConf']  # 小的模型参数配置 demo speed accuracy stable
    return MODELDICT[base['arithmetic']][base['modelConf']]


@server.route('/ccxModelApi/predict', methods=['post'])
def ccxModelApiPredict():
    # try:
    st = time.time()
    # 1.解析数据
    Input = json.loads(request.data.decode())
    reqId = Input.get('reqId')
    # modelreqId = Input.get('modelreqId') # 留着后期将其处理的更严谨
    modelPath = Input.get('modelPath')
    base = Input.get('base')
    indexName = base['indexName']
    print('前端请求接口数据', Input)

    # 获取到保存下来的processData 对象

    processData = f_load(modelPath)

    # 读取待预测的数据集
    test = f_readData(base)

    # 进行预测
    res = predictmodel(processData, test, indexName)

    # 结果保存
    predictResPath = f_save_predictRes(res, modelPath)

    # 正常情况下 返回结果
    rest = f_modelPredictOutput(reqId, predictResPath)
    print('返回预测接口的结果', rest)
    return rest

    # except Exception as e:
    # return json.dumps({'code': 503, 'Msg': str(e)}, ensure_ascii=False)


def f_load(modelPath):
    with open(modelPath, 'rb') as f:
        re = pickle.load(f)
    return re


if __name__ == '__main__':
    server.run(debug=True, port=6060, host='0.0.0.0')
