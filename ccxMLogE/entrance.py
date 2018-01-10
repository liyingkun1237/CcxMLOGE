"""
此脚本的作用主要为 开关函数 只有满足要求
1.底层代码不被纂改
2.通过中诚信计费系统发送的请求

后台算法才会被执行，简单理解为一个开关函数的模块

基本思想：
1.接收到前端请求时，将reqID，userName，sign等信息作为请求体，发送给计费系统
2.计费系统依据计费逻辑，核验后，给出是否运行后台算法的指令
"""
import requests
from ccxMLogE.IMPFILE import PAYMENTURL, USERNAME, PASSWORD
import hashlib
import json
from datetime import datetime


def mob2MD5(string):
    m = hashlib.md5()
    mob = string.encode('utf-8')
    m.update(mob)
    psw = m.hexdigest()
    return psw


def f_enter(reqId, random):
    '''
    接收到前端请求的reqId,发送请求至计费系统，同步接收到是否允许运行的指令
    :param reqId:
    :param random:随机码 数据的行ccx数据的列
    :return:
    '''
    # 请求计费系统
    header_dict = {"Content-Type": "application/json"}
    url = PAYMENTURL  # 线上生产环境请求接口
    userName = USERNAME
    passWord = PASSWORD
    # 加密方式 'reqId'+reqId+'userName'+userName+'passWord'+passWord+random 进行md5加密 编码类型为utf-8
    string = 'reqId' + str(reqId) + 'userName' + str(userName) + 'passWord' + str(passWord) + str(random)
    sign = mob2MD5(string)
    reqTime = datetime.today().strftime('%Y-%m-%d %H%M%S')
    reqs = json.dumps({"reqId": reqId, 'sign': sign, 'reqTime': reqTime}, ensure_ascii=False)
    reqs_ = reqs.encode('utf-8')
    r = requests.post(url, data=reqs_, headers=header_dict)
    # print('用时' * 20, (time.time() - st()))
    resp = json.loads(r.text)
    # 返回json示例 {"code":"0000"} /{'code':"0101"}
    if resp['code'] == "0000":
        return True
    elif resp['code'] == "0101":
        return False
    else:
        raise ValueError("return code out of dict")


if __name__ == '__main__':
    reqId = 'Xgboost-128-721742'
    userName = 'xiaoxinyong'
    passWord = '12345678'
    random = '1000ccx30'
    string = 'reqId' + str(reqId) + 'userName' + str(userName) + 'passWord' + str(passWord) + str(random)
    mob2MD5(string)