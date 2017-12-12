"""
配置脚本路径
#1.修改模型超参数配置文件的项目路径
"""
import os

from ccxmodel.modelconf import ModelConf


def f_mdConfproPath(userPath, confname):
    '''
    修改用户conf中的项目路径
    :param userPath: 用户路径
    :param confname: 需要修改的配置文件名
    :return: 修改之后 并且保存的配置文件
    需要解决的问题，如何标记当前用户已经修改过了 不用重复此操作了
    思路：返回一个flag标签，对于首次操作的用户，操作之后，返回false 以后就都不用修改了
    '''
    conf_path = os.path.join(userPath, 'conf')

    if os.path.exists(conf_path):
        # 路径存在时
        pathConf = os.path.join(conf_path, confname)
        if os.path.exists(pathConf):
            # 判断出来 超参数配置文件是否存在
            if 'ccxboost' in confname:
                # 判断出来 这是xgboost的配置文件，修改路径，修改后返回一个bool值
                mc = ModelConf(pathConf)
                mc.set_projectdir(proj_path=os.path.join(userPath, 'ccxboost/'))
                return True
            elif 'ccxgbm' in confname:
                # 判断出来 这是xgboost的配置文件，修改路径，修改后返回一个bool值
                mc = ModelConf(pathConf)
                mc.set_projectdir(proj_path=os.path.join(userPath, 'ccxgbm/'))
                return True
            elif 'ccxrf' in confname:
                # 判断出来 这是xgboost的配置文件，修改路径，修改后返回一个bool值
                mc = ModelConf(pathConf)
                mc.set_projectdir(proj_path=os.path.join(userPath, 'ccxrf/'))
                return True
        else:
            # 写日志，返回错误码 ，二期
            print('错误码002 找不到超参数配置文件')
            pass
    else:
        # 写日志，返回错误码 ，二期
        print('错误码001 用户路径下没有conf文件夹')
        pass


def f_mdAllconf(userPath):
    confnameLs = ['ccxboost_demo.conf', 'ccxboost_speed.conf', 'ccxboost_accuracy.conf', 'ccxboost_stable.conf',
                  'ccxgbm_demo.conf', 'ccxgbm_speed.conf', 'ccxgbm_accuracy.conf', 'ccxgbm_stable.conf',
                  'ccxrf_demo.conf', 'ccxrf_speed.conf', 'ccxrf_accuracy.conf', 'ccxrf_stable.conf'
                  ]
    ls = []
    for confname in confnameLs:
        _ = f_mdConfproPath(userPath, confname)
        ls.append(_)
        # print('项目路径已修改')
    return all(ls)  # 所有的都修改了


# 日志的路径
# ProjectPATH = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
# LOGFILEPATH = os.path.join(ProjectPATH, 'Log')
#
# 发现这么做的一个不好处 有问题
LOGFILEPATH = '/ccxMLogE/Log'  # 指定路径
# LOGFILEPATH = r'C:\Users\liyin\Desktop\CcxMLOGE'  # 指定路径

if __name__ == '__main__':
    # 测试一下
    userPath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\conf'
    re = f_mdAllconf(userPath)
    print(re)
