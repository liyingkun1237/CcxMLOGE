
from fabric.api import *

env.hosts = 'localhost'

def deploy():
    with lcd('/root/jupyterhub/CcxMLOGE/Testdeploy/modelDB/rtryyv01'):
        local('chmod 755 runmodel.sh')
        local('bash runmodel.sh start && sleep 1')
        print('本地的模型服务启动成功')


