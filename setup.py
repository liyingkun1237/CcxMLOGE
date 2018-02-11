from distutils.core import setup
from only_save_pyc import change_file as change_file_ccx  # 引入修改pyc文件的地址

setup(
    name='CcxMLOGE',
    version='0.2.1',
    packages=['ccxMLogE'],
    url='2018-01-08',
    license='ccx',
    author='liyingkun',
    author_email='liyingkun@ccx.cn',
    description='机器学习平台一期--万象智慧--MLogE',
    data_files=[('', ['setup.py', 'run.sh', 'only_save_pyc.py'])]
)

print(change_file_ccx("ccxMLogE"))
