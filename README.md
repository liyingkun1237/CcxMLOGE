
# 中诚信征信机器学习平台
<br></br>
## 更新日志

## 2017-12-12 整个流程走通
### 1.包括 数据描述 数据探索性分析 IV值分析 模型训练 模型报告生成 模型导出等功能
### 2.调整版本为 0.1.0
### 3.新增run.sh 文件 便于上线后的启动服务和暂停服务
<br></br>
## 2017-12-19 更新记录
### 1.将np.round 修改为 round 
<br></br>
## 2017-12-20 更新记录
### 1.新增了info日志并将记录的日志在前端实时展示
<br></br>
## 2017-12-25 更新记录
### 1.对于ccxmodel模块，对于数据量过少时模型无法正常计算的问题进行了修改，残留问题是，惩罚项过大的问题
### 2.对于日志突兀的问题，将缓冲机制改为了行缓冲，待测试。
<br></br>
## 2017-12-27 更新记录
### 1.新增了有监督查询的计算接口
### 2.修改了html中的文件覆盖的bug，bug原因，同一天的文件会被覆盖
### 3.提交代码，并修改版本号为0.1.1
<br></br>
## 2018-01-10 更新记录
### 1.新增了和计费系统通信的计费接口 并写了entrance函数
### 2.代码混淆由borui基本完成初版，实现了先混淆后编译的方法
### 3.解决了之前由于数据量较少，概率分箱出错的问题
