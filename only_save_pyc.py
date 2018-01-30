import sys
import os
import compileall

def change_file(packageName):
    path = ""
    for i in sys.path:
        if str(i).endswith("site-packages"):  # 找到第三方包
            path = str(i)
            break
    # 没有成功获取到地址信息
    if path == "":
        return "don't get path"
    path = os.path.join(path,packageName)
    if os.path.exists(path):
        compileall.compile_dir(path)
        for a in os.walk(path):
            file_root = a[0]    #全路径
            if len(a[-1]) > 0:
                for filename in a[-1]:   #文件名
                    pathfile = os.path.join(file_root,filename)
                    if str(filename).endswith(".py"):
                        os.remove(pathfile)     #删除.py文件
                    if str(filename).endswith(".pyc"):
                        if str(os.path.basename(file_root))=='__pycache__':
                            file_root_new = file_root.replace("__pycache__","")
                            file_new_name = str(filename).split(".")[0] + ".pyc"
                            try:
                                os.rename(pathfile,file_root_new + file_new_name)
                            except Exception as err:
                                print(err)
                                break
    return "success"