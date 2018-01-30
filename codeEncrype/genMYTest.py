"""
此文件的主要作用为生成秘钥类，对关键信息进行加密与解密
"""

from codeEncrype.CFCrypto import StringCrypto
from ccxMLogE.IMPFILE import USERNAME, PASSWORD
import pickle

# 生成加密字符串的类

stringcrypto = StringCrypto('lykccx19910701')

# 加密账户名和密码
user = stringcrypto.encrypt(USERNAME)
word = stringcrypto.encrypt(PASSWORD)

# 保存解密的类

keyfile = r'C:\Users\liyin\Desktop\CcxMLOGE\ccxMLogE\exdata\key.ccx'
with open(keyfile, 'wb') as f:
    pickle.dump(stringcrypto, f)

with open(keyfile, 'rb+') as f:
    sc = pickle.load(f)
