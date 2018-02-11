
import numpy as np
import pickle
import pandas as pd
import flask
from flask import request
import time
import json
from ccxMLogE.predictModel import predictmodel

server1 = flask.Flask(__name__,static_folder='')

print('我在这里')


@server1.route('/ApiName', methods=['post'])
def ccxdeployModelApi():
    print('12djvcndkjfncdskfnvjdsnvkjfnvg')
    return "cdfcdvsfvdsfvdsfvgdgvdfsvdsfvfgv"


if __name__ == '__main__':
    print('我到这里了')
    server1.run(debug=True, port=1027, host='0.0.0.0', processes=3)
    import pandas as pd

    pd.read_csv(r'10.0.31.245/Ccx_Fp_ABS/lyk_A/fp_base.csv')
