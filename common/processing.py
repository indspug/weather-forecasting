# -*- coding: utf-8 -*-

"""
データを加工するためのモジュール　
"""

import math, numpy

##################################################
# データ(data)をMAX-MIN標準化した結果を返す
# [Args]
#   data : 標準化するデータ(ndarray)
#   axis : 最大値を求める軸
# [Return]
#   MAX-MIN標準化したデータ
##################################################
def max_min_normalize(data, axis=None):
	
	max = data.max(axis=axis)
	min = data.min(axis=axis)
	result = (data - min) / (max - min)
	return result

