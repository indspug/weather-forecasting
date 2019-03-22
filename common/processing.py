# -*- coding: utf-8 -*-

import math, numpy

##################################################
# データ(data)をMAX-MIN標準化した結果を返す
##################################################
def max_min_normalize(data, axis=None):
	
	max = data.max(axis=axis)
	min = data.min(axis=axis)
	result = (data - min) / (max - min)
	return result

