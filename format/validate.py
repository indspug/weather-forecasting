# -*- coding: utf-8 -*-

"""
  気象データの中身を検証する
"""

from consts import *
import numpy

##################################################
# 降水量と天気の不整合を検証し、
# 不整合なデータのインデックスにTrueを格納したリストを返す
# [Args]
#   raifall : 降水量
#   weather : 天気
# [Return]
#   Booleanのリスト(ndarray)
#     不整合データはTrue, それ以外はFalse
##################################################
def validate_rainfall_inconsistency(rainfall, weather):
	
	is_inconsistency = []
	data_num = min([rainfall.shape[0], weather.shape[0]])
	
	for i in range(data_num):
		
		is_rain = weather[i][WEATHER_RAINY]
		
		# 降水量が0より大きくて、天気が雨以外
		if (rainfall[i] > 0) and (is_rain == 0):
			is_inconsistency.append(True)
		
		# 降水量が0以下で、天気が雨
		elif (rainfall[i] <= 0) and (is_rain == 1):
			is_inconsistency.append(True)
		
		else:
			is_inconsistency.append(False)
	
	return numpy.array(is_inconsistency)

