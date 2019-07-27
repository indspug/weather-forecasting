# -*- coding: utf-8 -*-

"""
データを加工するためのモジュール　
"""

import math, numpy
import re

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

##################################################
# 入力データ・出力データにNaNが含まれる行を削除する
##################################################
def remove_nan_data(input_data, label_data):
	
	# NaNの行を取得する
	isnan_row_input = numpy.isnan(input_data).any(axis=1)
	isnan_row_label = numpy.isnan(label_data).any(axis=1)
	
	# 入力データと出力データどちらか片方にNanが含まれる行を取得する
	isnan_row = numpy.logical_or(isnan_row_input, isnan_row_label)
	
	# NaNが含まれる行を削除する
	input_data = input_data[~isnan_row,]
	label_data = label_data[~isnan_row,]
	
	return input_data, label_data

##################################################
# 年月日時から各要素(年,月,日,時)を抽出する
##################################################
def extract_element_from_datetime(datetime):
	
	# 正規表現で抽出する
	#   YYYY/MM/DD hh:mm:ss (0埋め無し)
	regex = r'(\d{4})/(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})'
	pattern = re.compile(regex)
	
	data_num = len(datetime)
	elements = numpy.zeros( (data_num,6), dtype=float)
	
	for i in range(data_num):
		
		dt = datetime[i]
		result = pattern.match(dt)	
		
		if result == None:
			elements[i] = numpy.zeros(6, dtype=float)
		else:
			groups = [result.group(1), result.group(2), result.group(2), \
					result.group(4), result.group(5), result.group(6)]
			elements[i] = numpy.array( [float(s) for s in groups] )
		
	return elements
