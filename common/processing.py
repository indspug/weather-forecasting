# -*- coding: utf-8 -*-

"""
データを加工するためのモジュール　
"""

import math, numpy
import re

##################################################
# データ(data_list)をMAX-MIN標準化した結果を返す
# [Args]
#   data_list : 標準化するデータ(ndarrayのリスト)
#   axis      : 最大値を求める軸
# [Return]
#   MAX-MIN標準化したデータ
##################################################
def max_min_normalize(data_list, axis=None):
	
	# 各データ群の最大値・最小値を取得する
	max_list = []
	min_list = []
	for data in data_list:
		max_list.append(data.max(axis=axis))
		min_list.append(data.min(axis=axis))
	
	# 各データ群の最大値・最小値を統合する
	max_array = max_list[0]
	for i in range(2, len(max_list)):
		numpy.vstack(max_array, max_list[i])
	
	min_array = min_list[0]
	for i in range(2, len(min_list)):
		numpy.vstack(min_array, min_list[i])
	
	max = max_array.max(axis=axis)
	min = min_array.min(axis=axis)
	
	# 各データをMAX-MIN標準化する
	result_list = []
	for data in data_list:
		result = (data - min) / (max - min)
		result_list.append(result)
	
	return result_list

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
		
		# 正規表現にマッチするか確認する
		dt = datetime[i]
		result = pattern.match(dt)	
		
		if result == None:
			# マッチしなかった場合は全て0
			elements[i] = numpy.zeros(6, dtype=float)
		else:
			# マッチした場合は、各要素を抽出する
			groups = [result.group(1), result.group(2), result.group(2), \
					result.group(4), result.group(5), result.group(6)]
			elements[i] = numpy.array( [float(s) for s in groups] )
		
	return elements

