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
# 入力データにNaNが含まれる行がある場合は
# 付近のデータで補間する
##################################################
def interpolate_nan_input_data(input_data):
	
	# NaNでない行を取得する
	not_isnan = ~numpy.isnan(input_data)
	row_num, col_num = input_data.shape
	
	for col in range(col_num):
		not_isnan_row1 = -1
		not_isnan_row2 = -1
		
		for row in range(row_num):
			# NaNでないデータが見つかったら補間を行う
			if not_isnan[row][col]:
				# NaNでないデータの行Noを更新する
				not_isnan_row1 = not_isnan_row2
				not_isnan_row2 = row
				if (not_isnan_row2 - not_isnan_row1) == 1 :
					continue
				if not_isnan_row1 >= 0:
					# row1〜row2のうち、半分はrow1,もう半分はrow2の
					# 値で補間する
					row_mean = (not_isnan_row1 + not_isnan_row2 + 1) / 2
					
					not_isnan_data1 = input_data[row][col]
					for i in range(not_isnan_row1+1, row_mean):
						input_data[i][col] = not_isnan_data1
					
					not_isnan_data2 = input_data[row][col]
					for i in range(row_mean, not_isnan_row2):
						input_data[i][col] = not_isnan_data2
				else:
					# 先頭から初めてNaNでないデータが見つかった場合
					not_isnan_data2 = input_data[row][col]
					for i in range(0, row):
						input_data[i][col] = not_isnan_data2
		
		# 末尾の方にNaNがある場合も補間する
		not_isnan_data2 = input_data[not_isnan_row2][col]
		for i in range(not_isnan_row2+1, row_num):
			input_data[i][col] = not_isnan_data2
	
	return input_data
		
##################################################
# 出力データにNaNが含まれる行がある場合は
# 付近のデータで補間する
##################################################
def interpolate_nan_label_data(label_data):
	
	# NaNでない行を取得する
	not_isnan_row = ~numpy.isnan(label_data).any(axis=1)
	row_num, col_num = label_data.shape
	
	not_isnan_row1 = -1
	not_isnan_row2 = -1
	for row in range(row_num):
		# NaNでないデータが見つかったら補間を行う
		if not_isnan_row[row]:
			# NaNでないデータの行Noを更新する
			not_isnan_row1 = not_isnan_row2
			not_isnan_row2 = row
			if (not_isnan_row2 - not_isnan_row1) == 1 :
				continue
			if not_isnan_row1 >= 0:
				# row1〜row2のうち、半分はrow1,もう半分はrow2の
				# 値で補間する
				row_mean = (not_isnan_row1 + not_isnan_row2 + 1) / 2
				
				not_isnan_data1 = label_data[row]
				for i in range(not_isnan_row1+1, row_mean):
					label_data[i] = not_isnan_data1
				
				not_isnan_data2 = label_data[row]
				for i in range(row_mean, not_isnan_row2):
					label_data[i] = not_isnan_data2
			else:
				# 先頭から初めてNaNでないデータが見つかった場合
				not_isnan_data2 = label_data[row]
				for i in range(0, row):
					label_data[i] = not_isnan_data2
		
	# 末尾の方にNaNがある場合も補間する
	not_isnan_data2 = label_data[not_isnan_row2]
	for i in range(not_isnan_row2+1, row_num):
		label_data[i] = not_isnan_data2

	return label_data

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

