# -*- coding: utf-8 -*-

"""
  共通処理
"""

from consts import *
import numpy

##################################################
# 入力データ(input_data)からデータの列数を取得する
##################################################
def get_col_num(input_data):
	
	col_num = len(input_data[ROW_HEADER_START])
	return col_num
	
##################################################
# 入力データ(input_data)からヘッダーの行数を取得する
# 最初の2行は飛ばした行数を返す
##################################################
def get_header_row_num(input_data):
	
	row_num = 0
	max_row_num = len(input_data)
	for i in range(ROW_HEADER_START-1, max_row_num-1):
		# '年月日時'以外で空白以外の行までがヘッダー
		name = input_data[i][0]
		if (name != '年月日時') and (len(name) > 0) :
			row_num = (i+1) - ROW_HEADER_START
			break
	
	return row_num
	
##################################################
# 入力データ(input_data)から
# 指定したデータが格納された列のインデックスを取得する。
# 地点名称, データ名称1, データ名称2, データ名称3を指定する。
# [Args]
#   inputa_data : 入力データ
#   point_name  : 地点名称
#   data_name1  : データ名称1
#   data_name2  : データ名称2
#   data_name3  : データ名称3(省略可能)
# [Return]
#   col_index   : 指定した名称のデータ格納されている列のインデックス
#                 該当する列が見付からない場合は-1
##################################################
def get_col_index(input_data, point_name, data_name1, data_name2, data_name3=None):
	
	
	# ヘッダーの行数によって格納されているインデックスが異なるので、
	# 各データが格納されている行インデックスを計算する
	header_row_num = get_header_row_num(input_data)
	if header_row_num == 4:
		# ヘッダーが4行有り、data_name3指定無し
		if data_name3 is None:
			point_index = ROW_HEADER_START - 1
			data_index1 = ROW_HEADER_START
			data_index2 = ROW_HEADER_START + 2
			data_index3 = -1
		# ヘッダーが4行有り、data_name3指定有り
		else:
			point_index = ROW_HEADER_START - 1
			data_index1 = ROW_HEADER_START
			data_index2 = ROW_HEADER_START + 1
			data_index3 = ROW_HEADER_START + 2
	else:
		# ヘッダーが3行
		point_index = ROW_HEADER_START - 1
		data_index1 = ROW_HEADER_START
		data_index2 = ROW_HEADER_START + 1
		data_index3 = -1
	
	# 列数分ループ
	col_num = len(input_data[ROW_HEADER_START])
	col_index = -1
	
	if data_index3 < 0:
		for i in range(col_num):
			pname = input_data[point_index][i]
			name1 = input_data[data_index1][i]
			name2 = input_data[data_index2][i]
			if (pname == point_name ) and \
			   (name1 == data_name1 ) and \
			   (name2 == data_name2 ) :
				col_index = i
				break
	else:
		for i in range(col_num):
			pname = input_data[point_index][i]
			name1 = input_data[data_index1][i]
			name2 = input_data[data_index2][i]
			name3 = input_data[data_index3][i]
			if (pname == point_name ) and \
			   (name1 == data_name1 ) and \
			   (name2 == data_name2 ) and \
			   (name3 == data_name3 ) :
				col_index = i
				break
					
	return col_index
	
##################################################
# 入力データ(input_data)の指定した列の値(float)を
# ndarrayに格納して返す。
# 品質情報が一定値(ACCEPTABLE_QUALITY)異常なら正常値を設定し、
# それ以外の場合はNaNを設定する
##################################################
def get_value_array(input_data, value_index, quality_index):
	
	# 最初の5,6行はヘッダーなので読み飛ばす
	data_start_index = get_header_row_num(input_data) + ROW_HEADER_START - 1 
	data_num = len(input_data) - data_start_index
	value_array = numpy.zeros(data_num, dtype=float)
	
	# 品質情報有り
	if quality_index >= 0:
		for i in range(data_num):
			index = i + data_start_index
			value = float(input_data[index][value_index])
			quality = int(input_data[index][quality_index])
			if quality >= ACCEPTABLE_QUALITY:
				value_array[i] = value
			else:
				value_array[i] = numpy.nan
	# 品質情報無し
	else:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			if not value:
				value_array[i] = numpy.nan
			else:
				value_array[i] = float(value)
	
	return value_array
	
