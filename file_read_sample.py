# -*- coding: utf-8 -*-
'''
  気象庁データ(CSV形式)読み込みのサンプルソース。
'''
  
import sys, os
from common.file import *
from format.extract import *
import csv
import numpy

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	input_data_list=[]
	
	# (1) 指定フォルダのCSVファイルを読み込む
	dir_path = './train'
	
	# (1-1) 指定フォルダ内のCSVファイルのパスを取得
	csv_paths = get_filepaths(dir_path, '.csv')
        for csv_path in csv_paths:
		
		# (1-2) CSVファイル読み込み
		#       この時点では単なる文字列の形式
		csv_data = read_weather_csv(csv_path)
		
		# (1-3) 指定した気象データ抽出。
		#       単なる文字列からnumpy形式に変換する。
		point_name = '水戸'
		datetime = get_datetime(csv_data)
		temperature = get_temperature(csv_data, point_name)
		rainfall = get_rainfall(csv_data, point_name)
		# 降雪
		# 積雪
		daylight = get_daylight(csv_data, point_name)
		wind_speed = get_wind_speed(csv_data, point_name)
		wind_direction = get_wind_direction(csv_data, point_name)
		# 日射量
		atom_pressure = get_atom_pressure(csv_data, point_name)
		sea_level_pressure = get_sea_level_pressure(csv_data, point_name)
		humidity = get_humidity(csv_data, point_name)
		# 蒸気圧
		# 露天湿度
		get_weather(csv_data, point_name)
		# 雲量
		# 視程
		
		# (1-4) 入力データ結合。列を合体するイメージ。
		#       以下は、降水量,相対湿度,海面気圧を結合する例。
		#
		#       (実際のデータ例)
		#         0.5,1000.2,86
		#         ...
                input_data = numpy.stack(
                        [rainfall, humidity, sea_level_pressure], 1)
		
		# (1-5) 結合したデータを一旦リストに退避
		input_data_list.append(input_data)
	
	# (1-6) 全CSVファイルのデータを結合する。
	#       こちらは(1-4)と異なり、縦方向の結合
	input_data = input_data_list[0]
	for i in range(2, len(input_data_list)):
        	input_data = numpy.vstack( [input_data, input_data_list[i]] )

       
	print input_data
