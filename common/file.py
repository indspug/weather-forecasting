# -*- coding: utf-8 -*-

"""
ファイル操作をするためのモジュール　
"""

import os
import csv

##################################################
# 指定したディレクトリから
# 指定した拡張子のファイルのリストを取得する
# [Args]
#   dirpath   : ディレクトリパス　
#   extension : 拡張子
# [Rerutn]
#   filepaths : ファイルパスのリスト
##################################################
def get_filepaths(dirpath, extension):
	
	filepaths = []
	
	for filename in os.listdir(dirpath):
		# ディレクトリの場合はpass
		if os.path.isdir(dirpath + '/' + filename):
			continue
		
		# 拡張子が一致したらリストに追加
		base,ext = os.path.splitext(filename)
		if ext == extension:
			filepaths.append(dirpath + '/' + filename)
	
	return filepaths

##################################################
# CSVファイル(csv_path)から気象データを読み込む
# [Args]
#   csv_path : CSVファイルのファイルパス　
# [Rerutn]
#   csv_data : CSVから取り出したデータ(stringのlist)
##################################################
def read_weather_csv(csv_path):
	csv_data = []
	
	# Python2でも動くように変更。
	# CSVのエンコードは事前に'utf-8'に変更した。
	#  (変更前)with open(csv_path, 'r', encoding='shift_jis') as f:
	with open(csv_path, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			csv_data.append(row)
	
	return csv_data
