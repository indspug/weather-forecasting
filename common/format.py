# -*- coding: utf-8 -*-

import sys, math, numpy

##################################################
# 定数
##################################################
ROW_DATA_START = 6

# 風向を角度(-PI~PI)に変換する
WIND_DIRECTION_MAP = {
  '北': 0,          '北北東': math.pi/8,    '北東': math.pi/4,    '東北東': math.pi*3/8,
  '東': math.pi/2,  '東南東': math.pi*5/8,  '南東': math.pi*3/4,  '南南東': math.pi*7/8, 
  '南': math.pi,    '南南西': -math.pi*7/8, '南西': -math.pi*3/4, '西南西': -math.pi*5/8,
  '西': -math.pi/2, '西北西': -math.pi*3/8, '北西': -math.pi/4,   '北北西': -math.pi/8
}

# 天気の分類ルール
WEATHER_CLASS_NUM = 3
WEATHER_SUNNY = 0	# 晴れ
WEATHER_CLOUDY = 1	# くもり
WEATHER_RAINY = 2	# 雨
WEATHER_CLASSIFY_MAP = {
  1  : WEATHER_SUNNY,	# 快晴
  2  : WEATHER_SUNNY,	# 晴れ
  3  : WEATHER_CLOUDY,	# 薄曇
  4  : WEATHER_CLOUDY,	# 曇
  5  : WEATHER_CLOUDY,	# 煙霧
  6  : WEATHER_CLOUDY,	# 砂じん嵐
  7  : WEATHER_CLOUDY,	# 地ふぶき
  8  : WEATHER_RAINY,	# 霧
  9  : WEATHER_RAINY,	# 霧雨
  10 : WEATHER_RAINY,	# 雨
  11 : WEATHER_RAINY,	# みぞれ
  12 : WEATHER_RAINY,	# 雪
  13 : WEATHER_RAINY,	# あられ
  14 : WEATHER_RAINY,	# ひょう
  15 : WEATHER_RAINY,	# 雷
  16 : WEATHER_RAINY,	# しゅう雨または止み間のある雨	
  17 : WEATHER_RAINY,	# 着氷性の雨	
  18 : WEATHER_RAINY,	# 着氷性の霧雨	
  19 : WEATHER_RAINY,	# しゅう雪または止み間のある雪	
  22 : WEATHER_RAINY,	# 霧雪
  23 : WEATHER_RAINY,	# 凍雨
  24 : WEATHER_RAINY,	# 細氷
  28 : WEATHER_RAINY,	# もや
  101: WEATHER_RAINY,	# 降水またはしゅう雨性の降水
}

##################################################
# 入力データ(input_data)からデータの列数を取得する
##################################################
def get_col_num(input_data):
	
	col_num = len(input_data[3])
	return col_num
	
##################################################
# 入力データ(input_data)から
# 指定したデータが格納された列のインデックスを取得する。
# データの名称が3行あるうちの1行目の名称をvalue_name1,
# 2行目の名称をvalue_name2に指定する。
# [Args]
#   inputa_data : 入力データ
#   value_name1 : データの名称1
#   value_name2 : データの名称2(省略可)
# [Return]
#   value_index   : 値自体(float)が格納されている列のインデックス
#   quality_index : 品質番号が格納されている列のインデックス
##################################################
def get_col_index(input_data, value_name1, value_name2=None):
	
	col_num = len(input_data[3])
	
	value_index = 0
	quality_index = 0
	
	for i in range(col_num):
		
		# value_name1を含むデータを検索する
		name = input_data[3][i]
		if name.find(value_name1) >= 0 :
			name2 = input_data[4][i]
			info = input_data[5][i]
			
			# value_name2が指定無しの場合は
			# value_name1のみで判断する
			if value_name2 is None:
				
				# name2が空文字列(指定なし)の行のみ扱う
				if not name2 :
					if not info:
						value_index = i
					elif info.find('品質情報') >= 0 :
						quality_index = i
			
			# value_name2が指定されている場合は合致するデータを検索
			elif name2.find(value_name2) >= 0 :
				if not info:
					value_index = i
				elif info.find('品質情報') >= 0 :
					quality_index = i
			
	return (value_index, quality_index)
	
##################################################
# 入力データ(input_data)から
# 指定したデータが格納された列のインデックスを取得する。
# データの名称が3行あるうちの
# 1行目の名称をvalue_name1, 2行目の名称をvalue_name2, 
# 3行目の名称をvalue_name3に指定する。
# [Args]
#   inputa_data : 入力データ
#   value_name1 : データの名称1
#   value_name2 : データの名称2
#   value_name3 : データの名称3
# [Return]
#   col_index   : 指定した名称のデータ格納されている列のインデックス
##################################################
def get_col_index3(input_data, value_name1, value_name2, value_name3):
	
	col_num = len(input_data[3])
	
	col_index = 0
	
	for i in range(col_num):
		
		name1 = input_data[3][i]
		name2 = input_data[4][i]
		name3 = input_data[5][i]
		if (name1 == value_name1) and \
		   (name2 == value_name2) and \
		   (name3 == value_name3) :
			col_index = i
				
	return col_index
	
##################################################
# 入力データ(input_data)の指定した列の値(float)を
# ndarrayに格納して返す。
# 品質情報が8(正常値),5(20%以下の欠損)なら正常値を設定し、
# それ以外の場合はNaNを設定する
##################################################
def get_value_array(input_data, value_index, quality_index):
	
	# 最初の6行はヘッダーなので読み飛ばす
	data_num = len(input_data) - ROW_DATA_START
	value_array = numpy.zeros(data_num, dtype=float)
	for i in range(data_num):
		index = i + ROW_DATA_START
		value = float(input_data[index][value_index])
		quality = int(input_data[index][quality_index])
		if quality >= 5:
			value_array[i] = value
		else:
			value_array[i] = numpy.nan
		
	return value_array
	
##################################################
# 入力データ(input_data)から気温を抽出し整形して返す
##################################################
def get_temperature(input_data):
	
	value_index, quality_index = get_col_index(input_data, '気温')
	temperature = get_value_array(input_data, value_index, quality_index)
	return temperature
	
#################################################
# 入力データ(input_data)から降水量を抽出し整形して返す
##################################################
def get_rainfall(input_data):
	
	#value_index, quality_index = get_col_index(input_data, '降水量')
	#rainfall = get_value_array(input_data, value_index, quality_index)
	
	value_index = get_col_index3(input_data, '降水量(mm)', '', '')
	isnot_rain_index = get_col_index3(input_data, '降水量(mm)', '', '現象なし情報')
	quality_index = get_col_index3(input_data, '降水量(mm)', '', '品質情報')
	
	# 品質情報が8(正常値),5(20%以下の欠損)なら正常値を設定し、
	# それ以外の場合はNaNを設定する
	# 最初の6行はヘッダーなので読み飛ばす
	data_num = len(input_data) - ROW_DATA_START
	rainfall = numpy.zeros((data_num), dtype=float)
	for i in range(data_num):
		index = i + ROW_DATA_START
		quality = int(input_data[index][quality_index])
		isnot_rain = int(input_data[index][isnot_rain_index])
		
		# 品質が一定以上の場合
		if quality >= 5:
			
			# 降水がある(現象なし=0)場合で、
			# 降水量が0(mm)の場合は、0.1(mm)を設定する。
			value = float(input_data[index][value_index])
			if (isnot_rain == False) and \
			   (value < sys.float_info.epsilon) :
				rainfall[i] = 0.1
			else:
				rainfall[i] = value
			
		# 品質が一定未満の場合
		else:
			rainfall[i] = numpy.nan
	
	return rainfall
	
##################################################
# 入力データ(input_data)から風速(速度と風向き)を
# 抽出し整形して返す
##################################################
def get_wind_speed(input_data):
	
	# 風速のデータの中から、
	# 風速(m/s), 風速(m/s)の品質, 風向, 風向の品質が
	# 格納されている列の列番号を取得する
	value_speed_col, quality_speed_col = \
		get_col_index(input_data, '風速')
	value_dir_col, quality_dir_col = \
		get_col_index(input_data, '風速', '風向')
	
	# 品質情報が8(正常値),5(20%以下の欠損)なら正常値を設定し、
	# それ以外の場合はNaNを設定する
	# 最初の6行はヘッダーなので読み飛ばす
	# ndarray(wind_speed_array, wind_dir_array)に結果を格納する
	data_num = len(input_data) - ROW_DATA_START
	wind_speed_array = numpy.zeros(data_num, dtype=float)
	wind_dir_array = numpy.zeros(data_num, dtype=float)
	for i in range(data_num):
		index = i + ROW_DATA_START
		
		# 風速(m/s)
		value = float(input_data[index][value_speed_col])
		quality = int(input_data[index][quality_speed_col])
		if quality >= 5:
			wind_speed_array[i] = value
		else:
			wind_speed_array[i] = numpy.nan
		
		# 風向
		value = input_data[index][value_dir_col]
		quality = int(input_data[index][quality_dir_col])
		if quality >= 5:
			# 静穏の場合は風速=0にする
			if value.find('静穏') >= 0 :
				wind_dir_array[i] = 0
			else:
				wind_dir_array[i] = WIND_DIRECTION_MAP[value]
		else:
			wind_dir_array[i] = numpy.nan
		
	return (wind_speed_array, wind_dir_array)
	
##################################################
# 入力データ(input_data)から相対湿度を抽出し整形して返す
##################################################
def get_humidity(input_data):
	
	value_index, quality_index = get_col_index(input_data, '相対湿度')
	humidity = get_value_array(input_data, value_index, quality_index)
	return humidity
	
##################################################
# 入力データ(input_data)から日照時間を抽出し整形して返す
##################################################
def get_daylight(input_data):
	value_index, quality_index = get_col_index(input_data, '日照時間')
	daylight = get_value_array(input_data, value_index, quality_index)
	return daylight

#################################################
# 入力データ(input_data)から現地気圧を抽出し整形して返す
##################################################
def get_atom_pressure(input_data):
	
	value_index, quality_index = get_col_index(input_data, '現地気圧')
	atom_pressure = get_value_array(input_data, value_index, quality_index)
	return atom_pressure
	
##################################################
# 入力データ(input_data)から天気を抽出し整形して返す
# (晴れ、曇り、雨に分類)
##################################################
def get_weather(input_data):
	
	value_index, quality_index = get_col_index(input_data, '天気')
	
	# 品質情報が8(正常値),5(20%以下の欠損)なら正常値を設定し、
	# それ以外の場合はNaNを設定する
	# 最初の6行はヘッダーなので読み飛ばす
	# ndarray(weather_array)に結果を格納する
	data_num = len(input_data) - ROW_DATA_START
	weather_array = numpy.zeros((data_num, WEATHER_CLASS_NUM), dtype=float)
	for i in range(data_num):
		index = i + ROW_DATA_START
		quality = int(input_data[index][quality_index])
		if quality >= 5:
			value = int(input_data[index][value_index])
			j = WEATHER_CLASSIFY_MAP[value]
			weather_array[i,j] = 1.0
		else:
			weather_array[i,0] = numpy.nan
		
	return weather_array
	


