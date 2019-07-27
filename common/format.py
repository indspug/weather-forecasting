# -*- coding: utf-8 -*-

import sys, math, numpy

##################################################
# 定数
##################################################
ROW_HEADER_START = 3
ROW_PLACE_NAME = ROW_HEADER_START
ROW_ITEM_NAME = ROW_HEADER_START + 1
ROW_DATA_START = 6

# 最低限許容できる品質
# (値の意味 8:欠損なし, 5:20%以下の欠損, 4,3,2,1:20%を超える欠損)
ACCEPTABLE_QUALITY = 5

# 風向を角度(-PI~PI)に変換する
WIND_DIRECTION_MAP = {
  '北': 0,          '北北東': math.pi/8,    '北東': math.pi/4,    '東北東': math.pi*3/8,
  '東': math.pi/2,  '東南東': math.pi*5/8,  '南東': math.pi*3/4,  '南南東': math.pi*7/8, 
  '南': math.pi,    '南南西': -math.pi*7/8, '南西': -math.pi*3/4, '西南西': -math.pi*5/8,
  '西': -math.pi/2, '西北西': -math.pi*3/8, '北西': -math.pi/4,   '北北西': -math.pi/8,
  '静穏':0
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
# 指定したデータが品質情報を持つか否かを返す。
# [Args]
#   inputa_data : 入力データ
#   point_name  : 地点名称
#   data_name1  : データ名称
#   data_name2  : データ名称2(省略可能)
# [Return]
#   result      : 品質情報を持っている場合はTrue, 
#                 持っていない場合はFalse
##################################################
#def has_quality(input_data, point_name, data_name1, data_name2=None):
#	
#	# 品質情報が格納されている行を計算する
#	# 格納されているとしたらヘッダーの一番下の行
#	header_row_num = get_header_row_num(input_data)
#	quality_row = (ROW_HEADER_START-1) + header_row_num
#	
#	# 全列を検索する
#	col_num = get_col_num
#	result = False
#	if data_name2 is None:
#		
#		# data_name2は省略の場合
#		for i in range(col_num):
#			
#			pname = input_data[ROW_HEADER_START-1][i]
#			name1 = input_data[ROW_HEADER_START  ][i]
#			info  = input_data[quality_row][i]
#			if (pname == point_name ) and \
#			   (name1 == data_name1 ) and \
#			   (info  == '品質情報' ) :
#				result = True
#				break
#	else:
#		# data_name2は指定有りの場合
#		for i in range(col_num):
#			
#			pname = input_data[ROW_HEADER_START-1][i]
#			name1 = input_data[ROW_HEADER_START  ][i]
#			name2 = input_data[ROW_HEADER_START+1][i]
#			info  = input_data[quality_row][i]
#			if (pname == point_name ) and \
#			   (name1 == data_name1 ) and \
#			   (name2 == data_name2 ) and \
#			   (info  == '品質情報' ) :
#				result = True
#				break
#					
#	return col_index
	
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
# 入力データ(input_data)から
# 品質情報がか格納された列のインデックスを取得する。
# [Args]
#   inputa_data : 入力データ
#   point_name  : 地点名称
#   data_name1  : データ名称1
#   data_name2  : データ名称2(省略可能)
# [Return]
#   col_index   : 指定したデータの品質情報が格納されている列のインデックス
#                 該当する列が見付からない場合は-1
##################################################
#def get_quality_col_index(input_data, point_name, data_name1, data_name2=None):
#	
#	if data_name2 is None:
#		col_index = get_col_index(input_data, point_name, data_name1, '品質情報')
#	else:
#		col_index = get_col_index(input_data, point_name, data_name1, data_name2, '品質情報')
#	return col_index
	
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
	
##################################################
# 入力データ(input_data)から気温を抽出し整形して返す
##################################################
def get_temperature(input_data, point_name):
	
	# 気温の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '気温', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 気温の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '気温', '品質情報')
	
	# 気温の値のみの配列を取得する
	temperature = get_value_array(input_data, value_index, quality_index)
	
	return temperature
	
#################################################
# 入力データ(input_data)から降水量を抽出し整形して返す
##################################################
def get_rainfall(input_data, point_name):
	
	# 降水量の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '降水量(mm)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 降水量の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '降水量(mm)', '品質情報')
	
	#isnot_rain_index = get_col_index3(input_data, '降水量(mm)', '', '現象なし情報')
	
	# 降水量の値のみの配列を取得する
	rainfall = get_value_array(input_data, value_index, quality_index)
	
	return rainfall
	
##################################################
# 入力データ(input_data)から風速(速度)を
# 抽出し整形して返す
##################################################
def get_wind_speed(input_data, point_name):
	
	# 風速(速度)の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '風速(m/s)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 風速(速度)の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '風速(m/s)', '品質情報')
	
	# 風速(速度)の値のみの配列を取得する
	wind_speed = get_value_array(input_data, value_index, quality_index)
	
	return wind_speed
	
##################################################
# 入力データ(input_data)から風速(風向き)を
# 抽出し整形して返す
##################################################
def get_wind_direction(input_data, point_name):
	
	# 風速(風向き)の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '風速(m/s)', '風向', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 風速(風向き)の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '風速(m/s)', '風向', '品質情報')
	
	# 最初の5,6行はヘッダーなので読み飛ばす
	data_start_index = get_header_row_num(input_data) + ROW_HEADER_START - 1 
	data_num = len(input_data) - data_start_index
	wind_dir_array = numpy.zeros(data_num, dtype=float)
	
	# 品質情報有り
	if quality_index >= 0:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			quality = int(input_data[index][quality_index])
			if quality >= ACCEPTABLE_QUALITY:
				wind_dir_array[i] = WIND_DIRECTION_MAP[value]
			else:
				wind_dir_array[i] = numpy.nan
	# 品質情報無し
	else:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			if not value:
				wind_dir_array[i] = numpy.nan
			else:
				wind_dir_array[i] = WIND_DIRECTION_MAP[value]
	
	return wind_dir_array
	
##################################################
# 入力データ(input_data)から相対湿度を抽出し整形して返す
##################################################
def get_humidity(input_data, point_name):
	
	# 相対湿度の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '相対湿度(％)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 相対湿度の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '相対湿度(％)', '品質情報')
	
	# 相対湿度の値のみの配列を取得する
	humidity = get_value_array(input_data, value_index, quality_index)
	
	return humidity
	
##################################################
# 入力データ(input_data)から日照時間を抽出し整形して返す
##################################################
def get_daylight(input_data, point_name):
	
	# 日照時間の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '日照時間(時間)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 日照時間の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '日照時間(時間)', '品質情報')
	
	# 日照時間の値のみの配列を取得する
	daylight = get_value_array(input_data, value_index, quality_index)
	
	return daylight

#################################################
# 入力データ(input_data)から現地気圧を抽出し整形して返す
##################################################
def get_atom_pressure(input_data, point_name):
	
	# 現地気圧の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '現地気圧(hPa)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 現地気圧の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '現地気圧(hPa)', '品質情報')
	
	# 現地気圧の値のみの配列を取得する
	atom_pressure = get_value_array(input_data, value_index, quality_index)
	
	return atom_pressure
	
#################################################
# 入力データ(input_data)から海面気圧を抽出し整形して返す
##################################################
def get_sea_level_pressure(input_data, point_name):
	
	# 海面気圧の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '海面気圧(hPa)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 海面気圧の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '海面気圧(hPa)', '品質情報')
	
	# 海面気圧の値のみの配列を取得する
	sea_level_pressure = get_value_array(input_data, value_index, quality_index)
	
	return sea_level_pressure
	
##################################################
# 入力データ(input_data)から天気を抽出し整形して返す
# (晴れ、曇り、雨に分類)
##################################################
def get_weather(input_data, point_name):
	
	# 天気の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '天気', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 天気の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '天気', '品質情報')
	
	# 最初の5,6行はヘッダーなので読み飛ばす
	data_start_index = get_header_row_num(input_data) + ROW_HEADER_START - 1 
	data_num = len(input_data) - data_start_index
	value_array = numpy.zeros(data_num, dtype=float)
	weather_array = numpy.zeros((data_num, WEATHER_CLASS_NUM), dtype=float)
	
	# 品質情報有り
	if quality_index >= 0:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			quality = int(input_data[index][quality_index])
			if quality >= ACCEPTABLE_QUALITY:
				value = int(value)
				j = WEATHER_CLASSIFY_MAP[value]
				weather_array[i,j] = 1.0
			else:
				weather_array[i,0] = numpy.nan
	# 品質情報無し
	else:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			if not value:
				weather_array[i,0] = numpy.nan
			else:
				value = int(value)
				j = WEATHER_CLASSIFY_MAP[value]
				weather_array[i,j] = 1.0
	
	return weather_array
	


