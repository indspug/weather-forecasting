# -*- coding: utf-8 -*-

"""
  入力データから指定した気象データを取り出す
"""

from consts import *
from common import *
	
##################################################
# 入力データ(input_data)から気温を抽出し整形して返す
##################################################
def get_temperature(input_data, point_name):
	
	# 気温の値が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '気温(℃)', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 気温の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '気温(℃)', '品質情報')
	
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
	
#################################################
# 入力データ(input_data)から年月日時を抽出して返す
##################################################
def get_datetime(input_data):
	
	# 年月日時の値が格納されている列のインデックスを取得する
	col = get_col_index(input_data, '', '年月日時', '')
	if col < 0 :
		raise Exception('No data')
	
	# 年月日時は品質情報無し
	
	# 最初の5,6行はヘッダーなので読み飛ばす
	data_start_index = get_header_row_num(input_data) + ROW_HEADER_START - 1 
	data_num = len(input_data) - data_start_index
	
	datetime = []
	for i in range(data_num):
		row = i + data_start_index
		datetime.append( input_data[row][col])
	
	return datetime
	
##################################################
# 入力データ(input_data)から雲量を
# 抽出し整形して返す
##################################################
def get_cloud_cover(input_data, point_name):
	
	# 雲量が格納されている列のインデックスを取得する
	value_index = get_col_index(input_data, point_name, '雲量(10分比)', '', '')
	if value_index < 0 :
		raise Exception('No data')
	
	# 雲量の品質情報が格納されている列のインデックスを取得する
	quality_index = get_col_index(input_data, point_name, '雲量(10分比)', '', '品質情報')
	
	# 最初の5,6行はヘッダーなので読み飛ばす
	data_start_index = get_header_row_num(input_data) + ROW_HEADER_START - 1 
	data_num = len(input_data) - data_start_index
	cloud_cover = numpy.zeros(data_num, dtype=float)
	
	# 品質情報有り
	if quality_index >= 0:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			quality = int(input_data[index][quality_index])
			if quality >= ACCEPTABLE_QUALITY:
				if value == '0+':
					cloud_cover[i] = 0.5
				elif value == '10-':
					cloud_cover[i] = 9.5
				else:
					cloud_cover[i] = float(value)
			else:
				cloud_cover[i] = numpy.nan
	# 品質情報無し
	else:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			if not value:
				cloud_cover[i] = numpy.nan
			else:
				if value == '0+':
					cloud_cover[i] = 0.5
				elif value == '10-':
					cloud_cover[i] = 9.5
				else:
					cloud_cover[i] = float(value)
	
	return cloud_cover
	
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
				weather_array[i,0:WEATHER_CLASS_NUM] = numpy.nan
	# 品質情報無し
	else:
		for i in range(data_num):
			index = i + data_start_index
			value = input_data[index][value_index]
			if not value:
				weather_array[i,0:WEATHER_CLASS_NUM] = numpy.nan
			else:
				value = int(value)
				j = WEATHER_CLASSIFY_MAP[value]
				weather_array[i,j] = 1.0
	
	return weather_array

##################################################
# 入力データ(input_data)から天気を抽出して返す
#   - 0〜1の範囲の値に変換して返す
#   - 晴れ,曇り,雨の3分類なら、晴れ:0.0, 曇り:0.5, 雨:1.0
##################################################
def get_variable_weather(input_data, point_name):
	
	# 天気のラベルデータを取得する
	weather = get_weather(input_data, point_name)
	
	# データの入れ物準備と変換係数計算
	weather_v = numpy.zeros( (weather.shape[0]) )
	conv_rate = 1.0 / float(WEATHER_CLASS_NUM - 1)
	
	# 数値に変換した天気データを作成
	for i,label in enumerate(weather):
		if numpy.isnan(label[0]):
			weather_v[i] = numpy.nan
		else:
			index = numpy.argmax(label)
			weather_v[i] = float(index) * conv_rate
	
	return weather_v

