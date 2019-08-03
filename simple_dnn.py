# -*- coding: utf-8 -*-
  
import sys, os
from common.file import *
from format.extract import *
from format.validate import *
from common.processing import *
import csv
import numpy
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

POINT_NAME='水戸'
RAINFALL_INDEX=0

##################################################
# 指定したディレクトリしたCSVファイル群から気象データを読み込み、
# 学習に必要なデータを抽出して返す
# [Args]
#   dirpath  : CSVファイルが格納されたディレクトリのパス
# [Return]
#   input_data : 入力データ(ndarray)
#   label_data : 出力データ(ndarray)
##################################################
def extract_learning_data(dirpath):
	
	print 'ディレクトリ: ' + dirpath
	
	input_data_list = []
	label_data_list = []
	
	csv_paths = get_filepaths(dirpath, '.csv')
	for csv_path in csv_paths:
		
		# CSVファイル読み込み
		csv_data = read_weather_csv(csv_path)
	
		# 入力データ取得：降水量,相対湿度,海面気圧
		rainfall = get_rainfall(csv_data, POINT_NAME)
		humidity = get_humidity(csv_data, POINT_NAME)
		sea_level_pressure = get_sea_level_pressure(csv_data, POINT_NAME)
		#temperature = get_temperature(csv_data, POINT_NAME)
		#wind_speed = get_wind_speed(csv_data, POINT_NAME)
		#wind_dir = get_wind_direction(csv_data, POINT_NAME)
		#daylight = get_daylight(csv_data, POINT_NAME)
		#atom_pressure = get_atom_pressure(csv_data, POINT_NAME)
		
		# 時(hh)取得
		#datetime = get_datetime(csv_data)
		#datetime = extract_element_from_datetime(datetime)
		#hour = datetime[:,3]
		#hour = hour - 12.0  # 正午が0になるように調整
	
		# 入力データ結合
		input_data = numpy.stack(
			[rainfall, humidity, sea_level_pressure], 1)
		
		# 出力データ取得：天気
		label_data = get_weather(csv_data, POINT_NAME)
		
		input_data_list.append(input_data)
		label_data_list.append(label_data)
		
	# 複数のCSVから読み込んだデータを結合する
	input_data = input_data_list[0]
	for i in range(2, len(input_data_list)):
		input_data = numpy.vstack( [input_data, input_data_list[i]] )
	
	label_data = label_data_list[0]
	for i in range(2, len(label_data_list)):
		label_data = numpy.vstack([label_data, label_data_list[i]])
	
	print '  読込データ数: %d' % input_data.shape[0]
	
	# 入力データ・出力データにNaNが含まれている行を削る
	input_data, label_data = remove_nan_data(input_data, label_data)
	
	print '  NaNデータ削除後: %d' % input_data.shape[0]
	
	# 不整合データを削除する
	is_inconsistency_row = validate_rainfall_inconsistency(
		input_data[:,RAINFALL_INDEX], label_data)
	input_data = input_data[~is_inconsistency_row,]
	label_data = label_data[~is_inconsistency_row,]
	
	print '  不整合データ削除後: %d' % input_data.shape[0]
	
	return (input_data, label_data)
	
##################################################
# メイン
# [Args]
#   train_csv_dirpath : 訓練用CSVファイルを格納したディレクトリのパス
#   test_csv_dirpath  : テスト用CSVファイルを格納したディレクトリのパス
##################################################
if __name__ == '__main__':
	
	# コマンドライン引数のチェック
	argvs = sys.argv
	argc = len(argvs)
	if argc < 2:
		exe_name=argvs[0]
		print('Usage: python %s [train_csv_dirpath] [test_csv_dirpath]' % exe_name)
		quit()
	
	# CSVファイルパス取り出し
	train_csv_dirpath = argvs[1]
	test_csv_dirpath  = argvs[2]
	
	# CSVファイルから気象データを取得し、
	# 学習用の入力データと出力データを抽出
	train_input, train_label = extract_learning_data(train_csv_dirpath)
	test_input, test_label = extract_learning_data(test_csv_dirpath)
	
	# 入力データをMAX-MIN標準化する
	train_input, test_input = max_min_normalize([train_input, test_input], axis=0)
	
	# モデルの作成
	#  3 x 32 x 32 x 3 
	input_data_dim = train_input.shape[1]
	label_num = train_label.shape[1]
	model = Sequential()
	model.add(Dense(32, input_dim=input_data_dim))
	model.add(Activation('relu'))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dense(label_num))
	model.add(Activation('softmax'))
	#model.summary()
	
	#optimizer = optimizers.SGD(lr=0.01)
	#optimizer = optimizers.Adam(lr=0.001)
	optimizer = optimizers.RMSprop(lr=0.001)
	model.compile(
		optimizer=optimizer,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	
	# 学習実行
	epoch = 0
	for i in range(1):
                # 学習
		model.fit(
			train_input, train_label, 
			epochs=100, batch_size=16, shuffle=True, verbose=0)
		
		# 評価
		#  loss: 損失値(正解データとの誤差。小さい方が良い。)
		#  acc : 評価値(正解率。高い方が良い。)
		score = model.evaluate(test_input, test_label, verbose=0)
		print('%07d : loss=%f, acc=%f' % (epoch, score[0], score[1]))
		
		epoch = epoch + 100
	
	# 学習モデルの保存
	json_string = model.to_json()
	open(os.path.join('model','model.json'), 'w').write(json_string)
	yaml_string = model.to_json()
	open(os.path.join('model','model.yaml'), 'w').write(yaml_string)
	
	# 20データだけ値表示
	instant_num = 20
	instant_input = test_input[0:instant_num,]
	actual_label = test_label[0:instant_num,]
	for i in range(instant_num):
		predict_label = model.predict(instant_input)
		print('####################################')
		print(instant_input[i])
		print(predict_label[i])
		print(actual_label[i])
	
