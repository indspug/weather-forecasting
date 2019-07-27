# -*- coding: utf-8 -*-
  
import sys, os
from common.format import *
from common.processing import *
import csv
import numpy
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

POINT_NAME='水戸'

##################################################
# CSVファイル(csv_path)から気象データを読み込む
# [Args]
#   csv_path : CSVファイルのファイルパス　
# [Rerutn:
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

##################################################
# CSVファイルから取り出した気象データをから
# 学習に必要なデータを抽出して返す
# [Args]
#   csv_data : CSVファイルから取り出した気象データ(stringのlist)
# [Return]
#   input_data : 入力データ(ndarray)
#   label_data : 出力データ(ndarray)
##################################################
def extract_learning_data(csv_data):
	
	# 入力データ取得：降水量,相対湿度,海面気圧
	rainfall = get_rainfall(csv_data, POINT_NAME)
	humidity = get_humidity(csv_data, POINT_NAME)
	sea_level_pressure = get_sea_level_pressure(csv_data, POINT_NAME)
	#temperature = get_temperature(csv_data, POINT_NAME)
	#wind_speed = get_wind_speed(csv_data, POINT_NAME)
	#wind_dir = get_wind_direction(csv_data, POINT_NAME)
	#daylight = get_daylight(csv_data, POINT_NAME)
	#atom_pressure = get_atom_pressure(csv_data, POINT_NAME)
	
	# 入力データ結合
	input_data = numpy.stack(
		[rainfall, humidity, sea_level_pressure], 1)
	
	# 出力データ取得：天気
	label_data = get_weather(csv_data, POINT_NAME)
	
	# 入力データ・出力データにNaNが含まれている行を削る
	isnan_row_input = numpy.isnan(input_data).any(axis=1)
	isnan_row_label = numpy.isnan(label_data).any(axis=1)
	isnan_row = numpy.logical_or(isnan_row_input, isnan_row_label)
	input_data = input_data[~isnan_row,]
	label_data = label_data[~isnan_row,]
	
	# 不整合データを削除する
	is_inconsistency = check_rainfall_inconsistency(input_data[:,0], label_data)
	#print(is_inconsistency)
	input_data = input_data[~is_inconsistency,]
	label_data = label_data[~is_inconsistency,]
	
	# 入力データをMAX-MIN標準化する
	input_data = max_min_normalize(input_data, axis=0)
	
	return (input_data, label_data)
	
##################################################
# メイン
# [Args]
#   train_csv_path : 訓練用データのCSVファイルのパス
#   test_csv_path  : テスト用データのCSVファイルのパス
##################################################
if __name__ == '__main__':
	
	# コマンドライン引数のチェック
	argvs = sys.argv
	argc = len(argvs)
	if argc < 2:
		exe_name=argvs[0]
		print('Usage: python3 %s [train_csv_path] [test_csv_path]' % exe_name)
		quit()
	
	# CSVファイルパス取り出し
	train_csv_path = argvs[1]
	test_csv_path  = argvs[2]
	
	# CSVファイルから気象データ取得
	train_csv_data = read_weather_csv(train_csv_path)
	test_csv_data  = read_weather_csv(test_csv_path)
	
	# 学習用の入力データと出力データを抽出
	train_input, train_label = extract_learning_data(train_csv_data)
	test_input, test_label = extract_learning_data(test_csv_data)
	
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
	for i in range(100):
                # 学習
		model.fit(
			train_input, train_label, 
			epochs=100, batch_size=16, shuffle=False, verbose=0)
		
		# 評価
		#  loss: 損失値(正解データとの誤差。小さい方が良い。)
		#  acc: 評価値(正解率。高い方が良い。)
		score = model.evaluate(test_input, test_label, verbose=0)
		print('%07d : loss=%f, acc=%f' % (epoch, score[0], score[1]))
		
		epoch = epoch + 100
	
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
	
