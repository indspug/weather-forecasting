# -*- coding: utf-8 -*-
  
import sys, os
from common.format import *
from common.processing import *
import csv
import numpy
import pandas
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

##################################################
# CSVファイル(csv_path)から気象データを読み込む
# [Args]
#   csv_path : CSVファイルのファイルパス　
# [Rerutn:
#   csv_data : CSVから取り出したデータ(stringのlist)
##################################################
def read_weather_csv(csv_path):
	csv_data = []
	with open(csv_path, 'r', encoding='shift_jis') as f:
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
	
	# 入力データ取得：気温,降水量,風速,風向,相対湿度,日照時間,現地気圧
	temperature = get_temperature(csv_data)
	rainfall = get_rainfall(csv_data)
	wind_speed, wind_dir = get_wind_speed(csv_data)
	humidity = get_humidity(csv_data)
	daylight = get_daylight(csv_data)
	atom_pressure = get_atom_pressure(csv_data)
	
	# 入力データ結合
	input_data = numpy.stack(
		[rainfall, daylight], 1)
		#[rainfall, humidity, daylight, atom_pressure], 1)
		#[temperature, rainfall, wind_speed, wind_dir, 
		# humidity, daylight, atom_pressure], 1)
	
	# 出力データ取得：天気
	label_data = get_weather(csv_data)
	
	# 入力データ・出力データにNaNが含まれている行を削る
	isnan_row_input = numpy.isnan(input_data).any(axis=1)
	isnan_row_label = numpy.isnan(label_data).any(axis=1)
	isnan_row = numpy.logical_or(isnan_row_input, isnan_row_label)
	input_data = input_data[~isnan_row,]
	label_data = label_data[~isnan_row,]
	
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
	
	#print(train_input.shape)
	#print(test_input.shape)
	
	# モデルの作成
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
	for i in range(10):
		model.fit(
			train_input, train_label, 
			epochs=100, batch_size=32, shuffle=True, verbose=0)
		score = model.evaluate(test_input, test_label, verbose=0)
		
		epoch = epoch + 100
		print('%07d : loss=%f, acc=%f' % (epoch, score[0], score[1]))
		#print(model.metrics_names['accuracy'])
	
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
	
