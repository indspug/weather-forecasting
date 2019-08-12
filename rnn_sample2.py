# -*- coding: utf-8 -*-
'''
  RNNの動作確認用2
    24時間分の湿度,気圧,天気から、3時間後の天気を予測する
'''

import sys, os
import datetime
from common.file import *
from common.utility import *
from common.processing import *
from format import *
import numpy
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras import optimizers

LENGTH_OF_SEQUENCES = 24	# 過去24時間のデータから予測する
LENGTH_OF_SHIFT = 6 		# 3時間後のデータを予測する
NUMBER_OF_INPUT_NODES = 6	# 入力データ数(気温,相対湿度,海面気圧)
NUMBER_OF_HIDDEN_NODES = 64	# 隠れ層のノード数
NUMBER_OF_OUTPUT_NODES = 3	# 出力データ数(晴れ,曇り,雨
SIZE_OF_BATCH = 128		# バッチサイズ
DROPOUT_RATE = 0.5		# ドロップアウト率
LEARNING_RATE = 0.001		# 学習率
NUMBER_OF_EPOCHS = 10		# 1回の学習のエポック数
NUMBER_OF_TRAINING = 50		# 学習回数
RESULT_FILE_WHOLE = './result/rnn_result_190812_02_whole.csv'
RESULT_FILE_NAME  = './result/rnn_result_190812_02'

##################################################
# 学習用データ取得
##################################################
def load_data(dir_path, point_name):
	
	temperature = numpy.array([])
	humidity = numpy.array([])
	pressure = numpy.array([])
	weather_label = numpy.array([])
	weather_value = numpy.array([])
	
	csv_paths = get_filepaths(dir_path, '.csv')
	for csv_path in csv_paths:
		csv_data = read_weather_csv(csv_path)
		#temp = get_temperature(csv_data, point_name)
		#temperature = numpy.append(temperature, temp)
		humi = get_humidity(csv_data, point_name)
		humidity = numpy.append(humidity, humi)
		pres = get_sea_level_pressure(csv_data, point_name)
		pressure = numpy.append(pressure, pres)
		weat_v = get_variable_weather(csv_data, point_name)
		weather_value = numpy.append(weather_value, weat_v)
		weat_l = get_weather(csv_data, point_name)
		weather_label = numpy.append(weather_label, weat_l)
		
	#re_input = numpy.stack([temperature, humidity, pressure, weather_value], 1)
	re_input = numpy.stack([humidity, pressure, weather_value], 1)
	re_target = numpy.reshape(weather_label,
			(weather_label.shape[0]/WEATHER_CLASS_NUM, WEATHER_CLASS_NUM))
	
	re_input_interpolated  = interpolate_nan_input_data(re_input)
	re_target_interpolated = interpolate_nan_label_data(re_target)
	
	return re_input_interpolated, re_target_interpolated
	
##################################################
# データセット作成
##################################################
def make_dataset(input, target):
	
	data_num = input.shape[0]
	input_num = input.shape[1]
	target_num = target.shape[1]
	
	data_len = data_num - LENGTH_OF_SEQUENCES - LENGTH_OF_SHIFT
	re_input  = numpy.zeros( (data_len, LENGTH_OF_SEQUENCES, input_num) )
	re_target = numpy.zeros( (data_len, target_num) )
	
	for i in range(data_len):
		
		re_input[i,0:LENGTH_OF_SEQUENCES,] = input[i:i+LENGTH_OF_SEQUENCES,]
		re_target[i,] = target[i+LENGTH_OF_SEQUENCES+LENGTH_OF_SHIFT,]
		
	return re_input, re_target
	
##################################################
# 学習経過ファイルのヘッダーを出力する。
##################################################
def output_whole_result_header():
	
	fo = open(RESULT_FILE_WHOLE, 'a')
	fo.write('##################################################\n')
	fo.write('入力データ = 相対湿度 海面気圧 天気\n')
	fo.write('model = %d x LSTM(%dx%d)x DropOut(%d) x %d\n'
		% (NUMBER_OF_INPUT_NODES, NUMBER_OF_INPUT_NODES,
		   NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES, NUMBER_OF_OUTPUT_NODES) )
	fo.write('dropout rate = %f\n' % (DROPOUT_RATE) )
	fo.write('optimizer = RMSprop(lr=%f)\n' % (LEARNING_RATE) )
	fo.write('date: ' + get_datetime_string() + '\n')
	fo.write('##################################################\n')
	fo.write('epoch,loss acc\n')
	fo.close()
	
##################################################
# 学習結果をファイル出力する。
##################################################
def output_result(input, target, predicted, number):
	
	# 正解と予想結果をファイル出力
	filename = str.format('%s_%03d.csv' % (RESULT_FILE_NAME, number) )
	fo = open(filename, 'w')
	fo.write('湿度(水戸),気圧(水戸),天気(水戸),湿度(大阪),気圧(大阪),天気(大阪),晴れ,曇り,雨,晴れ(予測),曇り(予測),雨(予測),正解/不正解\n')
	
	# 全テストデータの正解と予想結果出力
	data_len = input.shape[0]
	for i in range(data_len):
		
		# 入力と出力を取得
		input_i = input[i]
		target_i = target[i]
		
		if i < (LENGTH_OF_SEQUENCES+LENGTH_OF_SHIFT):
			# 予想結果が出せないデータの場合(最初の方)
			fo.write('%f,%f,%f,%f,%f,%f,%.2f,%.2f,%.2f\n' %
				(input_i[0], input_i[1], input_i[2], 
				 input_i[3], input_i[4], input_i[5],
				 target_i[0], target_i[1], target_i[2] ) )
		else:
			pi = i - LENGTH_OF_SEQUENCES - LENGTH_OF_SHIFT
			
			# 正解/不正解を確認する
			correct_i = numpy.argmax(target)
			predict_i = numpy.argmax(predicted[pi])
			correct = 1 if (correct_i == predict_i) else 0
			
			fo.write('%f,%f,%f,%f,%f,%f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n' % 
				(input_i[0], input_i[1], input_i[2],
				 input_i[3], input_i[4], input_i[5],
				 target_i[0], target_i[1], target_i[2],
				 predicted[pi,0], predicted[pi,1], predicted[pi,2],
				 correct) )
				
	fo.close()

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 学習用データ取得
	train_input_raw1, train_target_raw1 = load_data('./train/mito',  '水戸')
	train_input_raw2, train_target_raw2 = load_data('./train/osaka', '大阪')
	train_input_raw = numpy.hstack( [train_input_raw1, train_input_raw2] )
	train_target_raw = train_target_raw1
	
	# テスト用データ取得
	test_input_raw1, test_target_raw1 = load_data('./test/mito',  '水戸')
	test_input_raw2, test_target_raw2 = load_data('./test/osaka', '大阪')
	test_input_raw = numpy.hstack( [test_input_raw1, test_input_raw2] )
	test_target_raw = test_target_raw1
	
	print(train_input_raw1.shape)
	print(train_input_raw2.shape)
	print(train_input_raw.shape)
	print(test_input_raw1.shape)
	print(test_input_raw2.shape)
	print(test_input_raw.shape)
	
	# Max-Minスケール化
	scaler, train_input_scaled, test_input_scaled = \
		 max_min_scale(train_input_raw, test_input_raw)
	#train_target, test_target = max_min_scale(train_target, test_target)
	
	train_input, train_target = make_dataset(train_input_scaled, train_target_raw)
	test_input, test_target = make_dataset(test_input_scaled, test_target_raw)
	
	# モデル作成
	model = Sequential()
	model.add( LSTM( NUMBER_OF_HIDDEN_NODES, 
			input_shape=(LENGTH_OF_SEQUENCES, NUMBER_OF_INPUT_NODES), 
			return_sequences=False) )
	model.add(Dense(NUMBER_OF_OUTPUT_NODES))
	model.add(Dropout(DROPOUT_RATE))
	model.add(Activation('softmax'))
	optimizer = optimizers.RMSprop(lr=LEARNING_RATE)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	model.summary()
	
	output_whole_result_header()
	
	# 学習実行
	for i in range(NUMBER_OF_TRAINING):
		model.fit(train_input, train_target, batch_size=SIZE_OF_BATCH, 
				epochs=NUMBER_OF_EPOCHS, validation_split=0.05)
		
		# 結果出力
		loss, acc = model.evaluate(test_input, test_target, verbose=0)
		print('%07d : loss=%f, acc=%f' % ((i+1)*NUMBER_OF_EPOCHS, loss, acc))
		fo = open(RESULT_FILE_WHOLE, 'a')
		fo.write('%07d,%f,%f\n' % ((i+1)*NUMBER_OF_EPOCHS, loss, acc))
		fo.close()
		
		# 予想結果取得
		predicted = model.predict(test_input)
		
		# 正解と予想結果をファイル出力
		output_result(test_input_raw, test_target_raw, predicted, i)
		

