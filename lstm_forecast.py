# -*- coding: utf-8 -*-
"""
  LSTMを用いた天気予測
    - 過去12時間の気象データから、現在の水戸の天気を予測する
"""

import sys, os
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

LENGTH_OF_SEQUENCES = 12	# 過去12時間のデータから予測する
LENGTH_OF_SHIFT = 0		# 6時間後のデータを予測する
#NUMBER_OF_INPUT_NODES = 5   	# 入力データ数:(水戸,前橋,東京,静岡,大阪)の天気
NUMBER_OF_INPUT_NODES = 9  	# 入力データ数:(水戸,前橋,東京)x(降水量,湿度,気圧)
NUMBER_OF_HIDDEN_NODES = 64	# 隠れ層のノード数
NUMBER_OF_OUTPUT_NODES = 3	# 出力データ数(晴れ,曇り,雨)
SIZE_OF_BATCH = 128		# バッチサイズ
DROPOUT_RATE = 0.5		# ドロップアウト率
LEARNING_RATE = 0.001		# 学習率
NUMBER_OF_EPOCHS = 10		# 1回の学習のエポック数
NUMBER_OF_TRAINING = 100	# 学習回数
OUTPUT_CYCLE = 1		# 学習経過出力周期
RESULT_FILE_WHOLE = './result/rnn_result_190812_08_9_whole.csv'
RESULT_FILE_NAME  = './result/rnn_result_190812_08_9'

##################################################
# 学習用データ取得
##################################################
def load_data(dir_path, point_name):
	
	temperature = numpy.array([])
	rainfall = numpy.array([])
	humidity = numpy.array([])
	pressure = numpy.array([])
	weather_label = numpy.array([])
	weather_value = numpy.array([])
	
	csv_paths = get_filepaths(dir_path, '.csv')
	for csv_path in csv_paths:
		csv_data = read_weather_csv(csv_path)
		#temp = get_temperature(csv_data, point_name)
		#temperature = numpy.append(temperature, temp)
		rain = get_rainfall(csv_data, point_name)
		rainfall = numpy.append(rainfall, rain)
		humi = get_humidity(csv_data, point_name)
		humidity = numpy.append(humidity, humi)
		pres = get_sea_level_pressure(csv_data, point_name)
		pressure = numpy.append(pressure, pres)
		#weat_v = get_variable_weather(csv_data, point_name)
		#weather_value = numpy.append(weather_value, weat_v)
		weat_l = get_weather(csv_data, point_name)
		weather_label = numpy.append(weather_label, weat_l)
		
	re_input = numpy.stack([rainfall, humidity, pressure], 1)
	#re_input = numpy.stack([rainfall, humidity, pressure, weather_value], 1)
	#re_input = numpy.stack([weather_value], 1)
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
	#fo.write('入力データ = 水戸,前橋,東京,静岡,大阪の天気\n')
	#fo.write('入力データ = (水戸,前橋,東京,静岡)x(降水量,湿度,気圧,天気)\n')
	fo.write('入力データ = (水戸,前橋,東京)x(降水量,湿度,気圧)\n')
	fo.write('model = %d x LSTM(%d x %d) x DropOut(%d) x %d\n'
		% (NUMBER_OF_INPUT_NODES, NUMBER_OF_INPUT_NODES,
		   NUMBER_OF_HIDDEN_NODES, NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) )
	fo.write('dropout rate = %f\n' % (DROPOUT_RATE) )
	fo.write('optimizer = Adam(lr=%f)\n' % (LEARNING_RATE) )
	fo.write('length of sequence = %d\n' % (LENGTH_OF_SEQUENCES) )
	fo.write('length of shift = %d\n' % (LENGTH_OF_SHIFT) )
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
	fo.write('水戸,,,前橋,,,東京,,,水戸(正解),,,水戸(予測),,,,\n')
	fo.write('降水量,湿度,気圧,降水量,湿度,気圧,降水量,湿度,気圧,晴れ,曇り,雨,晴れ,曇り,雨,正解/不正解\n')
	#fo.write('水戸,,,,前橋,,,,東京,,,,静岡,,,,水戸(正解),,,水戸(予測),,,,\n')
	#fo.write('降水量,湿度,気圧,天気,降水量,湿度,気圧,天気,降水量,湿度,気圧,天気,降水量,湿度,気圧,天気,晴れ,曇り,雨,晴れ,曇り,雨,正解/不正解\n')
	#fo.write('水戸,前橋,東京,静岡,大阪,水戸(正解),,,水戸(予測),,,,\n')
	#fo.write('天気,天気,天気,天気,天気,晴れ,曇り,雨,晴れ,曇り,雨,正解/不正解\n')
	
	# 全テストデータの正解と予想結果出力
	data_len = input.shape[0]
	for i in range(data_len):
		
		# 入力と出力を取得
		input_i = input[i]
		target_i = target[i]
		
		if i < (LENGTH_OF_SEQUENCES+LENGTH_OF_SHIFT):
			# 予想結果が出せないデータの場合(最初の方)
			#fo.write('%f,%f,%f,%f,%f,%.2f,%.2f,%.2f\n' %
			#	(input_i[0], input_i[1], input_i[2], input_i[3], input_i[4],
			#	 target_i[0], target_i[1], target_i[2] ) )
			#fo.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%.2f,%.2f,%.2f\n' %
			#	(input_i[0], input_i[1], input_i[2], input_i[3],
			#	 input_i[4], input_i[5], input_i[6], input_i[7],
			#	 input_i[8], input_i[9], input_i[10], input_i[11],
			#	 input_i[12], input_i[13], input_i[14], input_i[15],
			#	 target_i[0], target_i[1], target_i[2] ) )
			fo.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%.2f,%.2f,%.2f\n' %
				(input_i[0], input_i[1], input_i[2], input_i[3],
				 input_i[4], input_i[5], input_i[6], input_i[7],
				 input_i[8], #input_i[9], input_i[10], input_i[11],
				 target_i[0], target_i[1], target_i[2] ) )
		else:
			pi = i - LENGTH_OF_SEQUENCES - LENGTH_OF_SHIFT
			
			# 正解/不正解を確認する
			correct_i = numpy.argmax(target_i)
			predict_i = numpy.argmax(predicted[pi])
			correct = 1 if (correct_i == predict_i) else 0
			
			#fo.write('%f,%f,%f,%f,%f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n' % 
			#	(input_i[0], input_i[1], input_i[2], input_i[3], input_i[4],
			#	 target_i[0], target_i[1], target_i[2],
			#	 predicted[pi,0], predicted[pi,1], predicted[pi,2], correct ) )
			#fo.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n' % 
			#	(input_i[0], input_i[1], input_i[2], input_i[3],
			#	 input_i[4], input_i[5], input_i[6], input_i[7],
			#	 input_i[8], input_i[9], input_i[10], input_i[11],
			#	 input_i[12], input_i[13], input_i[14], input_i[15],
			#	 target_i[0], target_i[1], target_i[2],
			#	 predicted[pi,0], predicted[pi,1], predicted[pi,2],
			#	 correct) )
			fo.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n' % 
				(input_i[0], input_i[1], input_i[2], input_i[3],
				 input_i[4], input_i[5], input_i[6], input_i[7],
				 input_i[8], #input_i[9], input_i[10], input_i[11],
				 target_i[0], target_i[1], target_i[2],
				 predicted[pi,0], predicted[pi,1], predicted[pi,2],
				 correct) )
				
	fo.close()

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 学習用データ取得
	train_input_raw1, train_target_raw1 = load_data('./train/mito',     '水戸')
	train_input_raw2, train_target_raw2 = load_data('./train/maebashi', '前橋')
	train_input_raw3, train_target_raw3 = load_data('./train/tokyo',    '東京')
	#train_input_raw4, train_target_raw4 = load_data('./train/shizuoka', '静岡')
	#train_input_raw5, train_target_raw5 = load_data('./train/osaka',    '大阪')
	train_input_raw = numpy.hstack(
				[train_input_raw1, train_input_raw2, train_input_raw3] )
				#[train_input_raw1, train_input_raw2, train_input_raw3,
				# train_input_raw4 ] )
				# train_input_raw4, train_input_raw5] )
	#train_input_raw = train_input_raw1
	train_target_raw = train_target_raw1
	
	# テスト用データ取得
	test_input_raw1, test_target_raw1 = load_data('./test/mito',     '水戸')
	test_input_raw2, test_target_raw2 = load_data('./test/maebashi', '前橋')
	test_input_raw3, test_target_raw3 = load_data('./test/tokyo',    '東京')
	#test_input_raw4, test_target_raw4 = load_data('./test/shizuoka', '静岡')
	#test_input_raw5, test_target_raw5 = load_data('./test/osaka',    '大阪')
	test_input_raw = numpy.hstack(
				[test_input_raw1, test_input_raw2, test_input_raw3] )
				#[test_input_raw1, test_input_raw2, test_input_raw3,
				# test_input_raw4 ] )
				# test_input_raw4, test_input_raw5] )
	#test_input_raw = test_input_raw1
	test_target_raw = test_target_raw1
	
	# Max-Minスケール化
	scaler, train_input_scaled, test_input_scaled = \
		 max_min_scale(train_input_raw, test_input_raw)
	#train_target, test_target = max_min_scale(train_target, test_target)
	
	train_input, train_target = make_dataset(train_input_scaled, train_target_raw)
	test_input, test_target = make_dataset(test_input_scaled, test_target_raw)
	
	# モデル作成
	model = Sequential()
	model.add( LSTM(NUMBER_OF_HIDDEN_NODES, 
			input_shape=(LENGTH_OF_SEQUENCES, NUMBER_OF_INPUT_NODES), 
			dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE,
			return_sequences=False) )
	model.add(Dropout(DROPOUT_RATE))
	model.add(Dense(NUMBER_OF_OUTPUT_NODES))
	model.add(Activation('softmax'))
	optimizer = optimizers.Adam(lr=LEARNING_RATE)
	model.compile( loss='categorical_crossentropy', optimizer=optimizer, 
			metrics=['accuracy'])
	model.summary()
	
	output_whole_result_header()
	
	# 学習実行
	for i in range(NUMBER_OF_TRAINING):
		model.fit(train_input, train_target, batch_size=SIZE_OF_BATCH, 
				#epochs=NUMBER_OF_EPOCHS, verbose=0)
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
		if (i % OUTPUT_CYCLE) == 0 :
			output_result(test_input_raw, test_target_raw, predicted, i)
		

