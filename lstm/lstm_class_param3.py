# -*- coding: utf-8 -*-
"""
  LSTMを用いた天気予測(分類)
"""

import sys, os
sys.path.append(os.getcwd())
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

##############################
# 分類用のパラメータ
##############################
LENGTH_OF_SEQUENCE_FOR_CLASS = 24	# 過去24時間のデータから予測する
NUMBER_OF_INPUT_NODES_FOR_CLASS = 4  	# 入力データ数:(水戸)x(気温,降水量,湿度,気圧)
NUMBER_OF_HIDDEN_NODES_FOR_CLASS = 32	# 隠れ層のノード数
NUMBER_OF_OUTPUT_NODES_FOR_CLASS = 3	# 出力データ数(晴れ,曇り,雨)
DROPOUT_RATE_FOR_CLASS = 0.2		# ドロップアウト率
LEARNING_RATE_FOR_CLASS = 0.005		# 学習率
SIZE_OF_BATCH_FOR_CLASS = 128		# バッチサイズ
RESULT_FILE_NAME_FOR_CLASS  = './result/result_190917_lstm_class_03'

##############################
# 共通のパラメータ
##############################
NUMBER_OF_EPOCHS = 10		# 10回の学習のエポック数
NUMBER_OF_TRAINING = 1000	# 学習回数
OUTPUT_CYCLE = 1		# 学習経過出力周期
RESULT_FILE_WHOLE = './result/result_190917_lstm_class_03.csv'

##################################################
# 学習用データ取得(分類用)
##################################################
def load_data_for_class(dir_path, point_name):
	
	temperature = numpy.array([])
	rainfall = numpy.array([])
	humidity = numpy.array([])
	pressure = numpy.array([])
	weather_label = numpy.array([])
	weather_value = numpy.array([])
	
	csv_paths = get_filepaths(dir_path, '.csv')
	for csv_path in csv_paths:
		csv_data = read_weather_csv(csv_path)
		temp = get_temperature(csv_data, point_name)
		temperature = numpy.append(temperature, temp)
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
		
	re_input = numpy.stack([temperature, rainfall, humidity, pressure], 1)
	re_target = numpy.reshape(weather_label,
			(weather_label.shape[0]/WEATHER_CLASS_NUM, WEATHER_CLASS_NUM))
	
	re_input_interpolated  = interpolate_nan_input_data(re_input)
	re_target_interpolated = interpolate_nan_label_data(re_target)
	
	return re_input_interpolated, re_target_interpolated
	
##################################################
# データセット作成(分類用)
##################################################
def make_dataset_for_class(input, target):
	
	data_num = input.shape[0]
	input_num = input.shape[1]
	target_num = target.shape[1]
	
	data_len = data_num - LENGTH_OF_SEQUENCE_FOR_CLASS
	re_input  = numpy.zeros( (data_len, LENGTH_OF_SEQUENCE_FOR_CLASS, input_num) )
	re_target = numpy.zeros( (data_len, target_num) )
	
	for i in range(data_len):
		
		re_input[i,0:LENGTH_OF_SEQUENCE_FOR_CLASS,] = input[i:i+LENGTH_OF_SEQUENCE_FOR_CLASS,]
		re_target[i,] = target[i+LENGTH_OF_SEQUENCE_FOR_CLASS-1,]
		
	return re_input, re_target
	
##################################################
# データ取得(分類用)
##################################################
def get_data_for_class():
	
	# 学習用データ取得
	train_input_raw1, train_target_raw1 = load_data_for_class('./train/mito',     '水戸')
	#train_input_raw2, train_target_raw2 = load_data_for_class('./train/maebashi', '前橋')
	#train_input_raw3, train_target_raw3 = load_data_for_class('./train/tokyo',    '東京')
	#train_input_raw4, train_target_raw4 = load_data_for_class('./train/shizuoka', '静岡')
	#train_input_raw5, train_target_raw5 = load_data('./train/osaka',    '大阪')
	#train_input_raw = numpy.hstack(
	#			[train_input_raw1, train_input_raw3, train_input_raw4] )
	train_input_raw = train_input_raw1.reshape(train_input_raw1.shape[0], NUMBER_OF_INPUT_NODES_FOR_CLASS)
	train_target_raw = train_target_raw1
	
	# テスト用データ取得
	test_input_raw1, test_target_raw1 = load_data_for_class('./test/mito',     '水戸')
	#test_input_raw2, test_target_raw2 = load_data_for_class('./test/maebashi', '前橋')
	#test_input_raw3, test_target_raw3 = load_data_for_class('./test/tokyo',    '東京')
	#test_input_raw4, test_target_raw4 = load_data_for_class('./test/shizuoka', '静岡')
	#test_input_raw5, test_target_raw5 = load_data('./test/osaka',    '大阪')
	#test_input_raw = numpy.hstack(
	#			[test_input_raw1, test_input_raw3, test_input_raw4] )
	test_input_raw = test_input_raw1.reshape(test_input_raw1.shape[0], NUMBER_OF_INPUT_NODES_FOR_CLASS)
	test_target_raw = test_target_raw1
	
	# Max-Minスケール化
	scaler, train_input_scaled, test_input_scaled = \
		 max_min_scale(train_input_raw, test_input_raw)
	#train_target_scaled, test_target_scaled = max_min_scale(train_target, test_target)
	
	train_input, train_target = make_dataset_for_class(train_input_scaled, train_target_raw)
	test_input, test_target = make_dataset_for_class(test_input_scaled, test_target_raw)
	
	return train_input, train_target, test_input, test_target, scaler
	
##################################################
# 学習用のモデルを作成する(分類用)
##################################################
def make_model_for_class():
	
	# モデル作成
	model = Sequential()
	model.add( LSTM(NUMBER_OF_HIDDEN_NODES_FOR_CLASS, 
			input_shape=(LENGTH_OF_SEQUENCE_FOR_CLASS, NUMBER_OF_INPUT_NODES_FOR_CLASS), 
			dropout=DROPOUT_RATE_FOR_CLASS, recurrent_dropout=DROPOUT_RATE_FOR_CLASS,
			return_sequences=True) )
	model.add( LSTM(NUMBER_OF_HIDDEN_NODES_FOR_CLASS, 
			input_shape=(LENGTH_OF_SEQUENCE_FOR_CLASS, NUMBER_OF_INPUT_NODES_FOR_CLASS), 
			dropout=DROPOUT_RATE_FOR_CLASS, recurrent_dropout=DROPOUT_RATE_FOR_CLASS,
			return_sequences=False) )
	model.add(Dropout(DROPOUT_RATE_FOR_CLASS))
	model.add(Dense(NUMBER_OF_OUTPUT_NODES_FOR_CLASS))
	model.add(Activation('softmax'))
	optimizer = optimizers.Adam(lr=LEARNING_RATE_FOR_CLASS)
	model.compile( loss='categorical_crossentropy', optimizer=optimizer, 
			metrics=['accuracy'])
	model.summary()
	
	return model
	
##################################################
# 学習経過ファイルのヘッダーを出力する。
##################################################
def output_whole_result_header():
	
	fo = open(RESULT_FILE_WHOLE, 'a')
	fo.write('##################################################\n')
	fo.write('入力データ = (水戸)x(気温 降水量 湿度 気圧)\n')
	fo.write('model_for_class = %d x LSTM(%d x %d) x DropOut(%d) x %d\n'
		% (NUMBER_OF_INPUT_NODES_FOR_CLASS, NUMBER_OF_INPUT_NODES_FOR_CLASS,
		   NUMBER_OF_HIDDEN_NODES_FOR_CLASS, NUMBER_OF_HIDDEN_NODES_FOR_CLASS, 
		   NUMBER_OF_OUTPUT_NODES_FOR_CLASS) )
	fo.write('dropout rate = %f\n' % (DROPOUT_RATE_FOR_CLASS) )
	fo.write('optimizer = Adam(lr=%f)\n' % (LEARNING_RATE_FOR_CLASS) )
	fo.write('length of sequence = %d\n' % (LENGTH_OF_SEQUENCE_FOR_CLASS) )
	fo.write('date: ' + get_datetime_string() + '\n')
	fo.write('##################################################\n')
	fo.write('epoch,loss acc\n')
	fo.close()
	
##################################################
# 学習結果をファイル出力する(分類用)
##################################################
def output_result_for_class(input, target, predicted, number):
	
	# 正解と予想結果をファイル出力
	filename = str.format('%s_%03d.csv' % (RESULT_FILE_NAME_FOR_CLASS, number) )
	fo = open(filename, 'w')
	fo.write('水戸,,,,水戸(正解),,,水戸(予測),,,,\n')
	fo.write('気温,降水量,湿度,気圧,晴れ,曇り,雨,晴れ,曇り,雨,正解/不正解\n')
	
	# 全テストデータの正解と予想結果出力
	data_len = input.shape[0]
	sequence_len = LENGTH_OF_SEQUENCE_FOR_CLASS - 1
	for i in range(data_len):
		
		# 入力と出力を取得
		input_i = input[i]
		
		if i < sequence_len:
			# 予想結果が出せないデータの場合(最初の方)
			fo.write('%.1f,%.1f,%.1f,%.1f\n' %
				(input_i[0], input_i[1], input_i[2], input_i[3],
				 ) )
		else:
			pi = i - sequence_len
			
			# 正解/不正解を確認する
			correct_i = numpy.argmax(target[pi])
			predict_i = numpy.argmax(predicted[pi])
			correct = 1 if (correct_i == predict_i) else 0
			fo.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n' %
				(input_i[0], input_i[1], input_i[2], input_i[3],
				 target[pi,0], target[pi,1], target[pi,2],
				 predicted[pi,0], predicted[pi,1], predicted[pi,2],
				 correct
				) )
				
	fo.close()

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 分類用のデータ取得
	train_input_for_class, train_target_for_class, \
	test_input_for_class, test_target_for_class, \
	scaler_for_class = get_data_for_class()
	
	# 分類用のモデル作成
	model_for_class = make_model_for_class()
	
	# 結果ファイルのヘッダー出力
	output_whole_result_header()
	
	# 学習実行
	for i in range(NUMBER_OF_TRAINING):
		
		# 学習
		model_for_class.fit(
				train_input_for_class, train_target_for_class, 
				batch_size=SIZE_OF_BATCH_FOR_CLASS, 
				epochs=NUMBER_OF_EPOCHS, validation_split=0)
		
		# 結果出力
		loss_class, acc = model_for_class.evaluate(test_input_for_class, test_target_for_class, verbose=0)
		print('%07d : loss_c=%f, acc=%f' % ((i+1)*NUMBER_OF_EPOCHS, loss_class, acc))
		fo = open(RESULT_FILE_WHOLE, 'a')
		fo.write('%07d,%f,%f\n' % ((i+1)*NUMBER_OF_EPOCHS, loss_class, acc))
		fo.close()
		
		# 正解と予想結果をファイル出力
		if (i % OUTPUT_CYCLE) == 0 :
			
			# 分類の結果出力
			predicted_for_class = model_for_class.predict(test_input_for_class)
			input_c_inversed = scaler_for_class.inverse_transform(test_input_for_class[:,0,:])
			output_result_for_class(input_c_inversed, test_target_for_class, predicted_for_class, i)
		

