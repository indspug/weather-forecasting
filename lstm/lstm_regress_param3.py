# -*- coding: utf-8 -*-
"""
  LSTMを用いた天気予測(回帰)
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
# 回帰用のパラメータ
##############################
LENGTH_OF_SEQUENCE_FOR_REGRESS = 24	# 過去48時間のデータから予測する
LENGTH_OF_SHIFT_FOR_REGRESS = 0		# 0時間後のデータを予測する
NUMBER_OF_INPUT_NODES_FOR_REGRESS = 2 	# 入力データ数:(水戸)x(気温,降水量,湿度,気圧)
NUMBER_OF_HIDDEN_NODES_FOR_REGRESS = 32	# 隠れ層のノード数
NUMBER_OF_OUTPUT_NODES_FOR_REGRESS = 2	# 出力データ数:(水戸)x(気温,降水量,湿度,気圧)
DROPOUT_RATE_FOR_REGRESS = 0.1		# ドロップアウト率
LEARNING_RATE_FOR_REGRESS = 0.01	# 学習率
SIZE_OF_BATCH_FOR_REGRESS = 24		# バッチサイズ
RESULT_FILE_NAME_FOR_REGRESS  = './result/result_190815_lstm_regress_03'

##############################
# 共通のパラメータ
##############################
NUMBER_OF_EPOCHS = 1		# 10回の学習のエポック数
NUMBER_OF_TRAINING = 1000	# 学習回数
OUTPUT_CYCLE = 1		# 学習経過出力周期
RESULT_FILE_WHOLE = './result/result_190815_lstm_regress_03.csv'

##################################################
# 学習用データ取得(回帰用)
##################################################
def load_data_for_regress(dir_path, point_name):
	
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
		#rain = get_rainfall(csv_data, point_name)
		#rainfall = numpy.append(rainfall, rain)
		humi = get_humidity(csv_data, point_name)
		humidity = numpy.append(humidity, humi)
		#pres = get_sea_level_pressure(csv_data, point_name)
		#pressure = numpy.append(pressure, pres)
		#weat_v = get_variable_weather(csv_data, point_name)
		#weather_value = numpy.append(weather_value, weat_v)
		#weat_l = get_weather(csv_data, point_name)
		#weather_label = numpy.append(weather_label, weat_l)
		
	re_input = numpy.stack([temperature, humidity], 1)
	re_target = numpy.stack([temperature, humidity], 1)
	
	re_input_interpolated  = interpolate_nan_input_data(re_input)
	re_target_interpolated = interpolate_nan_label_data(re_target)
	
	return re_input_interpolated, re_target_interpolated
	
##################################################
# データセット作成(回帰用)
##################################################
def make_dataset_for_regress(input, target):
	
	data_num = input.shape[0]
	input_num = input.shape[1]
	target_num = target.shape[1]
	
	data_len = data_num - LENGTH_OF_SEQUENCE_FOR_REGRESS - LENGTH_OF_SHIFT_FOR_REGRESS
	re_input  = numpy.zeros( (data_len, LENGTH_OF_SEQUENCE_FOR_REGRESS, input_num) )
	re_target = numpy.zeros( (data_len, target_num) )
	
	for i in range(data_len):
		
		re_input[i,0:LENGTH_OF_SEQUENCE_FOR_REGRESS,] = input[i:i+LENGTH_OF_SEQUENCE_FOR_REGRESS,]
		re_target[i,] = target[i+LENGTH_OF_SEQUENCE_FOR_REGRESS+LENGTH_OF_SHIFT_FOR_REGRESS,]
		#print(re_input[i,0:LENGTH_OF_SEQUENCE_FOR_REGRESS])
		
	return re_input, re_target
	
##################################################
# データ取得(回帰用)
##################################################
def get_data_for_regress():
	
	# 学習用データ取得
	train_input_raw1, train_target_raw1 = load_data_for_regress('./train/mito',     '水戸')
	#train_input_raw2, train_target_raw2 = load_data_for_regress('./train/maebashi', '前橋')
	#train_input_raw3, train_target_raw3 = load_data_for_regress('./train/tokyo',    '東京')
	#train_input_raw4, train_target_raw4 = load_data_for_regress('./train/shizuoka', '静岡')
	#train_input_raw5, train_target_raw5 = load_data('./train/osaka',    '大阪')
	#train_input_raw = numpy.hstack(
	#			[train_input_raw1, train_input_raw3, train_input_raw4] )
	#train_target_raw = numpy.hstack(
	#			[train_target_raw1, train_target_raw3, train_target_raw4] )
	train_input_raw = train_input_raw1.reshape(train_input_raw1.shape[0], 2)
	train_target_raw = train_target_raw1
	
	# テスト用データ取得
	test_input_raw1, test_target_raw1 = load_data_for_regress('./test/mito',     '水戸')
	#test_input_raw2, test_target_raw2 = load_data_for_regress('./test/maebashi', '前橋')
	#test_input_raw3, test_target_raw3 = load_data_for_regress('./test/tokyo',    '東京')
	#test_input_raw4, test_target_raw4 = load_data_for_regress('./test/shizuoka', '静岡')
	#test_input_raw5, test_target_raw5 = load_data('./test/osaka',    '大阪')
	#test_input_raw = numpy.hstack(
	#			[test_input_raw1, test_input_raw3, test_input_raw4] )
	#test_target_raw = numpy.hstack(
	#			[test_target_raw1, test_target_raw3, test_target_raw4] )
	test_input_raw = test_input_raw1.reshape(test_input_raw1.shape[0], 2)
	test_target_raw = test_target_raw1
	
	# データの差分化
	train_input_diff = train_input_raw[1:,] - train_input_raw[0:-1,]
	train_target_diff = train_target_raw[1:,] - train_target_raw[0:-1,]
	test_input_diff = test_input_raw[1:,] - test_input_raw[0:-1,]
	test_target_diff = test_target_raw[1:,] - test_target_raw[0:-1,]
	
	# Max-Minスケール化
	scaler_input, train_input_scaled, test_input_scaled = \
		 max_min_scale(train_input_diff, test_input_diff)
	scaler_target, train_target_scaled, test_target_scaled = \
		 max_min_scale(train_target_diff, test_target_diff)
	#train_target_scaled = scaler.transform(train_target_raw)
	#train_target_scaled = scaler.transform(train_target_raw)
	#test_target_scaled = scaler.transform( test_target_raw)
	
	train_input, train_target = make_dataset_for_regress(train_input_scaled, train_target_scaled)
	test_input, test_target = make_dataset_for_regress(test_input_scaled, test_target_scaled)
	
	return	train_input_raw, train_target_raw, test_input_raw, test_target_raw, \
		train_input, train_target, test_input, test_target, scaler_input, scaler_target
	
	#train_input, train_target = make_dataset_for_regress(train_input_raw, train_target_raw)
	#test_input, test_target = make_dataset_for_regress(test_input_raw, test_target_raw)
	#
	#scaler_input, train_input_scaled, test_input_scaled = \
	#	 max_min_scale(train_input, test_input)
	#scaler_target, train_target_scaled, test_target_scaled = \
	#	 max_min_scale(train_target, test_target)
	#
	#return train_input_scaled, train_target_scaled, test_input_scaled, test_target_scaled, \
	#	scaler_input, scaler_target
	
##################################################
# 学習用のモデルを作成する(回帰用)
##################################################
def make_model_for_regress():
	
	# モデル作成
	model = Sequential()
	model.add( LSTM(NUMBER_OF_HIDDEN_NODES_FOR_REGRESS, 
			#input_shape=(NUMBER_OF_INPUT_NODES_FOR_REGRESS, LENGTH_OF_SEQUENCE_FOR_REGRESS), 
			input_shape=(LENGTH_OF_SEQUENCE_FOR_REGRESS, NUMBER_OF_INPUT_NODES_FOR_REGRESS), 
			dropout=DROPOUT_RATE_FOR_REGRESS, recurrent_dropout=DROPOUT_RATE_FOR_REGRESS,
			return_sequences=False) )
	model.add(Dropout(DROPOUT_RATE_FOR_REGRESS))
	model.add(Dense(NUMBER_OF_OUTPUT_NODES_FOR_REGRESS))
	model.add(Activation('linear'))
	optimizer = optimizers.Adam(lr=LEARNING_RATE_FOR_REGRESS)
	model.compile( loss='mean_squared_error', optimizer=optimizer )
	model.summary()
	
	return model
	
##################################################
# 学習経過ファイルのヘッダーを出力する。
##################################################
def output_whole_result_header():
	
	fo = open(RESULT_FILE_WHOLE, 'a')
	fo.write('##################################################\n')
	fo.write('入力データ = (水戸)x(気温,降水量,湿度,気圧)\n')
	fo.write('model_for_regress = %d x LSTM(%d x %d) x DropOut(%d) x %d\n'
		% (NUMBER_OF_INPUT_NODES_FOR_REGRESS, NUMBER_OF_INPUT_NODES_FOR_REGRESS,
		   NUMBER_OF_HIDDEN_NODES_FOR_REGRESS, NUMBER_OF_HIDDEN_NODES_FOR_REGRESS, 
		   NUMBER_OF_OUTPUT_NODES_FOR_REGRESS) )
	fo.write('dropout rate = %f\n' % (DROPOUT_RATE_FOR_REGRESS) )
	fo.write('optimizer = Adam(lr=%f)\n' % (LEARNING_RATE_FOR_REGRESS) )
	fo.write('length of sequence = %d\n' % (LENGTH_OF_SEQUENCE_FOR_REGRESS) )
	fo.write('length of shift = %d\n' % (LENGTH_OF_SHIFT_FOR_REGRESS) )
	fo.write('date: ' + get_datetime_string() + '\n')
	fo.write('##################################################\n')
	fo.write('epoch,loss\n')
	fo.close()
	
##################################################
# 学習結果をファイル出力する(回帰用)
##################################################
def output_result_for_regress(input, target, predicted, number):
#def output_result_for_regress(input, target, predicted, scaler_input, scaler_target,number):
	
	# 正解と予想結果をファイル出力
	filename = str.format('%s_%03d.csv' % (RESULT_FILE_NAME_FOR_REGRESS, number) )
	fo = open(filename, 'w')
	fo.write('水戸,,水戸(予測),,,,\n')
	fo.write('気温,湿度,気温,湿度\n')
	
	# 全テストデータの正解と予想結果出力
	data_len = input.shape[0]
	sequence_len = LENGTH_OF_SEQUENCE_FOR_REGRESS+LENGTH_OF_SHIFT_FOR_REGRESS + 1
	for i in range(data_len):
		
		# 入力と出力を取得
		input_i = input[i]
		
		if i < sequence_len :
			# 予想結果が出せないデータの場合(最初の方)
			fo.write('%.1f,%.1f\n' %
				(input_i[0], input_i[1],
				 ) )
		else:
			pi = i - sequence_len
			
			fo.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n' %
				(input_i[0], input_i[1],
				 target[pi,0], target[pi,1],
				 predicted[pi,0], predicted[pi,1],
				) )
				
	fo.close()
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 回帰用のデータ取得
	train_input_raw_for_regress, train_target_raw_for_regress, \
	test_input_raw_for_regress,  test_target_raw_for_regress, \
	train_input_for_regress, train_target_for_regress, \
	test_input_for_regress,  test_target_for_regress, \
	scaler_input_for_regress, scaler_target_for_regress = get_data_for_regress()
	
	# 回帰用のモデル作成
	model_for_regress = make_model_for_regress()
	
	# 結果ファイルのヘッダー出力
	output_whole_result_header()
	
	# 学習実行
	for i in range(NUMBER_OF_TRAINING):
		
		# 学習
		model_for_regress.fit(
				train_input_for_regress, train_target_for_regress, 
				batch_size=SIZE_OF_BATCH_FOR_REGRESS, 
				epochs=NUMBER_OF_EPOCHS, verbose=1)
		
		# 結果出力
		loss_regress = model_for_regress.evaluate(test_input_for_regress, test_target_for_regress, verbose=0)
		print('%07d : loss=%f' % ((i+1)*NUMBER_OF_EPOCHS, loss_regress))
		fo = open(RESULT_FILE_WHOLE, 'a')
		fo.write('%07d,%f\n' % ((i+1)*NUMBER_OF_EPOCHS, loss_regress))
		fo.close()
		
		# 正解と予想結果をファイル出力
		if (i % OUTPUT_CYCLE) == 0 :
			
			# 回帰の結果出力
			predicted_r = model_for_regress.predict(test_input_for_regress)
			predicted_r_inversed = scaler_target_for_regress.inverse_transform(predicted_r)
			predicted_r = predicted_r_inversed + test_target_raw_for_regress[LENGTH_OF_SEQUENCE_FOR_REGRESS:-1]
			#input_r_inversed = scaler_input_for_regress.inverse_transform(test_input_for_regress[:,0,:])
			#target_r_inversed = scaler_target_for_regress.inverse_transform(test_target_for_regress)
			input_r = test_input_raw_for_regress
			target_r = test_target_raw_for_regress
			#output_result_for_regress(input_r_inversed, target_r_inversed, predicted_r_inversed, i)
			output_result_for_regress(input_r, predicted_r_inversed, predicted_r, i)
			

