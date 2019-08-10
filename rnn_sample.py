# -*- coding: utf-8 -*-
'''
  RNNの動作確認用1
    48時間分の気温,湿度から、6時間後の気温,湿度を予測する
'''

import sys, os
from common.file import *
from format import *
import numpy
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

LENGTH_OF_SEQUENCES = 48	# 過去48時間のデータから予測する
LENGTH_OF_SHIFT = 6 		# 6時間後のデータを予測する
NUMBER_OF_INPUT_NODES = 2	# 入力データ数(気温,湿度)
NUMBER_OF_HIDDEN_NODES = 64	# 隠れ層のノード数
NUMBER_OF_OUTPUT_NODES = 2	# 出力データ数(気温,湿度)
SIZE_OF_BATCH = 128		# バッチサイズ
NUMBER_OF_EPOCHS = 10		# 1回の学習のエポック数
NUMBER_OF_TRAINING = 10		# 学習回数

##################################################
# 学習用データ取得
##################################################
def load_data(dir_path):
	
	temperature = numpy.array([])
	humidity = numpy.array([])
	
	csv_paths = get_filepaths(dir_path, '.csv')
	for csv_path in csv_paths:
		point_name = '水戸'
		csv_data = read_weather_csv(csv_path)
		temp = get_temperature(csv_data, point_name)
		temperature = numpy.append(temperature, temp)
		humi = get_humidity(csv_data, point_name)
		humidity = numpy.append(humidity, humi)
		
	re_data = numpy.stack([temperature, humidity], 1)
	
	return re_data
	
##################################################
# データセット作成
##################################################
def make_dataset(input_data):
	
	data_len = input_data.shape[0]
	data_num = input_data.shape[1]
	
	re_data   = numpy.zeros( (data_len - LENGTH_OF_SEQUENCES - LENGTH_OF_SHIFT, LENGTH_OF_SEQUENCES, data_num) )
	re_target = numpy.zeros( (data_len - LENGTH_OF_SEQUENCES - LENGTH_OF_SHIFT, data_num) )
	
	for i in range(data_len - LENGTH_OF_SEQUENCES-LENGTH_OF_SHIFT):
		
		re_data[i,0:LENGTH_OF_SEQUENCES,] = input_data[i:i+LENGTH_OF_SEQUENCES,]
		re_target[i,] = input_data[i+LENGTH_OF_SEQUENCES+LENGTH_OF_SHIFT,]
		
	return re_data, re_target
	
##################################################
# Max-Minスケール化(0〜1の範囲に収まるように標準化)
##################################################
def max_min_scale(train_data, test_data):
	
	scaler = MinMaxScaler()
	train_scaled = scaler.fit_transform(train_data)
	test_scaled  = scaler.transform(test_data)
	
	return scaler, train_scaled, test_scaled
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 学習用データ取得
	train_data = load_data('./train')
	
	# テスト用データ取得
	test_data = load_data('./test')
	
	# Max-Minスケール化
	scaler, train_scaled, test_scaled = max_min_scale(train_data, test_data)
	#train_target, test_target = max_min_scale(train_target, test_target)
	
	train_input, train_target = make_dataset(train_scaled)
	test_input, test_target = make_dataset(test_scaled)
	
	# モデル作成
	model = Sequential()
	model.add( LSTM( NUMBER_OF_HIDDEN_NODES, 
			batch_input_shape=(None, LENGTH_OF_SEQUENCES, NUMBER_OF_OUTPUT_NODES), 
			return_sequences=False) )
	model.add(Dense(NUMBER_OF_OUTPUT_NODES))
	model.add(Activation('linear'))
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	
	# 学習実行
	for i in range(NUMBER_OF_TRAINING):
		model.fit(train_input, train_target, batch_size=SIZE_OF_BATCH, 
				nb_epoch=NUMBER_OF_EPOCHS, validation_split=0.05)
		
		# 結果出力
		predicted = model.predict(test_input)
		predicted = scaler.inverse_transform(predicted)
		filename = str.format('./rnn_result_190810_03_%02d.csv' % (i) )
		fo = open(filename, 'w')
		fo.write('気温(正解),湿度(正解),気温(予測),湿度(予測)\n')
		test_data_len = test_data.shape[0]
		for i in range(test_data_len):
			correct = test_data[i]
			if i < (LENGTH_OF_SEQUENCES+LENGTH_OF_SHIFT):
				fo.write('%f,%f\n' % (correct[0], correct[1]) )
				#print('%f,' % (correct) )
			else:
				pi = i - LENGTH_OF_SEQUENCES - LENGTH_OF_SHIFT
				fo.write('%f,%f,%f,%f\n' % 
					(correct[0], correct[1], predicted[pi,0], predicted[pi,1]) )
				#print('%f,%f' % (correct, predicted[pi]) )
	
