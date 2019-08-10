# -*- coding: utf-8 -*-
'''
  RNNの動作確認用
'''

import sys, os
from common.file import *
from format import *
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

LENGTH_OF_SEQUENCES = 24
NUMBER_OF_EPOCHS = 30

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
		
	#re_temp = numpy.array(temperature).reshape(temperature.shape[0], 1)
	#re_humi = numpy.array(humidity).reshape(humidity.shape[0], 1)
	
	re_data = numpy.stack([temperature, humidity], 1)
	
	return re_data
	
##################################################
# データセット作成
##################################################
def make_dataset(input_data):
	
	data, target = [], []
	LENGTH_OF_SEQUENCES = 24	# 24データを1シーケンスとする
	
	data_len = input_data.shape[0]
	data_num = input_data.shape[1]
	
	re_data   = numpy.zeros( (data_len - LENGTH_OF_SEQUENCES, LENGTH_OF_SEQUENCES, data_num) )
	re_target = numpy.zeros( (data_len - LENGTH_OF_SEQUENCES, data_num) )
	
	for i in range(data_len - LENGTH_OF_SEQUENCES):
		
		re_data[i,0:LENGTH_OF_SEQUENCES,] = input_data[i:i+LENGTH_OF_SEQUENCES,]
		re_target[i,] = input_data[i+LENGTH_OF_SEQUENCES,]
		#data.append(input_data[i:i + LENGTH_OF_SEQUENCES,0])
		#target.append(input_data[i + LENGTH_OF_SEQUENCES,0])
		
	#re_data = numpy.array(data).reshape(len(data), LENGTH_OF_SEQUENCES, 1)
	#re_target = numpy.array(target).reshape(len(data), 1)
	
	#print('=======================')
	#print(re_data)
	#print('=======================')
	#print(re_target)
	#print('=======================')
	#print(re_data.shape)
	#print(re_target.shape)
	
	return re_data, re_target
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 学習用データ取得
	train_data = load_data('./train')
	
	# データセット作成
	input, target = make_dataset(train_data)
	
	# モデル作成
	model = Sequential()
	model.add( LSTM( 64, 
			batch_input_shape=(None, LENGTH_OF_SEQUENCES, 2), 
			return_sequences=False) )
	model.add(Dense(2))
	model.add(Activation('linear'))
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	
	# 学習実行
	model.fit(input, target, batch_size=64, nb_epoch=NUMBER_OF_EPOCHS, validation_split=0.05)
	
	# テスト用データ取得
	test_data = load_data('./test')
	input, target = make_dataset(test_data)
	
	# 結果出力
	predicted = model.predict(input)
	fo = open('./rnn_result.csv', 'w')
	fo.write('気温(正解),湿度(正解),気温(予測),湿度(予測)\n')
	test_data_len = test_data.shape[0]
	for i in range(test_data_len):
		correct = test_data[i]
		if i < LENGTH_OF_SEQUENCES:
			fo.write('%f,%f\n' % (correct[0], correct[1]) )
			#print('%f,' % (correct) )
		else:
			pi = i - LENGTH_OF_SEQUENCES
			fo.write('%f,%f,%f,%f\n' % 
				(correct[0], correct[1], predicted[pi,0], predicted[pi,1]) )
			#print('%f,%f' % (correct, predicted[pi]) )
	
