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

##################################################
# 学習用データ取得
##################################################
def load_temperature(dir_path):
	
	data = numpy.array([])
	
	csv_paths = get_filepaths(dir_path, '.csv')
	for csv_path in csv_paths:
		point_name = '水戸'
		csv_data = read_weather_csv(csv_path)
		temperature = get_temperature(csv_data, point_name)
		data = numpy.append(data, temperature)
		
	re_data = numpy.array(data).reshape(data.shape[0], 1)
	
	return re_data
	
##################################################
# データセット作成(気温)
##################################################
def make_dataset_temperature(input_data):
	
	data, target = [], []
	LENGTH_OF_SEQUENCES = 24	# 24データを1シーケンスとする
	
	input_data_len = input_data.shape[0]
	for i in range(input_data_len - LENGTH_OF_SEQUENCES):
		data.append(input_data[i:i + LENGTH_OF_SEQUENCES,0])
		target.append(input_data[i + LENGTH_OF_SEQUENCES,0])
		
	re_data = numpy.array(data).reshape(len(data), LENGTH_OF_SEQUENCES, 1)
	re_target = numpy.array(target).reshape(len(data), 1)
	
	return re_data, re_target
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 学習用データ取得
	train_data = load_temperature('./train')
	
	# データセット作成
	input, target = make_dataset_temperature(train_data)
	
	# モデル作成
	model = Sequential()
	model.add( LSTM( 64, 
			batch_input_shape=(None, LENGTH_OF_SEQUENCES, 1), 
			return_sequences=False) )
	model.add(Dense(1))
	model.add(Activation('linear'))
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	
	# 学習実行
	model.fit(input, target, batch_size=64, nb_epoch=10, validation_split=0.05)
	
	# テスト用データ取得
	test_data = load_temperature('./test')
	input, target = make_dataset_temperature(test_data)
	
	# 結果出力
	predicted = model.predict(input)
	fo = open('./rnn_result.csv', 'w')
	test_data_len = test_data.shape[0]
	for i in range(test_data_len):
		correct = test_data[i,0]
		if i < LENGTH_OF_SEQUENCES:
			fo.write('%f,\n' % (correct) )
			#print('%f,' % (correct) )
		else:
			pi = i - LENGTH_OF_SEQUENCES
			fo.write('%f,%f\n' % (correct, predicted[pi]) )
			#print('%f,%f' % (correct, predicted[pi]) )
	
