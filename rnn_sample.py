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
	
	train_csv = './train/mito_20170101-20170115data.csv'
	test_csv  = './train/mito_20170116-20170131data.csv'
	
	# 学習用ファイル読込
	csv_data = read_weather_csv(train_csv)
	
	# 気温取得
	point_name = '水戸'
	temperature = get_temperature(csv_data, point_name)
	input_data = numpy.stack( [temperature], 1 )
	
	# データセット作成
	g, h = make_dataset_temperature(input_data)
	
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
	model.fit(g, h, batch_size=64, nb_epoch=100, validation_split=0.05)
	
	# 結果出力
	predicted = model.predict(g)
	print(predicted.shape)
	fo = open('./rnn_result.csv', 'w')
	input_data_len = input_data.shape[0]
	for i in range(input_data_len):
		correct = input_data[i,0]
		if i < LENGTH_OF_SEQUENCES:
			fo.write('%f,\n' % (correct) )
			print('%f,' % (correct) )
		else:
			pi = i - LENGTH_OF_SEQUENCES
			fo.write('%f,%f\n' % (correct, predicted[pi]) )
			print('%f,%f' % (correct, predicted[pi]) )
	
