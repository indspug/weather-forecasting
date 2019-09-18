# -*- coding: utf-8 -*-
"""
  DNNを用いた天気予測(分類)
"""

import sys, os
sys.path.append(os.getcwd())
from common.file import *
from common.utility import *
from common.processing import *
from model import *
from format import *
import numpy
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras import optimizers

##############################
# 分類用のパラメータ
##############################
NUMBER_OF_INPUT_NODES = 16	# 入力データ数:(水戸,前橋,東京,静岡,大阪,秩父,河口湖,新潟,宇都宮)x(気温,降水量,湿度,気圧)
NUMBER_OF_HIDDEN_NODES1 = 64	# 隠れ層のノード数1
NUMBER_OF_HIDDEN_NODES2 = 64	# 隠れ層のノード数2
NUMBER_OF_OUTPUT_NODES = 3	# 出力データ数(晴れ,曇り,雨)
DROPOUT_RATE = 0.2		# ドロップアウト率
LEARNING_RATE = 0.0001    	# 学習率
SIZE_OF_BATCH = 128		# バッチサイズ
RESULT_FILE_NAME  = './result3/result_190917_dnn2_04'

##############################
# 共通のパラメータ
##############################
NUMBER_OF_EPOCHS = 100		# 1回の学習のエポック数
NUMBER_OF_TRAINING = 10000	# 学習回数
OUTPUT_CYCLE = 10		# 学習経過出力周期
RESULT_FILE_WHOLE = './result3/result_190917_dnn2_04.csv'
SAVE_CYCLE = 1000		# 保存周期(N回学習につき1回保存)
MODEL_DIR  = 'ckpt'		# モデル保存先ディレクトリ
MODEL_NAME = 'model_190917_dnn2_04'	# 保存するモデルのファイル名
NUMBER_OF_POINTS = 4		# 入力データの地点数

##################################################
# 学習用データ取得(分類用)
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
		temp = get_temperature(csv_data, point_name)
		temperature = numpy.append(temperature, temp)
		rain = get_rainfall(csv_data, point_name)
		rainfall = numpy.append(rainfall, rain)
		humi = get_humidity(csv_data, point_name)
		humidity = numpy.append(humidity, humi)
		if point_name == '河口湖':
			pres = get_atom_pressure(csv_data, point_name)
		else:
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
# データ取得(分類用)
##################################################
def get_data():
	
	# 学習用データ取得
	train_input_raw1, train_target_raw1 = load_data('./train/mito',        '水戸')
	train_input_raw2, train_target_raw2 = load_data('./train/maebashi',    '前橋')
	train_input_raw3, train_target_raw3 = load_data('./train/tokyo',       '東京')
	train_input_raw4, train_target_raw4 = load_data('./train/shizuoka',    '静岡')
	#train_input_raw5, train_target_raw5 = load_data('./train/osaka',       '大阪')
	#train_input_raw6, train_target_raw6 = load_data('./train/chichibu',    '秩父')
	#train_input_raw7, train_target_raw7 = load_data('./train/kawaguchiko', '河口湖')
	#train_input_raw8, train_target_raw8 = load_data('./train/nigata',      '新潟')
	#train_input_raw9, train_target_raw9 = load_data('./train/utsunomiya',  '宇都宮')
	train_input_raw = numpy.hstack( [
		 			train_input_raw1, train_input_raw2, train_input_raw3,
		 			train_input_raw4, #train_input_raw5,
		 			#train_input_raw6, train_input_raw7, train_input_raw8, train_input_raw9,
				] )
	#train_input_raw = train_input_raw1.reshape(train_input_raw1.shape[0], NUMBER_OF_INPUT_NODES)
	train_target_raw = train_target_raw1
	
	# テスト用データ取得
	test_input_raw1, test_target_raw1 = load_data('./test/mito',        '水戸')
	test_input_raw2, test_target_raw2 = load_data('./test/maebashi',    '前橋')
	test_input_raw3, test_target_raw3 = load_data('./test/tokyo',       '東京')
	test_input_raw4, test_target_raw4 = load_data('./test/shizuoka',    '静岡')
	#test_input_raw5, test_target_raw5 = load_data('./test/osaka',       '大阪')
	#test_input_raw6, test_target_raw6 = load_data('./test/chichibu',    '秩父')
	#test_input_raw7, test_target_raw7 = load_data('./test/kawaguchiko', '河口湖')
	#test_input_raw8, test_target_raw8 = load_data('./test/nigata',      '新潟')
	#test_input_raw9, test_target_raw9 = load_data('./test/utsunomiya',  '宇都宮')
	test_input_raw = numpy.hstack( [
					test_input_raw1, test_input_raw2, test_input_raw3,
					test_input_raw4, #test_input_raw5,
		 			#test_input_raw6, test_input_raw7, test_input_raw8, test_input_raw9,
				] )
	#test_input_raw = test_input_raw1.reshape(test_input_raw1.shape[0], NUMBER_OF_INPUT_NODES)
	test_target_raw = test_target_raw1
	
	# Max-Minスケール化
	scaler, train_input_scaled, test_input_scaled = \
		 max_min_scale(train_input_raw, test_input_raw)
	#train_target_scaled, test_target_scaled = max_min_scale(train_target, test_target)
	
	#train_input, train_target = make_dataset_for_class(train_input_scaled, train_target_raw)
	#test_input, test_target = make_dataset_for_class(test_input_scaled, test_target_raw)
	
	return train_input_scaled, train_target_raw, \
		test_input_scaled, test_target_raw, scaler
	
##################################################
# 学習用のモデルを作成する
##################################################
def make_model():
	
	# モデル作成
	model = Sequential()
	model.add(Dense(NUMBER_OF_HIDDEN_NODES1, input_dim=NUMBER_OF_INPUT_NODES))
	model.add(Activation('relu'))
	model.add(Dropout(rate=DROPOUT_RATE)),
	model.add(Dense(NUMBER_OF_HIDDEN_NODES2))
	model.add(Activation('relu'))
	model.add(Dropout(rate=DROPOUT_RATE)),
	model.add(Dense(NUMBER_OF_OUTPUT_NODES))
	model.add(Activation('softmax'))

	optimizer = optimizers.Adam(lr=LEARNING_RATE)
	
	model.compile(
		optimizer=optimizer,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	model.summary()
	
	return model
	
##################################################
# 学習経過ファイルのヘッダーを出力する。
##################################################
def output_whole_result_header():
	
	fo = open(RESULT_FILE_WHOLE, 'a')
	fo.write('##################################################\n')
	fo.write('入力データ = (水戸,前橋,東京,静岡)x(気温,降水量,湿度,気圧)\n')
	fo.write('model_for_class = %d x %d x DropOut(%d) x %d x DropOut(%d) x %d \n'
		% (NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES1, NUMBER_OF_HIDDEN_NODES1,
		   NUMBER_OF_HIDDEN_NODES2, NUMBER_OF_HIDDEN_NODES2, 
		   NUMBER_OF_OUTPUT_NODES) )
	fo.write('dropout rate = %f\n' % (DROPOUT_RATE) )
	fo.write('optimizer = Adam(lr=%f)\n' % (LEARNING_RATE) )
	fo.write('date: ' + get_datetime_string() + '\n')
	fo.write('##################################################\n')
	fo.write('epoch,loss acc\n')
	fo.close()
	
##################################################
# 天気ごとの正解率を計算する。
##################################################
def get_acc_by_weather(model, input_data, label_data):
	
	data_num = input_data.shape[0]
	label_num = label_data.shape[1]
	acc = [0] * label_num
	num = [0] * label_num
	
	# 予測
	predict_label = model.predict(input_data)
	
	# 正解と予測結果を比較正解率の計算
	for i in range(data_num):
		
		# 正解
		correct_idx = numpy.argmax(label_data[i])
		
		# 学習モデルで予測した結果
		predict_idx = numpy.argmax(predict_label[i])
		
		num[correct_idx] += 1
		if correct_idx == predict_idx:
			acc[correct_idx] += 1.0
	
	# 正解率の計算
	for i in range(label_num):
		if num[i] > 0:
			acc[i] /= num[i]
	
	return tuple(acc)
	
##################################################
# 学習結果をファイル出力する
##################################################
def output_result(input, target, predicted, number):
	
	data_len = input.shape[0]
	input_feature_num = input.shape[1]
	target_feature_num = target.shape[1]
	
	# 正解と予想結果をファイル出力
	filename = str.format('%s_%04d.csv' % (RESULT_FILE_NAME, number) )
	fo = open(filename, 'w')
	
	# ヘッダー１行目設定
	fo.write('水戸,,,,前橋,,,,東京,,,,静岡,,,,')
	#fo.write('水戸,,,,前橋,,,,東京,,,,静岡,,,,大阪,,,,秩父,,,,河口湖,,,,新潟,,,,宇都宮,,,,')
	fo.write('水戸(正解),,,,水戸(予測),,,,正解率,\n')
	fo.write('\n')
	
	# ヘッダー２行目設定
	for i in range(NUMBER_OF_POINTS):
		fo.write('気温,降水量,湿度,気圧,')
	fo.write('晴れ,曇り,雨,天気値,晴れ,曇り,雨,天気値,正解/不正解')
	fo.write('\n')
	

	# 全テストデータの正解と予想結果出力
	for i in range(data_len):
		
		# 入力
		for j in range(input_feature_num):
			fo.write('%.1f,' % (input[i,j]) )
		
		# 正解	
		for j in range(target_feature_num):
			fo.write('%.2f,' % (target[i,j]) )
		
		# 天気を0〜1の数値に変換
		correct_i = numpy.argmax(target[i])
		correct_value = float(correct_i) / float(WEATHER_CLASS_NUM - 1)
		fo.write('%.2f,' % (correct_value) )
		
		# 予想結果	
		for j in range(target_feature_num):
			fo.write('%.2f,' % (predicted[i,j]) )
		
		# 天気を0〜1の数値に変換
		predict_i = numpy.argmax(predicted[i])
		predict_value = float(predict_i) / float(WEATHER_CLASS_NUM - 1)
		fo.write('%.2f,' % (predict_value) )
		
		# 正解/不正解
		correct = 1 if (correct_i == predict_i) else 0
		fo.write('%d\n' % (correct) )
				
	fo.close()

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 分類用のデータ取得
	train_input, train_target, \
	test_input, test_target, scaler = get_data()
	
	# 分類用のモデル作成
	model = make_model()
	
	# 結果ファイルのヘッダー出力
	output_whole_result_header()
	
	# 学習実行
	for i in range(NUMBER_OF_TRAINING):
		
		# 学習
		model.fit(
			train_input, train_target, batch_size=SIZE_OF_BATCH, 
			epochs=NUMBER_OF_EPOCHS, validation_split=0, verbose=0)
		
		# 結果出力
		loss_class, acc = model.evaluate(test_input, test_target, verbose=0)
		acc_w = get_acc_by_weather(model, test_input, test_target)
		print('%07d : loss_c=%f, acc=%f, acc_w=%.2f,%.2f,%.2f' % ( \
			(i+1)*NUMBER_OF_EPOCHS, loss_class, acc, acc_w[0], acc_w[1], acc_w[2],
			))
		fo = open(RESULT_FILE_WHOLE, 'a')
		fo.write('%07d,%f,%f,%f,%f,%f\n' % ( \
				(i+1)*NUMBER_OF_EPOCHS, loss_class, acc, \
				acc_w[0], acc_w[1], acc_w[2],
			))
		fo.close()
		
		# 正解と予想結果をファイル出力
		if (i % OUTPUT_CYCLE) == 0 :
			
			# 分類の結果出力
			predicted = model.predict(test_input)
			input_inversed = scaler.inverse_transform(test_input)
			output_result(input_inversed, test_target, predicted, i)
		
		# 学習モデルの保存
		if (i % SAVE_CYCLE) == 0 :
			save_model(model, MODEL_DIR, 'model', i)

