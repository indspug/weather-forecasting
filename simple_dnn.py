# -*- coding: utf-8 -*-
  
import sys, os
import time
import datetime
from common.file import *
from format import *
from common.processing import *
from model import *
import numpy
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

POINT_NAME = '水戸'
MODEL_DIR  = 'ckpt'	# モデル保存先ディレクトリ
MODEL_NAME = 'model'	# 保存するモデルのファイル名
NUM_EPOCH = 10		# 何回繰り返し学習させるか
NUM_TRAING = 100	# 学習回数(NUM_TRAING * NUM_EPOCH)
SAVE_CYCLE = 10		# 保存周期(N回学習につき1回保存)
RESULT_FILE = 'result.csv'	# 学習経過保存ファイル
OUTPUT_CYCLE = 1		# 学習経過出力周期(N回学習につき1回出力)
RAINFALL_INDEX = 0

##################################################
# 指定したディレクトリしたCSVファイル群から気象データを読み込み、
# 学習に必要なデータを抽出して返す
# [Args]
#   dirpath  : CSVファイルが格納されたディレクトリのパス
# [Return]
#   input_data : 入力データ(ndarray)
#   label_data : 出力データ(ndarray)
##################################################
def extract_learning_data(dirpath):
	
	print 'ディレクトリ: ' + dirpath
	
	input_data_list = []
	label_data_list = []
	
	csv_paths = get_filepaths(dirpath, '.csv')
	for csv_path in csv_paths:
		
		# CSVファイル読み込み
		csv_data = read_weather_csv(csv_path)
	
		# 入力データ取得：降水量,相対湿度,海面気圧
		rainfall = get_rainfall(csv_data, POINT_NAME)
		humidity = get_humidity(csv_data, POINT_NAME)
		sea_level_pressure = get_sea_level_pressure(csv_data, POINT_NAME)
		cloud_cover = get_cloud_cover(csv_data,POINT_NAME)
		#temperature = get_temperature(csv_data, POINT_NAME)
		#wind_speed = get_wind_speed(csv_data, POINT_NAME)
		#wind_dir = get_wind_direction(csv_data, POINT_NAME)
		#daylight = get_daylight(csv_data, POINT_NAME)
		#atom_pressure = get_atom_pressure(csv_data, POINT_NAME)
		
		# 時(hh)取得
		#datetime = get_datetime(csv_data)
		#datetime = extract_element_from_datetime(datetime)
		#hour = datetime[:,3]
		#hour = hour - 12.0  # 正午が0になるように調整
	
		# 入力データ結合
		input_data = numpy.stack(
			[rainfall, humidity, sea_level_pressure, cloud_cover], 1)
		
		# 出力データ取得：天気
		label_data = get_weather(csv_data, POINT_NAME)
		
		input_data_list.append(input_data)
		label_data_list.append(label_data)
		
	# 複数のCSVから読み込んだデータを結合する
	input_data = input_data_list[0]
	for i in range(2, len(input_data_list)):
		input_data = numpy.vstack( [input_data, input_data_list[i]] )
	
	label_data = label_data_list[0]
	for i in range(2, len(label_data_list)):
		label_data = numpy.vstack([label_data, label_data_list[i]])
	
	print '  読込データ数: %d' % input_data.shape[0]
	
	# 入力データ・出力データにNaNが含まれている行を削る
	#input_data, label_data = remove_nan_data(input_data, label_data)
	#print '  NaNデータ削除後: %d' % input_data.shape[0]
	
	# 入力データにNaNが含まれているデータを補間する
	input_data = interpolate_nan_input_data(input_data)
	label_data = interpolate_nan_label_data(label_data)
	
	
	# 不整合データを削除する
	is_inconsistency_row = validate_rainfall_inconsistency(
		input_data[:,RAINFALL_INDEX], label_data)
	input_data = input_data[~is_inconsistency_row,]
	label_data = label_data[~is_inconsistency_row,]
	
	print '  不整合データ削除後: %d' % input_data.shape[0]
	
	return (input_data, label_data)
	
##################################################
# 学習モデルを作成する
##################################################
def make_model(input_data_dim, label_num):
	
	# モデルの作成
	#  3 x 32 x 32 x 3 
	model = Sequential()
	model.add(Dense(32, input_dim=input_data_dim))
	model.add(Activation('relu'))
	model.add(Dropout(rate=0.5)),
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dropout(rate=0.5)),
	model.add(Dense(label_num))
	model.add(Activation('softmax'))
	#model = model_from_json('ckpt/model_0009.json')
	
	#optimizer = optimizers.SGD(lr=0.01)
	#optimizer = optimizers.Adam(lr=0.001)
	optimizer = optimizers.RMSprop(lr=0.001)
	
	model.compile(
		optimizer=optimizer,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	
	#load_weights(model, 'ckpt/model_0009.h5')
	
	return model
	
##################################################
# 学習経過ファイルのヘッダーを出力する。
##################################################
def output_result_header():
	
	fo = open(RESULT_FILE, 'a')
	fo.write('##################################################\n')
	fo.write('入力データ = 降水量 相対湿度 海面気圧 雲量\n')
	fo.write('model = 4 x 32 x 32 x 3\n')
	fo.write('optimizer = RMSprop(lr=0.001)\n')
	fo.write('date: ' + get_datetime_string() + '\n')
	fo.write('##################################################\n')
	fo.write('epoch, loss, acc, elapsed_time, acc_sunny, acc_cloudy, acc_rainy\n')
	fo.close()
	
##################################################
# 日付の文字列表現を返す
##################################################
def get_datetime_string():
	
	dn = datetime.datetime.now()
	return "%04d/%02d/%02d %02d:%02d:%02d" % (
		dn.year, dn.month, dn.day,
		dn.hour, dn.minute, dn.second	
	)
	
##################################################
# 天気ごとの正解率を計算する。
##################################################
def get_acc_by_weather(model, input_data, label_data, random_num):
	
	data_num = input_data.shape[0]
	label_num = label_data.shape[1]
	acc = [0] * label_num
	num = [0] * label_num
	
	# ランダムでデータ取り出し
	rand_index = numpy.random.randint(0, data_num, size=random_num)
	random_input = input_data[rand_index]
	random_label = label_data[rand_index]
	
	# 予測
	predict_label = model.predict(random_input)
	
	# 正解と予測結果を比較正解率の計算
	for i in range(random_num):
		
		# 正解
		correct_idx = numpy.argmax(random_label[i])
		
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
# 学習経過をCSV形式でファイルに出力する。
##################################################
def output_result(format, objects):
	
	fo = open(RESULT_FILE, 'a')
	#fo.write( "%d, %f, %f, %f\n" % (epoch, loss, acc, elapsed_time) )
	fo.write( format % objects )
	fo.close()
	
##################################################
# メイン
# [Args]
#   train_csv_dirpath : 訓練用CSVファイルを格納したディレクトリのパス
#   test_csv_dirpath  : テスト用CSVファイルを格納したディレクトリのパス
##################################################
if __name__ == '__main__':
	
	# コマンドライン引数のチェック
	argvs = sys.argv
	argc = len(argvs)
	if argc < 2:
		exe_name=argvs[0]
		print('Usage: python %s [train_csv_dirpath] [test_csv_dirpath]' % exe_name)
		quit()
	
	# CSVファイルパス取り出し
	train_csv_dirpath = argvs[1]
	test_csv_dirpath  = argvs[2]
	
	# CSVファイルから気象データを取得し、
	# 学習用の入力データと出力データを抽出
	train_input, train_label = extract_learning_data(train_csv_dirpath)
	test_input, test_label = extract_learning_data(test_csv_dirpath)
	
	# 入力データをMAX-MIN標準化する
	train_input, test_input = max_min_normalize([train_input, test_input], axis=0)
	
	# モデルの作成
	input_data_dim = train_input.shape[1]
	label_num = train_label.shape[1]
	model = make_model(input_data_dim, label_num)
	
	# 学習経過保存ファイルのヘッダー出力
	output_result_header()
	
	# 学習実行
	epoch = 0	# epoch : 一つの訓練データを何回繰り返して学習させるか
	start_time = time.time()
	for i in range(NUM_TRAING):
                # 学習
		model.fit(
			train_input, train_label, 
			epochs=NUM_EPOCH, batch_size=128, shuffle=True, verbose=0)
		
		# 評価
		#  loss: 損失値(正解データとの誤差。小さい方が良い。)
		#  acc : 評価値(正解率。高い方が良い。)
		score = model.evaluate(test_input, test_label, verbose=0)
		loss = score[0]
		acc = score[1]
		print('%07d : loss=%f, acc=%f' % (epoch, loss, acc))
		
		# エポック数の合計加算
		epoch = epoch + NUM_EPOCH
	
		# 学習モデルの保存
		if (i % SAVE_CYCLE) == 0 :
			save_model(model, MODEL_DIR, 'model', i)
		
		# 学習経過の保存
		if (i % OUTPUT_CYCLE) == 0 :
			elapsed_time = time.time() - start_time
			acc_w = get_acc_by_weather(model, test_input, test_label, 100)
			output_result( "%d, %f, %f, %f, %f, %f, %f\n", 
			               (epoch, loss, acc, elapsed_time, 
			                acc_w[0], acc_w[1], acc_w[2]) )
		
