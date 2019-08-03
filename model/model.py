# -*- coding: utf-8 -*-

"""
  kerasのモデル操作処理
"""

import os
import keras

##################################################
# 学習したモデルを保存する
# [Args]
#   model      : モデル(keras.model)
#   dir_path   : ディレクトリパス　
#   model_name : モデル名
#   number     : モデル名に付加する番号(省略可)
##################################################
def save_model(model, dir_path, model_name, number=0):
	
	model_to_json(model, dir_path, model_name, number)
	model_to_yaml(model, dir_path, model_name, number)
	save_weights(model, dir_path, model_name, number)
	
##################################################
# JSONファイルから学習したモデルをロードする
# [Args]
#   json_path : JSONファイルのパス
# [Return]
#   model     : モデル(keras.model)
##################################################
def model_from_json(json_path):
	json_string = open(json_path).read()
	model = keras.models.model_from_json(json_string)
	return model
	
##################################################
# YAMLファイルから学習したモデルをロードする
# [Args]
#   yaml_path : YAMLファイルのパス
# [Return]
#   model     : モデル(keras.model)
##################################################
def model_from_yaml(json_path):
	yaml_string = open(yaml_path).read()
	model = keras.models.model_from_yaml(yaml_string)
	return model
	
##################################################
# 保存した重みをファイルからロードする
# [Args]
#   model     : モデル(keras.model)
#   file_path : 重みを保存したファイルのパス
##################################################
def load_weights(model, file_path):
	model.load_weights(file_path)
	
##################################################
# モデルを保存するファイル名を取得する
#   numberを指定すると、モデル名にnumberを付加した
#   ファイル名を返す
#   例)model_00001
# [Args]
#   model_name : モデル名
#   number     : モデル名に付加する番号(省略可)
# [Rerutn]
#   filename   : モデルを保存するファイル名(拡張子無し)
##################################################
def get_model_filename(model_name, number=0):
	filename = '{0}_{1:04d}'.format(model_name, number)
	return filename

##################################################
# 学習したモデルをJSON形式で保存する
# [Args]
#   model      : モデル(keras.model)
#   dir_path   : ディレクトリパス　
#   model_name : モデル名
#   number     : モデル名に付加する番号(省略可)
##################################################
def model_to_json(model, dir_path, model_name, number=0):
	json_string = model.to_json()
	
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	
	file_name = get_model_filename(model_name, number)
	file_path = os.path.join(dir_path, file_name + '.json')
	
	open(file_path, 'w').write(json_string)
 
##################################################
# 学習したモデルをYAML形式で保存する
# [Args]
#   model      : モデル(keras.model)
#   dir_path   : ディレクトリパス　
#   model_name : モデル名
#   number     : モデル名に付加する番号(省略可)
##################################################
def model_to_yaml(model, dir_path, model_name, number=0):
	yaml_string = model.to_yaml()
	
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	
	file_name = get_model_filename(model_name, number)
	file_path = os.path.join(dir_path, file_name + '.yaml')
	
	open(file_path, 'w').write(yaml_string)
 
##################################################
# 学習した重みを保存する
# [Args]
#   model      : モデル(keras.model)
#   dir_path   : ディレクトリパス　
#   model_name : モデル名
#   number     : モデル名に付加する番号(省略可)
##################################################
def save_weights(model, dir_path, model_name, number=0):
	
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	
	file_name = get_model_filename(model_name, number)
	file_path = os.path.join(dir_path, file_name + '.h5')
	
	model.save_weights(file_path)
 
