# -*- coding: utf-8 -*-

"""
  formatモジュールの定数定義ファイル
"""

import math

##################################################
# 定数
##################################################
ROW_HEADER_START = 3
ROW_PLACE_NAME = ROW_HEADER_START
ROW_ITEM_NAME = ROW_HEADER_START + 1
ROW_DATA_START = 6

# 最低限許容できる品質
# (値の意味 8:欠損なし, 5:20%以下の欠損, 4,3,2,1:20%を超える欠損)
ACCEPTABLE_QUALITY = 5

# 風向を角度(-PI~PI)に変換する
WIND_DIRECTION_MAP = {
  '北': 0,          '北北東': math.pi/8,    '北東': math.pi/4,    '東北東': math.pi*3/8,
  '東': math.pi/2,  '東南東': math.pi*5/8,  '南東': math.pi*3/4,  '南南東': math.pi*7/8, 
  '南': math.pi,    '南南西': -math.pi*7/8, '南西': -math.pi*3/4, '西南西': -math.pi*5/8,
  '西': -math.pi/2, '西北西': -math.pi*3/8, '北西': -math.pi/4,   '北北西': -math.pi/8,
  '静穏':0
}

# 天気の分類ルール
WEATHER_CLASS_NUM = 3
WEATHER_SUNNY = 0	# 晴れ
WEATHER_CLOUDY = 1	# くもり
WEATHER_RAINY = 2	# 雨
WEATHER_CLASSIFY_MAP = {
  1  : WEATHER_SUNNY,	# 快晴
  2  : WEATHER_SUNNY,	# 晴れ
  3  : WEATHER_CLOUDY,	# 薄曇
  4  : WEATHER_CLOUDY,	# 曇
  5  : WEATHER_CLOUDY,	# 煙霧
  6  : WEATHER_CLOUDY,	# 砂じん嵐
  7  : WEATHER_CLOUDY,	# 地ふぶき
  8  : WEATHER_RAINY,	# 霧
  9  : WEATHER_RAINY,	# 霧雨
  10 : WEATHER_RAINY,	# 雨
  11 : WEATHER_RAINY,	# みぞれ
  12 : WEATHER_RAINY,	# 雪
  13 : WEATHER_RAINY,	# あられ
  14 : WEATHER_RAINY,	# ひょう
  15 : WEATHER_RAINY,	# 雷
  16 : WEATHER_RAINY,	# しゅう雨または止み間のある雨	
  17 : WEATHER_RAINY,	# 着氷性の雨	
  18 : WEATHER_RAINY,	# 着氷性の霧雨	
  19 : WEATHER_RAINY,	# しゅう雪または止み間のある雪	
  22 : WEATHER_RAINY,	# 霧雪
  23 : WEATHER_RAINY,	# 凍雨
  24 : WEATHER_RAINY,	# 細氷
  28 : WEATHER_RAINY,	# もや
  101: WEATHER_RAINY,	# 降水またはしゅう雨性の降水
}

