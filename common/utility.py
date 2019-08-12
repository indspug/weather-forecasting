# -*- coding: utf-8 -*-

"""
  ユーティリティモジュール　
"""

import datetime

##################################################
# 日付の文字列表現を返す
##################################################
def get_datetime_string():
	
	dn = datetime.datetime.now()
	return "%04d/%02d/%02d %02d:%02d:%02d" % (
		dn.year, dn.month, dn.day,
		dn.hour, dn.minute, dn.second	
	)

