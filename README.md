weather-forecasting
====

Pythonで天気予測

## Description

## Demo

## VS. 

## Requirement

- python3
- numpy
- tensorflow
- keras

## Usage

```bash
python simple_dnn.py [train_csv_path] [test_csv_path]
```

学習・テスト用のCSVファイルは気象庁からダウンロードしてください。
[気象庁|過去の気象データ・ダウンロード](https://www.data.jma.go.jp/risk/obsdl/index.php)

## Install

Amazon Linux2でのインストール手順です。

python3, pip3, virtualenvをインストールする。
```bash
sudo yum install python3
sudo yum install python3-pip
pip3 install virtualenv
```

virtualenvで仮想環境を作成する。
```bash
virtualenv --no-site-packages weather_forecasting
```

仮想環境を有効にする。
```bash
cd weather_forecasting
source bin/activate
```

仮想環境にnumpyをインストールする。
```bash
pip3 install numpy
pip3 install tensorflow
pip3 install keras
```

GitHubからリポジトリをクローンする。
```bash
git clone 'https://github.com/indspug/weather-forecasting'
```

作業が終わったら仮想環境を無効にする。
```bash
deactivate
```

## Contribution

## Licence

## Author

[indspug](https://github.com/indspug)
