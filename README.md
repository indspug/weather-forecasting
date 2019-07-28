weather-forecasting
====

Pythonで天気予測

## Description

## Demo

## VS. 

## Requirement

- python
- numpy
- tensorflow
- keras

## Usage

```bash
python simple_dnn.py [train_csv_dirpath] [test_csv_dirpath]
```

- train_csv_dirpath: 学習用CSVファイル格納ディレクトリ
- test_csv_dirpath : テスト用CSVファイル格納ディレクトリ

### CSVファイルの用意

学習・テスト用のCSVファイルを気象庁のページからダウンロードする。
[気象庁|過去の気象データ・ダウンロード](https://www.data.jma.go.jp/risk/obsdl/index.php)

### ディレクトリ構成の例

```
simple_dnn.py
train
├── mito_20180101-20180331data.csv
├── mito_20180401-20180630data.csv
└── mito_20180701-20180930data.csv
test
└── mito_20181001-20181231data.csv
```

### 実行例

```bash
python simple_dnn.py train test
```

## Install

### Python 2.7 (on Amazon Linux)

python, pip, virtualenvをインストールする。
```bash
sudo yum install python27
sudo yum install python27-pip
pip install virtualenv
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

仮想環境にnumpy,tensorflow,kerasをインストールする。
```bash
pip install numpy
pip install tensorflow
pip install keras
```

GitHubからリポジトリをクローンする。
```bash
git clone 'https://github.com/indspug/weather-forecasting'
```

作業が終わったら仮想環境を無効にする。
```bash
deactivate
```
### Python3 (on Amazonn Linux2)

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

仮想環境にnumpy,tensorflow,kerasをインストールする。
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
