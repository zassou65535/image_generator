# image_generator
## 概要
メルアイコン生成器で用いたソースコードです。  
詳しい解説は<a href="https://qiita.com/zassou65535/items/cad3f61177880e8230ab">こちら</a>。

## 想定環境
python 3.7.1

## 使い方
1. `GAN.py`と同じディレクトリに、`dataset`という名前のディレクトリを作成
1. `GAN.py`と同じディレクトリに、`img`という名前のディレクトリを作成
1. `dataset`ディレクトリに、学習に使いたい画像を`*.jpg`という形式で好きな数入れる
1. `GAN.py`の置いてあるディレクトリで`python GAN.py`を実行
1. 学習し、その結果を用いて画像を生成、`img`ディレクトリ以下に生成結果を出力します  

学習には環境によっては10時間以上要する場合がありますので、注意いただきますようお願いします。   
入力された画像は64×64にリサイズされた上で学習に使われます。出力画像も64×64です。 

## Pillowというライブラリに関してエラーが出る場合
`pip install Pillow==6.1`  
Pillowのバージョンを6.1に指定すると動く場合があります。

