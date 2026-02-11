# Digit Recognition

このリポジトリは、数字認識のコードを含みます。
スクラッチのコードと、Pytorchのコードがあります。

## インストール手順

WindowsにPythonがインストール済みであることを前提にしています。

1. venv環境を作成する

    ~~~
    python -m venv venv
    .\testvenv\Scripts\activate
    ~~~

1. 必要なライブラリをインストール

    ~~~
    pip install matplotlib tqdm
    ~~~

1. リポジトリをクローン

    ~~~
    git clone https://github.com/dogcatr/digit_recognition.git
    cd digit_recognition
    ~~~

1. MNISTの画像データをダウンロード(pythonを実行するディレクト下に配置)

    [TensorFlowからダウンロード](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

1. 学習を実行

    ~~~
    python scratch\train.py
    ~~~

## TODOLIST

- Pytorchのコードを追加

## 参考

[ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
