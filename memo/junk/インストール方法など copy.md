# インストール方法

# 参考：https://techblog.asahi-net.co.jp/entry/2018/06/15/162951

# wheel 形式のビルド済み配布物の作成
# wheel 形式 (*.whl) は PEP 427 で定められているビルド済み配布物のフォーマットであり、現在のデファクトスタンダードです。 これを作成するには最初に wheel パッケージをインストールしておく必要があります。

pip3 install wheel

# 以下のコマンドで dist/*.whl が生成されます。

python3 setup.py bdist_wheel

#-------- 仮想環境を作ってビルド済み配布物をテストする場合
mkdir packaging_test
cd packaging_test
python3 -m venv .venv
source .venv/bin/activate
pip install ../dist/pyqcstrc-0.0.1a4-cp39-cp39-macosx_10_14_x86_64.whl

#-------- pyqcstrc/examplesにあるスクリプトのテスト
# 論文中に示したexample
cd ../pyqcstrc/examples
./example1.py  # Normal
./example2.py  # 凸多面体用
./example3.py  # Asymmetric unitのみ

# speed check
cd ./pyqcstrc/scripts
./pyqcstrc-test1.py


#-------- 仮想環境で配布物のテストが完了し、後片付けする.
cd ../../
deactivate
python3 ./setup.py clean --all
rm -r ./packaging_test/.venv

