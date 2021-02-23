# -*- coding: utf-8 -*-

# --------------------------
# 前回作成ファイルをクリーンアップ
# --------------------------
# (1) <パッケージ名>.egg-info ディレクトリ
# (2) dist ディレクトリ
rm -f -r ./pyqcstrc.egg-info/* dist/*


# --------------------------
# ソースコード配布物を作成
# --------------------------
python3 setup.py sdist

# --------------------------
# ライブラリのパッケージ作成
# --------------------------
python3 setup.py bdist_wheel


# --------------------------
# ライブラリをPyPIにアップロード
# --------------------------
# PyPIは、同一バージョン番号で上書きアップロードできないので注意．
# -----
# テスト
# -----
twine upload --repository testpypi dist/*

# PyPIページでアップロードされたことを確認
# https://test.pypi.org/project/pyqcstrc/

# パッケージがインストールできることを確認
pip3 --no-cache-dir install --upgrade --index-url https://test.pypi.org/simple/ pyqcstrc

# ---------------
# 本番アップロード
# ---------------
twine upload --repository pypi dist/*

# PyPIページでアップロードされたことを確認
https://pypi.org/project/pyqcstrc/

#パッケージがインストールできることを確認
pip3 --no-cache-dir install --upgrade pyqcstrc

