# -*- coding: utf-8 -*-

#-------- pyxのコンパイル ---------
# --inplaceオプションを付けた場合、pyxのディレクトリにモジュールが作られる．
#python3 setup2.py build_ext --inplace

# path指定
python3 setup2.py build_ext --build-lib=$HOME/dev/build

ln -s pyqcstrc/icosah/occupation_domain.py $HOME/dev/build/pyqcstrc/icosah/
ln -s pyqcstrc/icosah/two_occupation_domains.py $HOME/dev/build/pyqcstrc/icosah/

#-------- 配布用パッケージの作成 ---------
# ./dist に whl ファイルをつくる．
#python3 setup.py bdist_wheel

# コマンドは開発モードでパッケージをインストール
# ソースコードを変更すると反映
#pip3 install -e .


# whlファイルはpip3でインストールできる.
#pip install ./dist/pyqcstrc-2021.2-cp39-cp39-macosx_10_14_x86_64.whl

#-------- 仮想環境を作って配布物をテストする場合 ---------
#mkdir packaging_test
#cd packaging_test
#python3 -m venv .venv
#source .venv/bin/activate
#pip install ../dist/pyqcstrc-2021.2-cp39-cp39-macosx_10_14_x86_64.whl

# 仮想環境で配布物のテストが完了し、後片付けするとき．
#deactivate
#python ../setup.py clean --all
#rm -r .venv


python3 setup.py bdist_wheel

cd packaging_test;python3 -m venv .venv
source .venv/bin/activate
pip install ../dist/pyqcstrc-2021.2-cp39-cp39-macosx_10_14_x86_64.whl
cd ../pyqcstrc/scripts

cd ../../packaging_test/
deactivate
rm -r .venv
cd ..
