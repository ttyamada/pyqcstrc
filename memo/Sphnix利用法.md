# -*- coding: utf-8 -*-

# --------------------------
# Sphinxを利用した文章作成
# --------------------------


sphinx-apidoc -f -o ./docs .
sphinx-build -b html ./docs ./docs/_build