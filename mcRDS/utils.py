#
# 2026-04-28 Katsutoshi Yoshida
# 
__all__ = ['displayEQ']

import sympy as sp  #数式処理
from IPython.display import Math, display #display の整形用

def displayEQ(leftsym, rightsym, eq='='):
    """等式・定義式の表示"""
    if not isinstance(leftsym, str): #latex文字列はそのまま
        leftsym = sp.latex(leftsym)
    if not isinstance(rightsym, str):
        rightsym = sp.latex(rightsym)
    display(Math( leftsym + eq + rightsym ))
