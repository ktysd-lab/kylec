#
# 2026-04-28 Katsutoshi Yoshida
# 
__all__ = ['displayEQ']

import sympy as sp  #数式処理
from IPython.display import Math, display #display の整形用

def displayEQ(leftsym, rightsym, eq='='):
    """等式・定義式の表示"""
    if isinstance(leftsym, sp.Basic):
        leftsym = sp.latex(leftsym)
    if isinstance(rightsym, sp.Basic):
        rightsym = sp.latex(rightsym)
    display(Math( leftsym + eq + rightsym ))
