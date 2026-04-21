#
# 2026-04-17 Katsutoshi Yoshida
# 
import sympy as sp
from . import symPDFbase

class symPDFs(symPDFbase):
    '''
    確率密度関数の定義クラス
    '''
    def uniform1d(self):
        '''
        一様分布（1変数）
        '''
        x   = sp.symbols(r'x', real=True)   #独立変数（実数）
        a   = sp.symbols(r'a', real=True)   #下限（実数）
        b   = sp.symbols(r'b', real=True)   #上限（実数）

        xs, prm = (x,), (a, b)

        # 定義式
        p = sp.Piecewise(
            (0, x < a), (1, x <= b), (0, True)
            # if x<a: y=0 elif x<=b: y=1: else y=0
        ) / sp.Abs(b-a)

        # 結果
        return {
            'name': 'Uniform (1D)',
            'p':    p,      #確率密度
            'x':    xs,     #独立変数
            'prm':  prm     #パラメータ
        }

    def gauss1d(self):
        '''
        ガウス分布（1変数）
        '''
        x   = sp.symbols(r'x', real=True)       #独立変数（実数）
        m   = sp.symbols(r'm', real=True)       #平均（実数）
        s   = sp.symbols(r's', positive=True)   #標準偏差（正の実数）

        xs, prm = (x,), (m, s)

        # 定義式
        p = sp.exp(
            -(x - m)**2 / (2 * s**2)
        ) / (sp.sqrt(2 * sp.pi) * s)

        # 結果
        return {
            'name': 'Gaussian (1D)',
            'p':    p,      #確率密度
            'x':    xs,     #独立変数
            'prm':  prm     #パラメータ
        }

    def gauss2d(self):
        '''
        ガウス分布（2変数）
        '''
        x1, x2 = sp.symbols(r'x_1, x_2', real=True)     #独立変数（実数）
        m1, m2 = sp.symbols(r'm_1, m_2', real=True)     #平均（実数）
        s1, s2 = sp.symbols(r's_1, s_2', positive=True) #標準偏差（正の実数）
        rho    = sp.symbols(r'r', positive=True)        #相関係数（1 <= rho <= 1）

        xs, prm = (x1, x2), (m1, m2, s1, s2, rho)

        # 定義式
        p = sp.exp(
            -1/(2*(1-rho**2))*(
                (x1-m1)**2/(s1**2) + (x2-m2)**2/(s2**2)
                - 2*rho*(x1-m1)*(x2-m2)/(s1*s2)
            )
        ) / (2*sp.pi*s1*s2 * sp.sqrt(1-rho**2))

        # 結果
        return {
            'name': 'Gaussian (2D)',
            'p':    p,      #確率密度
            'x':    xs,     #独立変数
            'prm':  prm     #パラメータ
        }

    def rho_dummify(self, expr):
        '''
        簡約用の正数 rho_dummy := sqrt(1-rho**2)>0
        - positive=True の変数に置き換えて Sympy に整理を促す
        '''
        rho = self.pdf['prm'][4]
        self.rho_dummy = sp.symbols(r'SQRT(1-rho2)', positive=True)
        expr = expr.subs(rho**2, 1 - self.rho_dummy**2)
        return expr

    def rho_undummify(self, expr):
        '''簡約用の正数 rho_dummy を元に戻す'''
        rho = self.pdf['prm'][4]
        expr = expr.subs(self.rho_dummy**2, 1 - rho**2)
        return expr
