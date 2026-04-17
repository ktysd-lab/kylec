#
# 2026-04-17 Katsutoshi Yoshida
# 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  #3Dプロットのカラーマップ
import sympy as sp  #数式処理
from IPython.display import Math #display の整形用

class PDFBase:
    '''
    確率密度関数の数式処理
    '''
    def __init__(self, pdfname=None, name=None, p=None, x=None, prm=None):
        '''
        コンストラクタ
        (1) 分布の選択 pdfname != None
        (2) 手動で代入
        '''
        if pdfname is not None:
            self.pdf  = eval(f"self.{pdfname}()")
        else:
            self.pdf = {'name':name, 'p':p, 'x':x, 'prm':prm}

        #定義式のNumpy関数化
        self.p_np = self.p_numpy(self.pdf)

    def __call__(self, key):
        '''
        pdf = PDFBase('gauss')
        pdf(key)
        '''
        return self.pdf[key]

    ##### 記号操作 #####
    def replace(self, x, prm):
        '''変数とパラメータの記号を入れ替える'''
        old_args = self.pdf['x'] + self.pdf['prm']
        new_args = x + prm
        new_p    = self.pdf['p']

        for old, new in zip(old_args, new_args):
            new_p = new_p.replace(old, new)

        self.pdf['p']   = new_p
        self.pdf['x']   = x
        self.pdf['prm'] = prm

    @staticmethod
    def exp_simplify(expr):
        '''exp() の中身(arg)だけ簡約'''
        return expr.replace(sp.exp, lambda arg: sp.exp(sp.factor(arg)))
        # return expr.replace(sp.exp, lambda arg: sp.exp(sp.simplify(arg)))

    ##### 平均操作 #####
    def averaging( self, fx, x ):
        '''平均操作'''
        result = sp.integrate(fx * self.pdf['p'], (x, -sp.oo, sp.oo))
        return sp.simplify(result)

    ##### Numpy化 #####
    @staticmethod
    def p_numpy(pdf):
        '''pのSympyの定義式 => Numpy 関数'''
        args = pdf['x'] + pdf['prm']    #確率密度関数の全引数
        lambdify = sp.lambdify(args, pdf['p'], "numpy")
        prm_names = [str(x) for x in pdf['prm']]
        return np.vectorize(lambdify, excluded=prm_names)

    ##### 表示・確認用のメソッド #####
    @staticmethod
    def render_eqn(left, right):
        '''等式の出力用整形'''
        return Math( sp.latex(left) + '=' + sp.latex(right) )

    def render_p(self, p, args):
        '''確率密度関数の出力用整形'''
        psymbol = sp.Function('p')
        return self.render_eqn( psymbol(*args), p )

    def render_prm(self, prm, vals):
        '''パラメータの出力用整形'''
        valssymbol = f"{vals}"
        return self.render_eqn( prm, valssymbol )

    def summary(self, simplify=False):
        '''概要の出力'''
        p    = self.pdf['p']
        args = self.pdf['x'] + self.pdf['prm']    #確率密度関数の全引数
        if simplify:
            p = sp.simplify(p)

        print(f"===== {self.pdf['name']} =====")
        display(self.render_p(p, args))

    def plot(self, prm, dom, reso=80, labs=None):
        '''
        確率密度関数のプロット
        - prm ... パラメータ値
        - dom ... プロットする定義域
        '''
        dim     = len(self.pdf['x'])       #次元
        prmsym  = self.pdf['prm']          #パラメータ（数式処理）
        reso    = reso                     #独立変数の解像度

        # パラメータ値の表示
        display(self.render_prm(prmsym, prm))

        # 確率密度関数のプロット
        if dim == 1:
            fig, ax = plt.subplots(1,1,figsize=(4.8, 2.4))
            xs = np.linspace(*dom, reso)
            ps = self.p_np(xs, *prm) #確率密度
            ax.plot(xs, ps)
            if labs is None:
                labs=[r'$x$', r'$p(x)$']
            ax.set_xlabel(labs[0])
            ax.set_ylabel(labs[1])
        elif dim == 2:
            fig, ax = plt.subplots(1,1, figsize=(4, 4),
                            subplot_kw={'projection': '3d'})
            X, Y = np.meshgrid(
                np.linspace(*dom[0], reso),
                np.linspace(*dom[1], reso)
            )
            Z = self.p_np(X, Y, *prm)
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            if labs is None:
                labs=[r'$x_1$', r'$x_2$', r'$p(x_1,x_2)$']
            ax.set_xlabel(labs[0])
            ax.set_ylabel(labs[1])
            ax.set_zlabel(labs[2])
            ax.set_box_aspect(None, zoom=0.85) #Z軸ラベルが切れるので
        else:
            print("dim >= 3: Not implemented!")

        ax.grid()
        fig.tight_layout()
        plt.show()

        return fig, ax

class PDFs(PDFBase):
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

__all__ = ['PDFBase', 'PDFs']
