## クラスタリング
K-平均法(k-means)と混合正規分布(Gaussian Mixture Model, GMM)を使い、身長データをクラスタリングしてみる。

ある学校で5名の身長xを測った。
　身長(m) X = [1.50, 1.55, 1.60, 1.70, 1.80]　
この身長データを男女の2つにクラスタリングする。
本来2つのクラスタに分割できたとしても、その分割理由が性別によるものか、年齢によるものか、出身国によるものか、アルゴリズムは答えてくれない。しかし今回はアルゴリズムの理解の為、性別でクラスタに分かれると仮定して話を進める。





## k-平均法(k-means法)によるクラスタリング

データ$x_n$(今回はn=1,2, ... ,5）をあらかじめ指定したK個のクラスタに分ける。今回はクラスタの番号である混合要素kをk= 1[女], 2[男] と想定して話を進めるので、K=2個である。

クラスタの中心を$\mu_k$とする。

各データ$x_n$に対し、2値指示変数 $r_{n k} \in \{0,1 \}$ を定める。これは1-of-K表現をとり、データ点$x_n$がk番目のクラスタに含まれるとき1、それ以外は0になる。例えば$x_5$の人が男とすると、$r_{n,k}$ は $r_{5,1}=0$ であり、$r_{5,2}=1$ である。

目的関数Jを定義する。一般的には次のようになる。
$$ \hspace{50pt} \displaystyle J = \sum_{n=1}^N \sum_{k=1}^K r_{n k} ||\bf{x_n} - \bf{\mu_k}||^2$$
今回のデータxは1次元空間の確率変数である為、次の定義となる。
$$ \hspace{50pt} \displaystyle J = \sum_{n=1}^5 \sum_{k=1}^2 r_{n k} (x_n - \mu_k)^2 $$

各データ点からそれが割り当てられた$\mu_k$までの二乗距離の総和を表している。我々の目的は、Jを最小化する$r_{nk}$と$\mu_k$を求める事である。これは次のStep1と2を繰り返す手続きで実現する。

#### 初期化:
$\mu_k$にランダムな値を代入する。

#### Step1: 
$\mu_k$を固定化して$J$を最小化するよう$r_{n k}$を決定する。
各$x_n$について$r_{nk}=1$とした時に$(x_n - \mu_k)^2 $が最小になるようなkの値に対して$r_{nk}=1$とおく。
${\begin{align}
\hspace{20pt} r_{nk} = \begin{cases} 1 \ (k = arg\min_j (x_n - \mu_j)^2 \\\
                       0 \ (other wise)
\end{cases}
\end{align}}
$

#### Step2:
$r_{n k}$を固定化して$J$を最小化するよう$\mu_k$の最適化を行う。
Jを$\mu_k$に関する偏微分を行い、0と置く。
$\begin{align}\hspace{20pt} \displaystyle 
  \frac{\partial J}{\partial \mu_k} 
&=\frac{ \partial} {\partial \mu_k} (\sum_{n} \sum_{k} r_{nk} (x_n - \mu_k)^2 \;)\\\
&=\sum_{n} r_{nk} \frac{\partial}{\partial \mu_k} (x_n - \mu_k)^2 \\\
&=-2\sum_{n} r_{nk} (x_n - \mu_k) \\\
\sum_{n} r_{nk} (x_n - \mu_k) &=0\\\
 \mu_k &= \frac{\sum_{n} r_{nk} x_n}{\sum_n r_{nk}}
\end{align}$



### Pythonサンプル

```python:k-means
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

X = np.array([1.50, 1.55, 1.60, 1.70, 1.80])#身長(m)
N = X.shape[0]
K = 2 #クラスター数（男女で2）
X_range0 = [1.4, 2] #グラフのX軸の表示範囲
X_color = ['red', 'blue'] # クラスターの表現色
Mu = np.array([1.5, 1.6])  # 分布の平均μ
R = np.c_[np.ones( (N, 1), dtype=int)   ,np.zeros((N, 1), dtype=int)]  # R [[1,0][1,0]・・]で初期化

# データ図示関数
def show_prm(x, r, mu, x_color):
    plt.grid(True)
    #正規分布の描写用 解像度
    X_arange = np.linspace(X_range0[0], X_range0[1], 40 )

    for k in range(K):
        # データ分布の描写
        plt_x = x[r[:, k] == 1]
        plt_y = np.zeros((len(plt_x)))
        plt.plot(plt_x, plt_y, marker='o', markerfacecolor=x_color[k], markeredgecolor='black', markersize=12, alpha=0.5, linestyle='none')

        # データの平均 μ(mu)を星マークで描写
        plt.plot(mu[k], 0, marker='*', markerfacecolor=x_color[k], markersize=10, markeredgecolor='black')


# Step1 r決定 
def step1_kmeans(x0, mu):
    N = len(x0)
    r = np.zeros((N, K))
    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = (x0[n] - mu[k])**2 
        r[n, np.argmin(wk)] = 1
    return r


# Step2 Mu決定
def step2_kmeans(x0, r):
    mu = np.zeros(K)
    for k in range(K):
        mu[k] = np.sum(r[:, k] * x0) / np.sum(r[:, k])
    return mu


# main
plt.figure(2, figsize=(10, 10) ,facecolor='gray')
plt.subplot(3, 3, 1)
plt.tick_params(labelleft=False)
plt.title('No.0 Initial')
show_prm(X, R, Mu, X_color)

max_it = 5 # 繰り返しの回数
for it in range(1, max_it):
    step = 'step1 (update R)'
    R = step1_kmeans(X, Mu)
    plt.subplot(3, 3, it*2 )
    plt.title('No.{0:d} '.format(it + 1) + step)
    plt.tick_params(labelleft=False)
    show_prm(X, R, Mu, X_color)
    
    step = 'step2 (update Mu)'
    Mu= step2_kmeans(X, R)
    plt.subplot(3, 3, it*2 +1)
    plt.title('No.{0:d} '.format(it + 1) + step)
    plt.tick_params(labelleft=False)
    show_prm(X, R, Mu, X_color)

plt.show()
```



丸印が身長データを、星印が各クラスタ（男と女）の中心を表している。計算式では$k \in \\{1(女),2(男)\\}$と考えて説明していたが、pythonの各種配列型はインデックスが0から始まるので$k \in \\{0,1 \\}$となっているので注意してほしい。 今回は赤が女で青が男になるように、恣意的μの初期値で赤が青より小さくしてある。
結果の図のように、今回の例は単純なので、数回のステップで収束してデータが女と男に分かれている。


![クラスタk-means.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/244452/be846ab7-0558-47ab-6f6f-fdc6e074d139.png)







## 混合正規分布(混合ガウス分布)(Gaussian Mixture Model, GMM)によるクラスタリング
データ$x_n$(今回はn=1,2, ... ,5）をあらかじめ指定したK個のクラスタに分ける。今回はクラスタの番号である混合要素kをk= 1[女], 2[男] と想定して話を進めるので、K=2個である。

クラスタの中心を$\mu_k$とする。

データに確率モデルを当てはめて，各データがどのクラスタに属するかを確率的に決めよう。確率モデルとして複数の正規分布を用いたものを 混合正規分布モデルと呼ふ。パラメータを$\boldsymbol \theta = (\boldsymbol \mu , \boldsymbol\Sigma , \boldsymbol\pi)$ として混合正規分布$p(\boldsymbol x|\boldsymbol \theta)$はK個のガウス分布の線形重ね合わせで次のように表される。
$\hspace{20pt} \displaystyle p(\boldsymbol x| \boldsymbol \theta) = \sum_{k=1}^K \pi_k N(\boldsymbol x|\boldsymbol \mu_k, \boldsymbol \Sigma_k)$
$\pi_k$ は k番目の確率モデルの重みを表すパラメータで混合係数と呼び、次の条件を満たす。
$\hspace{20pt} \ 0 \leq \pi_k \leq 1, \ \sum_{k = 1}^K \pi_k = 1$

k 番目のクラスタの d 次元正規分布関数を以下のように表す。
$\hspace{20pt} {N(\boldsymbol x \ | \ \boldsymbol \mu_k, \boldsymbol \Sigma_k) 
 =\frac{1}{(2\pi)^{d/2} \ |\boldsymbol \Sigma_k|^{1/2}}
  \exp\bigl(-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu_k)^T \boldsymbol \Sigma_k^{-1}(\boldsymbol x - \boldsymbol \mu_k)\bigr)}$

上記式は$\boldsymbol x$が2次元以上である場合にも対応した式だが、今回の身長データは1次元なので次のように表す。
$\hspace{20pt} {N(x \ | \ \mu_k, \ \sigma_k^2) 
 =\frac{1}{(2\pi\sigma_k^2)^{1/2}}
  \exp \big(-\frac{1}{2\sigma_k^2}(x - \mu_k)^2 \big)}$

#### 単純に最尤推定するのは困難
単純に考えると、混合正規分布$p(x|\theta)$の対数尤度をパラメータ$ \theta = (\mu , \sigma^2 , \pi)$ で偏微分して、結果を0と置いて解く、いわゆる最尤推定法で推定値 $\hat \theta $ を得れば、クラスタリングは完了である。しかしこの方法では次の式のように問題が発生する。
$\begin{align}\hspace{20pt}
\hat \theta &= arg\max_\theta \sum_{n=1}^N \ln p(x_n | \theta) \\\
           &= arg\max_\theta \sum_{n=1}^N \ln \sum_{k=1}^K \pi_k N(x_n|\mu_k, \sigma_k^2)
\end{align}$
lnの後ろに正規分布関数があればlnとexpが相殺して計算が単純化できるが、今回はlnの後ろにΣがあるためにlnが直接に正規分布関数に掛らない。これでは計算が困難になる。
そこで潜在変数zを導入し、EMアルゴリズムで繰り返し計算を行う。以下はその方法を説明している。

#### 潜在変数zの導入
潜在変数（隠れ変数,latent variable）zを定義する。
各データ$x_n$に対し、2値指示変数 $z_{n k}$ を定める。データ点$x_n$がk番目のクラスタに含まれるとき1、それ以外は0になる。つまり次の条件を満たす。
$\hspace{20pt} z_{n k} \in \{ 0,1 \}$ かつ $\sum_{k = 1}^K z_{n k} = 1$
例えば$x_5$の人が男とすると、$z_{n,k}$ は $z_{5,1}=0$ であり、$z_{5,2}=1$ である。
$z_{n k}$ はk-平均法の$r_{n k}$と同じ構造で同じ意味の値を保持するが、潜在的に使われる点が異なる。
またK次元の2値確率変数ベクトル$\boldsymbol z_n$は1-of-K表現をとるように定める。今回はカテゴリ数は2個なので次のようになる。
$\hspace{20pt} \boldsymbol z_n = (z_{n 1},z_{n 2})^T $
$\boldsymbol z_n$の分布は混合係数$\pi_k$によって定まる。
$\hspace{20pt} p(z_{n k} = 1) = \pi_k $
　　※ちなみに下記プログラムでは$\pi_1$(女の確率)=0.597, $\pi_2$(男の確率)=0.403 に収束した。
$p(\boldsymbol z_n)$はカテゴリカル分布(categorical distribution)で表される。
$\begin{align}\hspace{20pt} \displaystyle
 p(\boldsymbol z_n) 
 &= Cat(\boldsymbol z_n | \boldsymbol \pi)  \\\
 &= \prod_{k = 1}^K \pi_k^{z_{n k}}  \\\
\end{align}$

観測データの潜在変数による条件付き分布 は次のようになる。
$\hspace{20pt} \displaystyle p(x_n|z_{n k} = 1) = N(x_n|\mu_k, \sigma_k^2)^1= N(x_n|\mu_k, \sigma_k^2)$
$\hspace{20pt} \displaystyle p(x_n | \boldsymbol z_n) = \prod_{k = 1}^K N(x_n | \mu_k, \sigma_k^2)^{z_{kn}}$

#### 潜在変数の必要性
潜在変数zが分かれば、先の$\ln p(x_n | \theta)$の最尤推定を行う代わりに $\ln p(x_n,z_n | \theta) $ の同時確率分布の最尤推定でパラメータを求める事ができそうだ。ｚが分かるという事はｘとｚの組が観測されているという状態であり、これを完全データと呼ぶ。最大化したい完全データ対数尤度を L と置いて計算してみよう。
■完全データ対数尤度関数
$\begin{align}\hspace{20pt}
 L
 &= \ln \prod_{n=1}^N p(x_n, \boldsymbol z_n | \theta) \\\
 &= \sum_{n=1}^N \ln p(x_n, \boldsymbol z_n | \theta) \\\
 &= \sum_{n=1}^N \ln p(x_n | \boldsymbol z_n, \theta) p(\boldsymbol z_n | \theta) \\\
 &= \sum_{n=1}^N \ln \prod_{k = 1}^K N(x_n | \mu_k, \sigma_k^2)^{z_{kn}}  \prod_{k = 1}^K \pi_k^{z_{n k}} \\\
 &= \sum_{n=1}^N \sum_{k = 1}^K \ln N(x_n | \mu_k, \sigma_k^2)^{z_{kn}} \pi_k^{z_{n k}} \\\
 &= \sum_{n=1}^N \sum_{k = 1}^K z_{n k} (\ln N(x_n | \mu_k, \sigma_k^2) +  \ln \pi_k) \\\
 &= \sum_{n=1}^N \sum_{k = 1}^K z_{n k} (\ln ( \frac{1}{(2\pi\sigma_k^2)^{1/2}} \exp \big(-\frac{(x-\mu)^2} {2\sigma_k^2} \big)) +  \ln \pi_k) \\\
 &= \sum_{n=1}^N \sum_{k=1}^K z_{n k} (-\frac{1}{2}\ln (2\pi)  - \frac{1}{2} \ln \sigma_k^{2} - \frac{1}{2 \sigma_k^2}(x_n - \mu_k)^2 ＋\ln \pi_k)
\end{align}$

たしかに今回はlnが直接に正規分布関数に掛っているので、lnとexpが相殺して計算が単純化できる事がわかった。しかし $z_{n k}$ は式も値も不明なままである。これではパラメータ$ \theta = (\mu , \sigma^2 , \pi)$の最尤推定はできない。
結局 $z_{n k}$ の正体は分からず終いではあるが、パラメータを適当に決めてしまえば、$z_{n k}$ の適当な期待値くらいは計算できる。
そこで、大きな誤差を承知で $z_{n k}$ の期待値 $E[z_{n k}]$ を算出して(①Expectation Step)、これを $z_{n k}$ の代替にしてパラメータを最尤推定しよう(②Maximization Step)。
最初に与えるパラメータは適当な値になるので、期待値 $E[z_{n k}]$の精度はそれこそ期待できないが、①E-Stepと②M-Stepを繰り返せば、期待値 $E[z_{n k}]$ と $z_{n k}$ の誤差は小さくなり、真の $z_{n k}$ にたどり着けるだろう。

p(x)を知りたいが、どのような関数で表現してよいのか不明である。よって同時分布　p(x,z)=p(x|z)p(z)　を周辺化する事でp(x)を表現する。周辺化では、zの全ての場合(今回は$z_n$=[1,0],[0,1])について和を取る。
$\begin{align}\hspace{20pt} \displaystyle
 p(x) 
 &= \sum_{z} p(x, z) \\\
 &= \sum_{z} p(z) p(x|z) \\\
 &= \sum_{z} (\prod_{k = 1}^K \pi_k^{z_{n k}}) \; (\prod_{k = 1}^K N(x| \mu_k, \sigma_k^2)^{z_k}) \\\
 &= \sum_{z}  \pi_1^{z_{n 1}}\pi_2^{z_{n 2}}  N(x| \mu_1, \sigma_1^2)^{z_1} N(x| \mu_2, \sigma_2^2)^{z_2}\\\
 &=\pi_1 N(x | \mu_1, \sigma_1^2)^{z_1} +\pi_2 N(x| \mu_2, \sigma_2^2)^{z_2}\\\
 &= \sum_{k=1}^2 \pi_k N(x|\mu_k, \sigma_k^2)
\end{align}$
これは最初に記載した混合正規分布と同じ形になっている。遠回りして p(x) を再度導出したように見えるが、EMアルゴリズムで潜在変数zを使う為に必要な計算である。次のE-Stepで潜在変数zを使用する。

#### ①Expectation Step:
■$z_{n k}$ の期待値 ($\gamma(z_{nk})$)
$z_{n k}$ の期待値(Expectation )を求める。初回のE-Stepでは、パラメータ$(\mu , \sigma^2 , \pi)$には適当な初期値を与えて期待値を算出する事になる。2回目以降のE-Stepでは、M-Stepの最尤推定で改善したパラメータを与えて期待値を算出する。
数式を見やすくする為に $\gamma(z_{nk}) \equiv E[z_{nk}] $ として定義する。

$\begin{align}\hspace{20pt} \displaystyle
\gamma(z_{nk}) \equiv E_{z_{nk}} [z_{nk}]
 &= \sum_{z_{nk} = \\{ 0, 1 \\}} z_{nk} \ p(z_{nk} | x_n, \pi_k, \mu_k, \sigma_k^2) \\\
 &= 0 \times p(z_{nk} = 0 | x_n) + 1 \times p(z_{nk} = 1 | x_n)\\\
 & = \frac{p(z_{nk} = 1)p(x_n \ | \ z_{nk} = 1)}{\sum_{j = 1}^K p(z_{nj} = 1)p(x_n \ | \ z_{nj} = 1)} \;\;←ベイズの法則より\\\
 &= \frac{\pi_kN(x_n | \mu_k, \sigma_k^2)}{\sum_{j = 1}^K \pi_jN(x_n | \mu_j, \sigma_j^2)} \\\
\end{align}$



■負担率(responsibility)
$\gamma(z_{nk})$ はzの期待値だが、上記の計算途中からわかるように、$P(z_{nk}=1|x_n)$ でもある事から、「負担率」と呼ぶ。言い換えれば、混合要素kがxを負担する度合いを表している。
今回の例では、$\gamma(z_{5\ 2})$とは、混合要素k=2（男要素）がn=5の身長データ1.8mを負担する率である。潜在変数 z は観測できない値だが、今回の身長データの分布の中で1.8mは一番大きいので、$z_{5\ 1} = 0$ (k=1の所属ではない。つまり女所属ではない。）で$z_{5\ 2} = 1$ (k=2の所属である。つまり男所属である。）になっているはずだ。すると$\gamma(z_{5\ 1})$　は低い確率になり$\gamma(z_{5\ 2})$ は高い確率になる。

※参考までに下記プログラムのγの最終値を記す。行がx(身長）のn、列が混合要素のk(k=1は女要素,k=2は男要素）を表している。
　[[1.000 0.000]
　 [1.000 0.000]
　 [0.982 0.018]
　 [0.004 0.996]
　 [0.000 1.000]]

#### ②Maximization Step:
■潜在変数に関する対数尤度関数の期待値
対数尤度Lの中に潜在変数 $z_{nk}$ があるとパラメータを解けないので、e-stepで得た $z_{nk}$ の期待値 $\gamma(z_{nk})$ で代替する。この代替行為は、潜在変数zに関して対数尤度Lの期待値を取得してるともいえる。対数尤度Lの期待値関数を Q関数を呼ぶ。
$$\begin{align}\hspace{20pt}
 Q= E_z[L] 
 &= \sum_{n=1}^N \sum_{k=1}^K E_z[ z_{n k} ] (-\frac{1}{2}\ln (2\pi) - \frac{1}{2} \ln \sigma_k^{2} - 
\frac{1}{2 \sigma_k^2}(x_n - \mu_k)^2 +\ln \pi_k ) \\\
 &= \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) (-\frac{1}{2}\ln (2\pi) - \frac{1}{2} \ln \sigma_k^{2} - \frac{1}{2 \sigma_k^2}(x_n - \mu_k)^2 +\ln \pi_k ) \\\
\end{align}$$

■μを求める。
　対数尤度期待値関数Qをμで偏微分して0と置く。
$\begin{align}\hspace{20pt} \displaystyle
 \frac{\partial}{\partial \mu_k} Q
 = - \sum_{n=1}^N \gamma(z_{n k})  \frac{(x_n-\mu_k)}{\sigma^2_k} &= 0\\\
 \mu_k  &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n
 \;\;(ただし N_k = \sum_{n=1}^N \gamma(z_{nk}) )
\end{align}$

■$σ^2$ を求める。
　対数尤度期待値関数Qを$σ^2$で偏微分して0と置く。
$\begin{align}\hspace{20pt} \displaystyle
 \frac{\partial}{\partial \sigma_k^2} Q
 &= - \sum_{n=1}^N \gamma(z_{n k}) \big(\frac{1}{2\sigma_k^2}  +\frac{(x_n-\mu_k)^2}{2(\sigma_k^2)^2} \big)\\\
 &= - \sum_{n=1}^N \gamma(z_{n k}) \big(\frac{\sigma_k^2 + (x_n-\mu_k)^2}{2(\sigma_k^2)^2} \big) 
\end{align}$

$\begin{align}\hspace{20pt} \displaystyle
\sum_{n=1}^N \gamma(z_{n k}) \big(\frac{\sigma_k^2 + (x_n-\mu_k)^2}{2(\sigma_k^2)^2} \big) &=0\\\
\sum_{n=1}^N \gamma(z_{n k}) \ \sigma_k^2 &=\sum_{n=1}^N \gamma(z_{n k}) (x_n-\mu_k)^2\\\
 \sigma_k^2 &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk})(x_n - \mu_k)^2
\end{align}$

■π を求める。
　対数尤度期待値関数Qをπで偏微分して0と置く。
　このとき $\pi_k$ の総和が1である制約があるので、ラグランジュ未定係数法を用いる。
$\begin{align}\hspace{20pt} \displaystyle
 \frac{\partial}{\partial \pi_k}  \left(Q + \lambda(\sum_{k=1}^K \pi_k - 1) \right)
 &=\sum_{n=1}^N \frac{\gamma(z_{n k})}{\pi_k} + \lambda \\\
 \sum_{n=1}^N \frac{\gamma(z_{n k})}{\pi_k} + \lambda  &= 0\\\
 \pi_k \lambda  &= -\sum_{n=1}^N \gamma(z_{n k}) \\\
\end{align}$

　　kについて和をとると、πの合計は1となり式から消える。
$\begin{align}\hspace{20pt} \displaystyle
 \sum_{k=1}^K \pi_k \lambda  &= - \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{n k}) \\\
 \lambda  &= - \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{n k}) = - N\\\
\end{align}$
　　したがって
$\begin{align}\hspace{20pt} \displaystyle
  \sum_{n=1}^N \frac{\gamma(z_{n k})}{\pi_k} + N  &= 0\\\
  \pi_k &= \frac{1}{N} \sum_{n=1}^N \gamma(z_{n k}) =\frac{N_k}{N}
\end{align}$

#### EMアルゴリズムはE-StepとM-Stepの繰り返し

上記のM-Stepでパラメータ（μ,σ,π) の最尤解が計算できたが、どれも負担率 $γ$ に依存してる。よって事前にE-Stepで $γ$ を計算しておき、M-Stepでは $γ$ を定数と見立ててパラメータを求めている。一方E-Stepでは、（初回は適当だが）事前のM-Stepで得たパラメータを定数と見立てて $γ$ を計算する。このようにE-StepとM-stepを繰り返し、最適化を進めていくのがEMアルゴリズムである。



#### ＊＊＊＊＊＊参考情報：②Maximization Stepの別の方法＊＊＊＊＊＊
■最尤推定する対数尤度関数
　次の対数尤度関数を最大化(Maximization)するパラメータ $\theta = (\mu_k , \sigma_k^2 , \pi_k)$を求める.
$\hspace{20pt} \displaystyle 
 \hat \theta = arg\max_\theta \sum_{n=1}^N \ln p(x_n|\boldsymbol \pi,\boldsymbol \mu,\boldsymbol \sigma^2) 
= \sum_{n=1}^N \ln \{ \sum_{k=1}^K \pi_k N(x_n|\mu_k, \sigma_k^2) \}$

■μを求める。
$\hspace{20pt} 
\Biggl(参考：\frac{d}{dx}N(x | \mu, \sigma) 
 = \frac{d}{dx} \frac{1}{(2\pi\sigma^2)^{1/2}} \exp \big(-\frac{(x-\mu)^2} {2\sigma^2} \big)
 = -\frac{x - \mu}{\sqrt{2\pi}\sigma^3} \exp \big(- \frac{(x - \mu)^2}{2\sigma^2} \big) \Biggr)$

　対数尤度関数をμで偏微分して0と置く。
$\begin{align}\hspace{20pt} \displaystyle
 \frac{\partial}{\partial \mu_k} \sum_{n=1}^N \ln \sum_{k=1}^K \pi_k N(x_n|\mu_k, \sigma_k^2)
 &=\sum_{n=1}^N \frac{\pi_k N(x_n|\mu_k, \sigma_k^2)}  {\sum_{j=1}^K \pi_j N(x_n|\mu_j, \sigma_j^2)}
  \frac{(x_n-\mu_k)}{\sigma^2_k} \\\
 &=\sum_{n=1}^N \gamma(z_{n k})  \frac{(x_n-\mu_k)}{\sigma^2_k} 
\end{align}$

$\begin{align}\hspace{20pt} \displaystyle
 \sum_{n=1}^N \gamma(z_{n k})  \frac{(x_n-\mu_k)}{\sigma^2_k}  &=0\\\
 \mu_k  &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n
 \;\;(ただし N_k = \sum_{n=1}^N \gamma(z_{nk}) )
\end{align}$



■$σ^2$ を求める。
　対数尤度関数を$σ^2$で偏微分して0と置く。
$\begin{align}\hspace{20pt} \displaystyle
 \frac{\partial}{\partial \sigma_k^2} \sum_{n=1}^N \ln \sum_{k=1}^K \pi_k N(x_n|\mu_k, \sigma_k^2)
 &=\sum_{n=1}^N \frac{\pi_k }  {\sum_{j=1}^K \pi_j N(x_n|\mu_j, \sigma_j^2)} 
 \big(\frac{1}{2\sigma_k^2} N + \frac{(x_n-\mu_k)^2}{2(\sigma^2_k)^2} N \big)\\\
 &=\sum_{n=1}^N \gamma(z_{n k}) \big(\frac{1}{2\sigma_k^2}  +\frac{(x_n-\mu_k)^2}{2(\sigma_k^2)^2} \big)\\\
 &=\sum_{n=1}^N \gamma(z_{n k}) \big(\frac{\sigma_k^2 + (x_n-\mu_k)^2}{2(\sigma_k^2)^2} \big)
\end{align}$

$\begin{align}\hspace{20pt} \displaystyle
\sum_{n=1}^N \gamma(z_{n k}) \big(\frac{\sigma_k^2 + (x_n-\mu_k)^2}{2(\sigma_k^2)^2} \big) &=0\\\
\sum_{n=1}^N \gamma(z_{n k}) \ \sigma_k^2 &=\sum_{n=1}^N \gamma(z_{n k}) (x_n-\mu_k)^2\\\
 \sigma_k^2 &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk})(x_n - \mu_k)^2
\end{align}$

■π を求める。
　対数尤度関数をπで偏微分して0と置く。
　このときパラメータの総和が1である制約があるので、ラグランジュ未定係数法を用いる。

$\begin{align}\hspace{20pt} \displaystyle
 \frac{\partial}{\partial \pi_k} \left( \sum_{n=1}^N \ln \sum_{k=1}^K \pi_k N(x_n|\mu_k, \sigma_k^2) \lambda(\sum_{k=1}^K \pi_k - 1) \right)
&=\sum_{n=1}^N \frac{N(x_n|\mu_k, \sigma_k^2)}  {\sum_{j=1}^K \pi_j N(x_n|\mu_j, \sigma_j^2)} + \lambda \\\
 &=\sum_{n=1}^N \frac{\gamma(z_{n k})}{\pi_k} + \lambda \\\
 \sum_{n=1}^N \frac{\gamma(z_{n k})}{\pi_k} + \lambda  &= 0\\\ \pi_k \lambda  &= -\sum_{n=1}^N \gamma(z_{n k}) \\\
\end{align}$

　　kについて和をとると、πの合計は1となり式から消える。
$\begin{align}\hspace{20pt} \displaystyle
 \sum_{k=1}^K \pi_k \lambda  &= - \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{n k}) \\\
 \lambda  &= - \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{n k}) = - N\\\
\end{align}$
　　したがって
$\begin{align}\hspace{20pt} \displaystyle
  \sum_{n=1}^N \frac{\gamma(z_{n k})}{\pi_k} + N  &= 0\\\
  \pi_k &= \frac{1}{N} \sum_{n=1}^N \gamma(z_{n k}) =\frac{N_k}{N}
\end{align}$

#### ＊＊＊＊＊＊参考情報：ここまで＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

$\begin{align}\hspace{20pt} \displaystyle
\end{align}$



#### pythonサンプル:
$\mu_k$に加えて$\sigma^2_k$にもランダムな値を代入するが、$\sigma^2_k$は大きい数値だと収束が遅いので0.05とした。$\pi_k$は合計値が1になる必要があるので0.5を初期値とする。安定した結果を求めるなら、まずk-means法でクラスタリングを実施し、その結果を混合正規分布の初期値に使うとよい。

```python:mixgauss
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

X = np.array([1.50, 1.55, 1.60, 1.70, 1.80]) # 身長(m)
N = X.shape[0]
K = 2 #クラスター数（男女で2）
X_range0 = [1.4, 2] #グラフのX軸の表示範囲
X_color = ['red', 'blue'] # クラスターの表現色
X_color_array=np.array([[0.99, 0, 0],[0.0, 0.0, 0.99] ]) # クラスターのベクトル表現色
Mu = np.array([1.5, 1.6])  # 分布の平均μ
Sigma2 = np.array([.05, .05])  # 分布の分散σ**2
Pi = np.array([0.5, 0.5])  # 混合比(男女比)π
Gamma = np.c_[np.ones( (N,1), dtype=int), np.zeros((N,1), dtype=int)]  # クラスター確率γ　[[1,0][1,0]・・]で初期化

# データ図示関数
def show_prm(x, gamma, mu, sigma2):
    plt.grid(True)
    # 正規分布の描写用 解像度
    X_arange = np.linspace(X_range0[0], X_range0[1], 40 )

    for n in range(N):
        # データ分布の描写
        color=gamma[n,0]*X_color_array[0]+gamma[n,1]*X_color_array[1]
        plt.plot(x[n], 0, 'o',  color=tuple(color), markeredgecolor='black',  markersize=12, alpha=0.7)

    for k in range(K):
        # 正規分布の確率密度関数にX,平均、標準偏差を代入して描写
        plt_norm_y= norm.pdf(X_arange,loc=mu[k],scale=sigma2[k]**0.5)
        plt.plot(X_arange, plt_norm_y, color=X_color[k], alpha=0.5)

        # データの平均 μ(mu)を星マークと縦線で描写
        plt.plot(mu[k], 0.2, marker='*', markerfacecolor=X_color[k], markersize=10, markeredgecolor='black')
        plt.vlines(mu[k], 0, plt_norm_y.max(), colors=X_color[k], linestyle=':', alpha=0.8)


# E Step  (gamma を更新) 
def e_step_mixgauss(x, pi, mu, sigma2):
    N, = x.shape
    K = len(pi)
    y = np.zeros((N, K))
    gamma = np.zeros((N, K))
    
    for k in range(K):
        y[:, k] = norm.pdf(x,loc=mu[k],scale=sigma2[k]**0.5)

    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = pi[k] * y[n, k]
        gamma[n, :] = wk / np.sum(wk)
    return gamma


# M step (Pi, Mu, Sigma2 を更新)
def m_step_mixgauss(x, gamma):
    N, K = gamma.shape
    # pi計算
    pi = np.sum(gamma, axis=0) / N
    # mu計算
    mu = np.zeros((K))
    for k in range(K):
        mu[k] = np.dot(gamma[:, k], x[:]) / np.sum(gamma[:, k])
    # sigma2計算
    sigma2 = np.zeros((K))
    for k in range(K):
        sigma2_numerator = 0
        for n in range(N):
            sigma2_numerator += gamma[n][k] * (x[n] - mu[k]) ** 2
        sigma2[k] = sigma2_numerator / np.sum(gamma[:,k])
    return pi, mu, sigma2

# main
plt.figure(1, figsize=(10, 16), facecolor='gray')

plt.subplot(4, 2, 1)
plt.title('No.0 Initial')
show_prm(X, Gamma, Mu, Sigma2)

show_no_list = list([1,10,30])
max_it = 31 # 繰り返しの回数
for it in range(1, max_it):
    Gamma          = e_step_mixgauss(X, Pi, Mu, Sigma2)
    if it in show_no_list:
        plt.subplot(4, 2, (show_no_list.index(it)+1)*2+1)
        plt.title('No.{0:d}'.format(it) + ' e_step (update Gamma)')
        show_prm(X, Gamma, Mu, Sigma2)

    Pi, Mu, Sigma2 = m_step_mixgauss(X, Gamma)
    if it in show_no_list:
        plt.subplot(4, 2, (show_no_list.index(it)+1)*2+2)
        plt.title('No.{0:d}'.format(it) + ' m_step (update Pi, Mu, Sig2)')
        show_prm(X, Gamma, Mu, Sigma2)

plt.show()
```
丸印は身長データだが、赤に近い色が女の比率が高く、青に近い色が男の比率が高い事を表している。e-stepとm-stepを数十回繰り返す事で収束している事がわかる。
![クラスタmix_gauss.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/244452/8b95044f-445c-9fb6-1090-7f75d5c369ed.png)



## 参考
パターン認識と機械学習(下) (シュプリンガー・ジャパン)
あたらしい機械学習の教科書（翔泳社)