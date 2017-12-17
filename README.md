# Neural-Network
## 実装について
　ニューラルネットワークの詳しい説明は割愛する。  
　今回用いるのは、浅いニューラルネットワークと呼ばれるもので、3層パーセプトロンである。  
　ライブラリは主にtensorflowとnumpyを用い、活性化関数はReLu、損失関数は2乗誤差という極めてオーソドックスなやり方である。  
　データはkaggleにある家の価格付け問題のものを用いた。適宜必要なさそうなデータは枝切りして、データの次元は落としてある。  

## 初期値設定
　ただし、初期値設定に関しては一工夫加える。  
　中間層-出力層については、後述のリッジレット変換の近似を用い、  
　入力層-中間層については、後述のオラクルサンプリングを用いる。  

## ニューラルネットワークの積分表現
　入力層から中間層への重みづけパラメータ<img src="https://latex.codecogs.com/gif.latex?(a,b)&space;a\in\mathbb{R}^d,b\in\mathbb{R}" />がとる空間を<img src="https://latex.codecogs.com/gif.latex?\mathbb{Y}^{d+1}(=\mathbb{R}^{d+1})" />と表記する。  
　関数<img src="https://latex.codecogs.com/gif.latex?f:\mathbb{R}^d\rightarrow\mathbb{C}" />の、リッジレット関数<img src="https://latex.codecogs.com/gif.latex?\psi:\mathbb{R}\rightarrow\mathbb{C}" />によるリッジレット変換は次のように定義される。  
 <img src="https://latex.codecogs.com/gif.latex?(\mathcal{R}_\psi&space;f)(a,b):=\int_{\mathbb{R}^d}f(x)\overline{\psi(a\cdot&space;x-b)}|a|dx" />

　次に、関数<img src="https://latex.codecogs.com/gif.latex?T:\mathbb{Y}^{d+1}\rightarrow\mathbb{C}" />の、活性化関数<img src="https://latex.codecogs.com/gif.latex?\eta:\mathbb{R}\rightarrow\mathbb{C}" />による双対リッジレット変換を次のように定義する。  
<img src="https://latex.codecogs.com/gif.latex?(\mathcal{R}^*_\eta&space;T)(x):=\int_{\mathbb{Y}^{d+1}}T(a.b)\eta(a\cdot&space;x-b)|a|^{-1}dadb" />

　適当な条件のもとで、次の「再構成公式」が成り立つ。  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\int_{\mathbb{Y}^{d+1}}(\mathcal{R}_\psi&space;f)(a,b)\eta(a\cdot&space;x-b)dadb" />

　これを「ニューラルネットワークの積分表現」といい、中間層のノード数を無限大に増やした場合のニューラルネットワークは、そのパラメータを解析的に決定できることを示している。  
　実際のニューラルネットワークは中間層のノード数が有限になるが、この積分表現の離散化をパラメータの初期値に用いると、最初から答えの近くから始めることになる。なので通常の正規分布からのサンプリングで初期値を決めるよりも圧倒的に早く収束させることができ、勾配法でも局所解にハマるリスクがかなり小さい。交差検証を行わなくてよくなる場合も多いようだ。  
　再構成公式が成り立つための「ニューラルネットで近似したい関数」と「活性化関数」の条件や、リッジレット関数の計算方法は全て関数解析学の言葉で書ける。具体的な条件や離散化の評価方法については[1]の4,5章を参照のこと。

## オラクルサンプリングとリッジレット変換の近似
　ここでは、オラクルサンプリングという手法での入力層→中間層への重みづけパラメータ初期値の決定方法と、そこからリッジレット変換での中間層→出力層の重みづけの近似計算について解説する。

　次のようなルベーグ測度に対するラドンニコディム導関数を持つ<img src="https://latex.codecogs.com/gif.latex?\mathbb{Y}^{d+1}" />上の確率測度を「オラクル測度」と呼び、その確率測度の下での乱数生成を「オラクルサンプリング」と呼ぶ。
<img src="https://latex.codecogs.com/gif.latex?\mu(a,b):=\frac{|(\mathcal{R}_\psi&space;f)(a,b)|}{\int_{\mathbb{Y}^{d+1}}|(\mathcal{R}_\psi&space;f)(a,b)|dadb}" />

入力層が低次元なら棄却法で直接これをサンプリングすればいいが、高次元になると値の信頼性がかなり落ちてくるため、単純な測度で近似してサンプリングを行う。詳細なアルゴリズムは[1]の7章を参照。

x_sとy_s:=f(x_s)が紐づけられたS個のデータがあったとする。リッジレット変換は次のように近似できる。
<img src="https://latex.codecogs.com/gif.latex?(\mathcal{R}_\psi&space;f)(a,b)\approx\Sigma^S_{s=1}y_s\psi(a\cdot&space;x_s-b)" />


## 注釈
　実際にリッジレット解析を用いて実装されたニューラルネットワークについては後日アップロード予定。
　現時点では通常の正規分布からのサンプリングで実装している。
 「リッジレット解析を使うとどうなるのか今すぐ見たい！」という方は[1]の7章後半を参照


## 参考文献
[1]園田翔,深層ニューラルネットワークの積分表現理論(2017)
