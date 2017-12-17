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
　関数<img src="https://latex.codecogs.com/gif.latex?f:\mathbb{R}^d\rightarrow\mathbb{C}" />の、関数<img src="https://latex.codecogs.com/gif.latex?\psi:\mathbb{R}\rightarrow\mathbb{C}" />によるリッジレット変換は次のように定義される。
 <img src="https://latex.codecogs.com/gif.latex?f:(\mathcal{R}_\psi&space;f)(a,b):=\int_{\mathbb{R}^d}f(x)\psi(a\cdot&space;x-b)|a|dx" />


## 参考文献
[1]園田翔,深層ニューラルネットワークの積分表現理論(2017)
