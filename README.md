# RidgeletCalculus for NeuralNetwork
## 実装について
　ニューラルネットワークの初歩的な説明は割愛する。  
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
　実際のニューラルネットワークは中間層のノード数が有限になるが、この積分表現の離散化をパラメータの初期値に用いると、最初から答えの近くから始めることになる。なので通常の正規分布からのサンプリングで初期値を決めるよりも早く収束させることができ、勾配法でも悪い局所解にハマるリスクがかなり小さい。
　再構成公式が成り立つための「ニューラルネットで近似したい関数」と「活性化関数」の条件や、リッジレット関数の計算方法は全て関数解析学の言葉で書ける。具体的な条件や離散化の評価方法については[1]の4,5章を参照のこと。  

## オラクルサンプリングとリッジレット変換の近似
　ここでは、オラクルサンプリングという手法での入力層→中間層への重みづけパラメータ初期値の決定方法と、そこからリッジレット変換での中間層→出力層の重みづけの近似計算について解説する。


### オラクルサンプリング
　ルベーグ測度に対してつぎのようなラドンニコディム導関数を持つ<img src="https://latex.codecogs.com/gif.latex?\mathbb{Y}^{d+1}" />上の確率測度を「オラクル測度」と呼び、その確率測度の下での乱数生成を「オラクルサンプリング」と呼ぶ。  
 <img src="https://latex.codecogs.com/gif.latex?\mu(a,b):=\frac{|(\mathcal{R}_\psi&space;f)(a,b)|}{\int_{\mathbb{Y}^{d+1}}|(\mathcal{R}_\psi&space;f)(a,b)|dadb}" />

　入力層が低次元なら棄却法で直接これをサンプリングすればいいが、高次元になると値の信頼性がかなり落ちてくるため、単純な測度で近似してサンプリングを行う。詳細なアルゴリズムは[1]の7章を参照。  

### リッジレット変換の近似
　リッジレット変換は、次のようにモンテカルロ積分で近似する。  
 <img src="https://latex.codecogs.com/gif.latex?(\mathcal{R}_\psi&space;f)(a,b):=\frac{1}{nZ}\Sigma^n_{i=1}y_i\psi(a\cdot&space;x-b)" />  
ただし<img src="https://latex.codecogs.com/gif.latex?Z:=K_{\psi,\eta}\int_{\mathbb{Y}^{d+1}}|(\mathcal{R}_\psi&space;f)(a,b)|dadb" />は正規化定数と核の積で、その値を具体的に計算するのは困難である。そのため、Zは1.0とした。（[1]ではリッジレット変換せずに線形回帰で求めていたが、ただでさえ過学習しやすい手法であるため、比率だけは保持してやや大雑把に動かしたい）

## 研究との差分及び環境設定
 ただ[1]の内容をなぞるだけでは味気ないので、独自の工夫も加えた。  
 ・学習ではドロップアウトを採用  
 ・中間層→出力層では、[1]では線形回帰でパラメータを求めていたが、かなり過学習しやすいと思われる（実際論文の結果には過学習の気が見られ、著者もそれは認めている）ため、リッジレット変換を定数倍は放置したうえで直接近似した。  
 中間層のノード数は10000、学習回数は1000回、ドロップアウト率は50%とした。python3.6及びtensorflow1.4.0を用いている。
 

## 今後予定している改善点
 リッジレット変換の定数倍を今回はかなり適当に扱った。学習する際に一直線にいい解に向かってくれるのである程度は問題ないが、やはりここは線形回帰などでちゃんとフィッティングしたほうがいいと思われる。  
 
## バージョン
  Ver3.0 2018/2/22 一通りのコードを完成。今後はより活用できるように上乗せをしていく。

## 参考文献
[1]園田翔,深層ニューラルネットワークの積分表現理論(2017)
