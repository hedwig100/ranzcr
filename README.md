# RANZCR 
RANZCR用のレポジトリ

# Goal 
画像コンペに慣れる <br>
tensorflowをちゃんと書けるようにする<br>

# Data
|name|explanation|
|:--:|:--:|
|StudyInstanceUID|画像ごとのユニークなID|
|ETT-Abnormal|気管チューブの場所に異常あり|
|ETT-Borderline|気管チューブの場所が異常があるかないかギリギリのところ|
|ETT-Normal|気管チューブが通常の位置にある|
|NGT-Abnormal|鼻から入れる栄養を胃に流し込むチューブの場所に異常あり|
|NGT-Borderline|鼻から入れる栄養を胃に流し込むチューブの場所が異常があるかないかギリギリのところ|
|NGT-Incompletely Imaged|鼻から入れる栄養を胃に流し込むチューブの異常について画像からはわからない|
|NGT-Normal|鼻から入れる栄養を胃に流しこむチューブが通常の位置にある|
|CVC-Abnormal|中心静脈カルーテルの場所に異常あり|
|CVC-Borderline|中心静脈カルーテルの場所が異常があるかないかギリギリのところ|
|CVC-Normal|中心静脈カルーテルが通常の位置にある|
|Swan Ganz Catheter Present|肺動脈カルーテルをつけているかどうか|
|PatientID|患者ごとのID|
|Images|患者のX線画像|

- クラスが不均衡(Normal系が多い) <br>

# Metrics
11このクラスそれぞれに属する確率を出力して, クラスごとにAUCをとってから11このAUCを平均する<br>

# Notebooks
***[最初に提出してみたNotebook](https://www.kaggle.com/xhlulu/ranzcr-efficientnet-gpu-starter-train-submit)*** <br>
- [Efficientnetb2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)をimagenetで学習させたものを転移学習させる<br>
- dropconnectというdropoutに似た過学習を防ぐ手法 <br>
- GlobalAveragePooling2Dはチャンネルごとの平均をとって重みを減らす手法 <br> 
- auto-select-accelerator()は分散学習するための関数 <br>
- 画像系のコンペではdecoderを利用して逐次的にデータを読み込むのが(メモリ的に)普通? <br>
- tf.Datasetの使い方は[tensorflowの公式の説明](https://www.tensorflow.org/tutorials/load_data/images?hl=ja)がわかりやすかった <br>
- datasetのload,整形 -> キャッシュしておく -> augmentaion -> repeat -> shuffle <br>
- augmentation(左右, 上下反転)をする <br>
- prefetchしておくことで訓練中にバッチを取得できて早くなる <br> 


***[edaで参考にした1](https://www.kaggle.com/parthdhameliya77/ranzcr-clip-eda-class-imbalance-patient-overlap)*** <br>
- classの不均衡性を扱っていた<br>

***[edaで参考にした2](https://www.kaggle.com/foolofatook/ranzcr-clip-one-stop-for-all-eda-needs)*** <br>
- annotationはすべてのデータについているわけではない? <br>
- 自分で見ると何が違うか全くわからない<br>
- CVCとNGTはラベルのオーバーラップがあるがそれは一人の患者さんが複数のカルーテルを持っていて, どのカルーテルかによって状態が異なるからでラベル付をミスったとかではない <br>

# Log
***20200226*** <br> 
- コンペに参加する.<br>
- コンペを把握する. <br>
- とりあえず他のNotebookを参考にしてsubmitまでしてみた. <br> 

***20200227*** <br>
- EDAのnotebookを書いた. <br>
- 昨日submitしたコードを読んで何をしているか理解した. <br> 
- クラスが不均衡なのでweighted lossを使うことを検討する <br>
- 学習率, augmentaionなどもっと工夫できそう <br> 
- EfficientNetも他のものを使ってみても良さそう <br> 
 



