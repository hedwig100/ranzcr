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
***[1. Notebook](https://www.kaggle.com/xhlulu/ranzcr-efficientnet-gpu-starter-train-submit)*** <br>
- 最初に提出してみた
- [Efficientnetb2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)をimagenetで学習させたものを転移学習させる<br>
- dropconnectというdropoutに似た過学習を防ぐ手法 <br>
- GlobalAveragePooling2Dはチャンネルごとの平均をとって重みを減らす手法 <br> 
- auto-select-accelerator()は分散学習するための関数 <br>
- 画像系のコンペではdecoderを利用して逐次的にデータを読み込むのが(メモリ的に)普通? <br>
- tf.Datasetの使い方は[tensorflowの公式の説明](https://www.tensorflow.org/tutorials/load_data/images?hl=ja)がわかりやすかった <br>
- datasetのload,整形 -> キャッシュしておく -> augmentaion -> repeat -> shuffle <br>
- augmentation(左右, 上下反転)をする <br>
- prefetchしておくことで訓練中にバッチを取得できて早くなる <br> 

***[2. Notebook](https://www.kaggle.com/parthdhameliya77/ranzcr-clip-eda-class-imbalance-patient-overlap)*** <br>
- edaで参考にした1
- classの不均衡性を扱っていた<br>

***[3. Notebook](https://www.kaggle.com/foolofatook/ranzcr-clip-one-stop-for-all-eda-needs)*** <br>
- edaで参考にした2
- annotationはすべてのデータについているわけではない? <br>
- 自分で見ると何が違うか全くわからない<br>
- CVCとNGTはラベルのオーバーラップがあるがそれは一人の患者さんが複数のカルーテルを持っていて, どのカルーテルかによって状態が異なるからでラベル付をミスったとかではない <br>

***[4. Notebook](https://www.kaggle.com/yasufuminakama/ranzcr-resnext50-32x4d-starter-training)***
- 学習率, エポック, スケジューラなどの設定を参考にした. 

***[5. Notebook](https://www.kaggle.com/underwearfitting/how-to-properly-split-folds)***
- Validationの参考にした

# Papers 
|No.|Status|name|detail|
|:--:|:--:|:--:|:--:|
|01|todo|[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)|Efficientnetの元論文|
|02|todo|[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)|ResNeXtの元論文|

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
- EfficientNetは[この記事](https://qiita.com/omiita/items/83643f78baabfa210ab1)をみて理解した. (20200208追記) <br>

***20200228*** <br> 
- Validationの仕方を考えた. 同じ患者さんが含まれていることとクラスが不均衡であることを考えると, *StratifiedGroupKFold*が良いと思った. <br>
5Foldで分けるものを他のNoteBookを参考に作成した. TfRecordについても理解してtfrecordを用いてデータを前処理しておきたい. <br>
- class_weightを多次元の出力のそれぞれに適用する方法を調べる <br>
- RexNeXt?というモデルがよく使われているっぽい. <br>
- [RexNeXtの記事](https://medium.com/lsc-psd/resnext%E3%81%AE%E8%AB%96%E6%96%87%E3%82%92%E5%88%9D%E5%BF%83%E8%80%85%E5%90%91%E3%81%91%E3%81%AB%E5%9B%B3%E8%A7%A3%E3%81%99%E3%82%8B-a93a1b8138e5), 要するにResNetのshortcutしない層が複数存在しているようなもの? 
- `keras.metric.AUC(multi_label=True)`は各クラスごとでAUCを計算してから平均をとる(このコンペの指標と同じ)ことをやってくれている. [src](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC) <br>
- 学習に時間がかかるのでepochを大きくできないので学習率をAnnealingした. `lr`は`5e-3`からスタートして`max(lr*np.exp(-0.1),1e-6)`のように変化させた. 
これは[このNoteBook](https://www.kaggle.com/yasufuminakama/ranzcr-resnext50-32x4d-starter-training)にならった. <br>
 
***20200301*** <br> 
- 昨日学習させたモデルの学習率がうまく調節できていないせいか, モデルの精度が他の人が上げているNotebookに比べて明らかに悪い
- エポックが少ない? -> エポックを15回にして1fold分だけ学習させてみた. -> それでも`val_auc=0.7`くらいで公開kernelの`val_auc=0.9`とかにはいかなかった. 
- augmentationをalbumentationでやろうとしたら, `tf.data.Dataset`の扱いがわからなくてバグらせまくったので`tf.data.Dataset`について基本的なところを確認した. 
- albumentationじゃなくて, tensorflowのものを使ってもいいかもしれない. 
- KaggleKernel上でDensenet50で学習できるようにした. 明日回す. 