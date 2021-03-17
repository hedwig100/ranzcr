# RANZCR 
RANZCR用のレポジトリ

# Goal 
画像コンペに慣れる <br>
tensorflowをちゃんと書けるようにする<br>

# Rank 
||PublicLB|PrivateLB|
|:--:|:--:|:--:|
|score|0.965|0.967|
|rank|800|712|

# Solution 
## 1-st place 
[discussion1](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226576) 
[discussion2](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226633)
[code]()

### 概要
- 5-segmentation-models(imsize=1024~1536) + 67-classification-models(imsize=384,512)
- StratifiedGroupKFold 
### 感想
- めっちゃ画像サイズおおきいsegmentation(重要な部分のみ切り出す?) => 分類する

## 6-th place 
### 概要
- 
### 感想
-

## 7-th place 
[discussion](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226621) 

### 概要
- imsize=768のモデル. 
- UNet-CNNのArchitecture, UNetはsegmentation用のもの, segmentationはETT,NGT,CVS+SGC(大まかなクラスごと)のchannelで出力した. CV-LB-correlatioinがよりrobustになった. 
- ResNet200D,EfficientNetB7,NFNet-f1の大きなモデルを使った.
- ResNet200D,EfficientNetB7はUNetEncoderを用いて使えたけど, NFNetは難しかったので3StageTrainingした. 
- 外部データはpseudo-labelingしてクラスのバランスよくデータを増やした. 
- 外部データでのtrainingはteacher-student形式でtrainingした後に, 元のdatasetでfine-tuningした. 

### 感想

## 11-th place 
[discussion](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226557)

### 概要
- HighResolutionな画像を使うために, 一層目のCNNでimsize=2048の画像をimsize=1024の画像に縮小してから(downconv), 普通のCNN(ResNetとか?)に入れて学習させた. 
- annotationを使うためにsegmentationを用いた. segmentationとclassificationを同時に用いるのは(multi-task learning)成功しなかったが, まずsegmentationをするモデルを学習させてから, その後にclassificationするモデルを使うとよかった. segmentation時に正例に重みをつけないと動かなかった(classificationではなくて?). segmentationにはUNetというmodelを用いた. 
- unlabeledな外部データがあり, それを利用した. Pretrainはそれほど機能しなかった. Pseudo-trainingは効いた. 5foldで学習したあとに, unlabeled-dataの予測をして, 確信度が高い(確率0.5以上)データと合わせてもう一回trainingした. 
- hard agumentation

### 感想
- imsizeが大きい方が良いのはわかっていたけれど, 2048までいくとは思わなかった. 高解像度な画像を用いたいが, memory的に厳しい時にこれは役に立つかもしれない. 
- UNetを初めて知った. Segmentationに使われるモデルらしい. [参考](https://qiita.com/hiro871_/items/871c76bf65b76ebe1dd0)
- pseudo labelingは使うことも考えたけど, そもそものモデルの性能が悪すぎて全然labelingできなさそうだと思ったのでやめていた. discussionに書いてあったけどもしpretrainedする場合はデータの重複がないことを確認しないとLeakになってしまうのでその辺が難しそう. 


# 反省点
- TensorFlowを書けるようにするという目標もあったので仕方ないが, torchの方が学習済みモデルがたくさんあったので, torchを使うことをもっと前から検討すればよかった. 
- CVとLBが相関していたので出さなくてもいいかと思っていた時期があったが, submitしないと単純にモチベーションが下がるのできちんと提出はした方が良い. 
- 3stage-modelingなどannotationをうまく使うことができなかった. 学習がうまくいかなかった. 
- kaggle-notebook以外の環境(Google Colab)を使わなかった, 画像コンペだと計算資源も大事. 
- 最終日にsubmittion errorを出した. コンペの期限から逆算して余裕を持って終わるくらいの方がいい? 
- コードが汚かった. めちゃくちゃわかりやすくなくても良いけど, 後から見てわかるくらいにはしておくべき. 

# 学習したこと
- tf.dataset, tfrecordなどの扱い方
- tfでTPUを使うやり方
- torchの基本的な書き方. 
- 画像コンペではimsizeは想像より重要だった. 
- heavyなaugmentationをかけることはoverfittingを防ぐために大事だった. 
- 学習率, schedulerは結構重要だった. 
- 指標がaucの場合のアンサンブルの仕方. 
- teacher-student-modelでannotationを使う. 
- いろんなモデル(efficientnetB6,7,ResNet152)などを知ることができた. 

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

***[6. Notebook](https://www.kaggle.com/ragnar123/ranzcr-efficientnetb6-baseline)***
- いろいろ参考にした. <br>
- Efficientnet6 <br>
- TPU <br>
- tf.keras.imageを用いたAugmentation 参考リンク[tf.image](https://www.tensorflow.org/api_docs/python/tf/image) <br>
- TestTimeAugmentation <br>
- nb04 <br>

***[7. Notebook](https://www.kaggle.com/amritpal333/clahe-augmentation-ranzcr-comp)***
- CLAHEというdata-augmentationの方法, みた感じ画像がクリアになっている感じがする. 

***[8. Notebook](https://www.kaggle.com/tomohiroh/pytorch-starter)***
- Pytorchが何もわからなかったので, とりあえず書けるようになるには良さそうだと思った. 

***[9. Notebook](https://www.kaggle.com/underwearfitting/single-fold-training-of-resnet200d-lb0-965)***
- PytorchでResNet200Dを用いている, 過去の医療画像コンペの優勝解法を参考にしているっぽい? 
- めっちゃHighScoreKernelで, 憚られるけどCvとLBの相関があるので全然Shakeしなさそうだし, ResNet200Dを使ってみたいので. 
- 計算時間多すぎていまからやるのが向いてなさそう. 

***[10 Notebook](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step1)***
- 3-Stage Modellingの1-Stage
- 1-Stage annotated-imageを用いた学習
- 2-Stage 1-Stageで用いたモデルは固定して, 1-Stageのモデルと2-Stageのモデルの出力が近づき, かつBCEが小さくなるように2-Stageのモデルを学習させる
- 3-Stage 2-StageのモデルをlossをBCEのみにして, fine-tuningする
- annotated imageの一つの利用法
- annotationをどうやってつかうかが参考になった. 

***[11 Notebook](https://www.kaggle.com/underwearfitting/resnet200d-public-benchmark-2xtta-lb0-965)***
- 重みを使わせていただきました... 
- heavy augmentationはこんなにheavy. 
- [Training Code](https://www.kaggle.com/underwearfitting/single-fold-training-of-resnet200d-lb0-965)

# Papers 
|No.|Status|name|detail|
|:--:|:--:|:--:|:--:|
|01|todo|[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)|Efficientnetの元論文|
|02|todo|[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)|ResNeXtの元論文|

# Submission
括弧なしのスコアは1fold目のみのスコアで, 括弧がついている場合は, 括弧内が1fold目のスコアで括弧がついていないものが,5foldの
平均のスコア.

|No.|explanation|CV|LB|
|:--:|:--:|:--:|:--:|
|1|試しにsubしてみた|0.9424|0.907|
|2|1回目のがバグってたっぽくてもう一回出してみた|0.9359|0.916|
|3|StratifiedGroupKFoldのFold1回分(時間がないので)|0.87864|0.849|
|4|Resnet50のfold1|0.857|0.857|
|5|no03_5_0,Resnet50のfold1,augmentation追加|0.8873|0.864|
|6|nb03_7_0,Resnet50,|0.856|0.835|
|7|nb05_1_0,Resnet50,weightedloss,tta|0.843|0.860|
|8|nb05_5_0,Resnet152,weightedloss,tta|0.843|0.854| 
|9|nb05_6_0,Resnet152,lrを変えた|0.852|0.858|
|10|nb05_7_0,ResNet152,imsize=(512),CosineAnnealing|0.894|0.908|
|11|nb05_11_0,ResNet152,MultiHead,WeightedLoss|0.878|0.897|
|12|nb05_12_0,ResNet152,imsize=(768,768)|0.916|0.930|
|13|nb05_13_0,ReNet152,WeightedLoss|0.920|0.932| 
|14|nb05_13_0,ReNet152,WeightedLoss(バグの確認),cacheするとerrorになる|0.920|Error|
|15|nb05_13_0,ReNet152,WeightedLoss(バグの確認)|0.920|0.933|
|16|nb05_13_0,ReNet152,WeightedLoss,ttaにclaheを追加|0.920|0.906|
|17|nb05_13_7,ResNet152,claheかけて学習|0.913|0.929|
|18|ResNet152,imsize=512,5fold分全て用いる|0.919(0.911)|0.942|
|19|ResNet152,imsize=7685fold|0.930(0.922)|Timeout|
|20|ResNet152,imsize=768とResnet200Dの公開重みのアンサンブル|0.958(0.956)|0.965|

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
- KaggleKernel上でResnet50で学習できるようにした. 明日回す. 

***2020302*** <br> 
- Resnet50,でスケジューラを`lr=1e-3`からスタートして, `ReduveLROnPlateau()`に戻して実行してみることにした. (nb_03)
- 時間がないのでfold1つ分で実験することにした.
- efficientnetb7はメモリがおかしくなって自分の実装じゃできなかった. 
- 6Notebookをとりあえず理解していた. 
- Resnet50は割といい感じな気がする. 学習率をもうちょっと調整したい. 
- Resnt50 + Augmentation(saturate + hue + flip(上下左右))で明日動かしてみる. 
- TPUでeffcientnetb7を学習させてみたら, めちゃくちゃ早くて感動した.
- TPUのは本当にStratifiedGroupKFoldで学習させているのか
- ttaありのEffcientnetB6ではauc0.9くらいまでいった.
- 明日やるべきことはEfficientnetb6とdensenet50の学習をすること

***20200303*** <br> 
- Resnet50 + augmentationはaugmentationをしっかりやっていないものよりだいぶcvがよくなった. <br>
- EffcientnetB6は昨日動かしたのが, keras.application.effcientnetのものではなく, inferenceの際には使えなかったので, kerasで学習し直した. <br>
そうすると, 学習率が小さくなりすぎてほとんど学習していなかったので, もう一度学習させた. <br>
- そうしたらもっと悪くなった?? 
```
def WeightedBinaryCrossentropy(y_true,y_pred):
    y_true = tf.cast(y_true,tf.float32)
    pos_loss = -mul(mul(    y_true,tf.math.log(    y_pred)),pos_weight) 
    neg_loss = -mul(mul(1.0 - y_true,tf.math.log(1 - y_pred)),neg_weight) 
    loss = tf.add(pos_loss,neg_loss)
    return tf.math.reduce_mean(loss,0)
```
- 上のような重み付き損失関数で学習させてみた. (nb03,ResNet50) <br>
- EffcientNetB6の提出時にエラーが起こる. <br>
- dataload-augmentation-model-ttaのパイプラインを作る. (ranzcr_nb05)
- そのために, TPUとalubumentationを使う. 
- **validデータをtestからとる**というバグを埋め込んでいた. 気づかなかった. ラベルとの対応もおかしかった. nb03の結果が何も信頼できなくなった. 
- 明日 tfrecordをつくる and albumentation をもちいて学習させる. 

***20200304*** <br> 
- Tfrecordを作成した
- TTAがうまくいかない(shapeが合わなくてエラーになる)のでstepを切り上げにして調整した. 
- nb04_4 weightedloss + effcientnetB6 
- nb05_1 weightedloss + resnet50 + augmentation + ttaをだす. 
- 明日 albumentationをやる. 

***20200305*** <br> 
- albumentationでやろうとしたら, 異常な画像ができた. -> tf.imageのaugmentationで足りそうだしいいか. 
- 画像コンペはどんな方法が効くのかわからない
- lossとして, SigmoidFocaCrossEntropyというクラス不均衡に対応する損失関数を用いてみた. => そんなよくならなかった. 
- 画像サイズをもっと大きくした方が良い?(224,224) => (768,768) => 全然実行が終わらない. 
- train_metricとval_metricが離れすぎてて, なんかバグってる気がしないでもない. 

***20200306*** <br> 
- Imageサイズが医療画像ではかなり重要だというのをみて, 画像サイズを(512,512)にしてみた. (768,768)はTFRecordがつくれなかった. => outputファイルが大きすぎただけなので, 作り直せた. 
- ValidのAUCがずっと0.5で流石におかしかったのでバグを探す. (nb_06) => validデータは/255.0していなかった! => それでもおかしいnb06バグってる. 
- 上のをバグっていないであろうコードで動かすと, 0.89くらいのCrossValidationとなった. 
- CLAHEという画像をよりはっきりさせる前処理を試してみたい. 
- MultiHeadモデルは試してみたい. => nb07 
- 3-Stage Training は時間があれば試す. 

***20200307*** <br> 
- 512*512のResNet152はCV 0.89, LB 0.908となっていい感じだった. 画像サイズはやっぱり重要? 
- 同じことをEffcientNetB6に試したが, それほどいいcvじゃなかった. 
- MultiHeadModelも試したけど, 層のサイズの調整がまだできていなくて, それほど良いcvじゃなかった. まだ試す価値はあると思う. 
- ResNet200Dが使えないとお話にならない感があるので, Pytorchを書く決意をした. => nb08
- Notebook9をとりあえず理解する... => nb09 
- annotation使った方がいいってDiscuttionで言われているので(当たり前だけど), 使うことにする. => どうやって使うの? => 3-stage Model? 
- 明日, tensorflow + CLAHE と 3-Stage Modelをやってみる. 

***20200308*** <br> 
- WeightedBinaryCrossEntropyが最高だった. SigmoidFocalLossよりもよいCVだった. FocalLossとかより断然いい. 
- keras + clahe を実装した. keras.dataset + albumentation の実装とcastとかで手間取った. 
- tf.numpy_function + TPUは動かないらしい... そうすると, pytorchかGPUで動かさないとalbumentationが使えない. 
- annotationを使うのが厳しい. 

***20200309*** <br> 
- imsize = 768でResNet152が今のところ0.924のCVで一番いい. 
- 3-Stage TrainingのためにAnnotationつきのDatasetを作った. 
- 3-Stage nb10 => nb11 
- annotation付き + claheかけた　datasetを作った. 
- 1Stageは完了した. 1StageのAUCはAnnotationがついているので確かに0.99ぐらいまで行った. 
- 2Stageの実装を完了した. kerasのカスタム損失の実装に手間取った. 
- なぜかわからないけどval_lossがnanになる. なんでだろう. 
- 2Stage目の学習がうまくいかない- lossがnanになる. 
- claheかけただけのimsize=512画像を使って学習しようとしたけど, GPUだとメモリエラーになってしまった. nb13

***20200310*** <br> 
- 3Stage目をやったけどそれほど良い結果ではなかった(cv0.90)くらい. なんかおかしいのかな? 
- nb09にResNet200Dの実装をきちんと自分で行った. そんなに工夫がないので期待できないが,ResNet200Dの性能をみてみる. 
- Pytorch+GPUだとメモリがきつすぎて,batchsize=4とかになってめちゃくちゃ時間かかる. 
- 他の人がやってる実装に比べてそれほど違うわけではないのに, なんでこんなにScoreが違うんだ?

***20200311*** <br> 
- Pytorchの計算を回してたけど, 止まっていた. (学習時間の制限?)
- 2Stage目がバグってた. 確率の二乗誤差をとっても意味ない. sigmoidをかける前のものと二乗誤差をとって, sigmoidをかけた後にbceをとるべき. 
- バグを直したけれど, lossが小さくならなくて, auc=0.7くらいにしかならない. 

***20200312*** <br> 
- アンサンブルする. 
- Resnet152 + imsize 768
- EfficientNetB6 + imisize 512 + CLAHE 
- ResNet152 + imsize 512 + 3StageModeling
- MultiHeadModel? + imsize 512 (そんなに精度が良くない)
- lossはすべてWeightedBinaryCrossentropyにする. TPU Quotaがないので, 明日から5Fold分回す. 

***20200313*** <br> 
- アンサンブルする時に各モデルの確率pをp**0.5で足すとCVが良くなる? [discussion](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/211194)
- 学習を回していた. 
- submit時にalbumentation + kerasで実行するように実装した. 

***20200314*** <br> 
- imsize = 768の5fold分の学習も行った. 
- とてもではないけどメダル圏内に届きそうではないけど, 一応アンサンブルする. 
- Submit時間的に2モデル*5foldくらいが限界か? 
- 初Notebook投稿した. 
- 明日くらいにアンサンブルしたものを提出する, アンサンブルしてもそんなに良くなってないけど.

***20200315*** <br> 
- aucは普通にaverageするよりrank avarageした方がいいらしい. 
- aucの[https://www.kaggle.com/c/santander-customer-satisfaction/discussion/20783](アンサンブル)の方法の一つ, 何乗かしてから平均をとることで. aucが良くなることがある. 
- 2乗,4乗,0.5乗など試したけれどrank averageが一番よかった.
- pred1*pred**4みたいに掛け算の場合も重みをつけた方がよかった. 

***20200316*** <br> 
- 昨日からアンサンブルしたものが全部submittion scoring errorで焦っている. 
- 多分提出するcsvファイルにfile_pathっていう無駄なcolumnをつけていたことが問題だったけど, あと1subしかない. 
- もっと早くからアンサンブルの準備をして, 最後の一日は予備日として残しておくくらいではないといけないと思った. 
- 公開されてる重みとアンサンブルした. 
- もうやることがないのでコードの整理をしてこのレポジトリにあげた. 

***20200317*** <br> 
- 他の人のsolution見たいたら, めちゃくちゃアンサンブルしてるけど, 自分のコードだと10modelくらいが限界でなんでこんなアンサンブルできるのかわからない. 