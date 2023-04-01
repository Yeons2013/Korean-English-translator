# Seq2Seq Model을 활용한 한글-영문 번역기 만들기

 Third Mini Project(NLP)

 ## 프로젝트 주제


+ 주어진 타겟문장을 번역하는 Model을 생성하고, 성능을 끌어올려 최고의 Bleu Score를 기록한다.
  
<br>


---


## 사용 데이터
+ AI Hub의 '한국어-영어 번역(병렬) 말뭉치' 중 '구어체1~2, 대화체' 사용 <br>
[➡️<AI Hub 접속>](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=126)
<img src='https://media.discordapp.net/attachments/1002189622912221250/1090074898405281842/image.png' width=600>
+ 타겟 문장이 구어체 형식의 문장들이며, 모든 데이터를 활용하기에는 자원적 한계가 존재
+ 총 구어체1 20만개, 구어체2 20만개, 대화체 10만개로 총 50만개 데이터


<br>

---



## RNN Model

Recurrent Neural Network에 attention을 활용해 seq2seq Model 생성




### 1. 전처리
+ 중복 및 Null값 확인 후 제거
+ 영어의 경우 띄어쓰기를 기준으로 토큰화
+ 한글의 경우 형태소 분석기를 활용(Okt, Mecab, Kkma, Hannanum, Komoran  → Mecab 선정 : 다른 형태소 분석기와 비교했을 때 정확도도 높은편이고, 빠른 속도를 보임)
+ 각 토큰을 숫자로 임베딩
+ (중간 점검 후 오역이 심해 데이터 추가 투입 + 토큰의 개수가 많은 문장 삭제 작업)
  <img src='https://media.discordapp.net/attachments/1002189622912221250/1090086531357753434/image.png' width=600>




<br>

### 2. 모델 구축 및 학습
+ LSTM, GRU 두 Base Model 구축

<br>

+ LSTM(Long Short Term Memory)
  + VanRNN의 기울기 소실 문제를 줄이기 위한 모델
  + 장기 기억과 단기 기억을 따로 학습해 문장이 길어짐으로써 발생하는 소실 문제 해결

<br>

+ GRU(Gated Recurrent Unit)
  + LSTM이 가지는 3개의 게이트를 2개로 간소화하고 Cell State를 없앤 모델
  + 파라미터가 감소하여 LSTM보다 빠른 학습 속도와 비슷한 수준의 성능

<br>

+ Attention
  + 주어진 쿼리(Query)에 대해 모든 키(Key)의 유사도를 각각 구함
  + 해당 유사도를 키와 맵핑되어있는 각각의 값(Value)에 반영
  + 유사도가 반영된 값(Value)을 모두 더해서 리턴
  + 쿼리(Query) : t 시점의 디코더 셀에서의 은닉 상태
  + 키(Key) : 모든 시점의 인코더 셀의 은닉 상태들(Querry에 대한 attention 기여도를 계산할 대상)
  + 값(Value) : 모든 시점의 인코더 셀의 은닉 상태들(attention 크기를 계산하기 위한 값)

<br>

+ 성능 개선을 위한 다양한 Model Handling 작업
  + Hidden Units : 32 ~ 256 → 256에 가장 높은 ACC
  + Embedding dim : 32 ~ 128 → 32에서 가장 높은 ACC
  + Depth(Num of Layer) : 1 ~ 4 → 4에서 가장 높은 ACC
  + Dropout : 0.1 ~ 0.7 →  0.3에서 가장 높은 ACC

<br>

+ 결과 추론   
  + Hidden Units, Depth(Num of Layer) : 높았을 때 좋은 ACC를 보였던 것으로 미루어 보아, 문제 자체의 복잡도가 높기 때문에 Model의 복잡도가 높아질 수록 좋은 ACC를 보이는 것으로 추정.
  + Embedding dim : 대부분 문장의 길이가 길지 않고, 사용되는 단어의 폭 또한 범위가 많이 넓지는 않기 때문에 Embedding 차원의 수가 지나치게 많은면 오히려 학습에 방해가 되는 것으로 추정.
  + Dropout : 0.3이 Best, 과적합 문제가 크지 않은 것으로 판단.

<br>

### 3. 모델 평가
+ Valiadation Accuracy + 사람에 의한 평가 지표 + BLEU Score
+ 높은 Acc에도 불구하고 사람의 눈으로 확인했을 때 심한 오역과 낮은 BLEU Score
+ 평가 결과 추론
  + 번역 모델을 단순 Acc로 평가하는 것은 적절치 않음 (Transformer 모델에서는 Masked Acc 적용)
  + 단순 ACC를 높이는 방향으로 학습, OOV 문제, RNN 계열 모델의 한계 등으로 인해 정상적인 번역기를 생성하지 못함. → Transformer Model로 전환

<br>
<br>

---

## Transformer Model
**Attention is all you need**

+ Transformer
  + CNN이나 RNN을 사용하지 않고, 인코더-디코더 구조를 설계
  + 번역 성능에서 RNN보다 우수한 성능을 보임
  + Attention 과정을 여러 레이어에서 반복
  + Attention을 활용해 RNN의 Long-term dependecty problem을 해결하고, 순차적인 계산이 아닌 행렬 병렬연산으로 빠르게 학습 가능
  
<br>

### 1. 전처리
+ 구두점에 대해 띄어쓰기
+ 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
+ 시작 토큰과 종료 토큰에 대한 정수 부여
+ 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
+ 텐서플로우 dataset을 이용하여 셔플(shuffle), 배치 크기로 데이터를 묶음
+ 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성




<br>

### 2. 모델 구축 및 학습
+ 포지셔널 인코딩(positional_encoding)
  + 단어의 위치 정보를 얻기 위해 각 단어의 임베딩 벡터에 위치 정보들을 더해 모델의 입력으로 사용
  + 싸인, 코싸인 활용
  
``` python
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)
```

``` python
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
```

+ 인코더(Encoder)
  + 입력시퀀스의 데이터 정보 추출
  + 하나의 인코더 층에는 Self-Attention과 Feed Forward 신경망으로 구성
  + Multi-Head Attention(Self-Attention을 병렬적으로 사용)
  + 성능 향상을 위해 residual learning 사용
``` python
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
```
+ 디코더(Deoder)
  + N개의 디코더 층을 쌓음
  + 교사 강요(Teach Forcing) 사용
    + 학습과정에서 번역할 문장에 해당되는 문장 행렬을 한 번에 입력 받음.
    + 디코더는 이 문장 행렬로부터 각 시점의 단어를 예측하도록 훈련<br><br>


  + 룩-어헤드 마스크(Look-ahead mask) 사용 : 현재 시점의 예측에서 현재 시점보다 미래에 있는 단어를 참고하지 못하도록 함.

``` python
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
```


<br>

### 3. 모델 평가
+ Masked Accuracy + 사람에 의한 평가 지표 + BLEU Score
+ RNN Model과 비교했을 때 줄어든 오역 + 높은 BLEU Score
+ 평가지표를 Masked Acc & Loss로 변경함으로써 모델이 올바른 방향으로 Loss를 줄이며 학습할 수 있었고, Trasnformer 모델 또한 번역 Task에 적합하기 때문에 좋은 번역 성능을 보여준 것으로 추론

