손실함수 (Loss function) 코드 분석
--------------------------------
```
# 손실함수: Cross-entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model)), axis=1)
```
```
예) Y            model            -> Y * tf.log(model)
    [[1 0 0]     [[0.1 0.7 0.2]      [[-1. 0     0]
     [0 1 0]]     [0.2 0.8 0.0]]      [0   -0.09 0]]
```
```
Y * tf.log(model) -> reduce_sum(axis=1) # column vector 생성
[[-1. 0     0]       [-1. -0.09]
 [0   -0.09 0]]
```
```
reduce_sum -> reduce_mean
[-1. -0.09]   -0.545
```
