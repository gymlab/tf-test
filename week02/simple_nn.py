import tensorflow as tf
import numpy as np

# [털, 날개]: 있느냐 없느냐 -> One-hot coding
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

X = tf.placeholder(tf.float32)  # 입력 데이터
Y = tf.placeholder(tf.float32)  # 출력 데이터

W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

# Weight node 할당: (X*W)+b
L = tf.add(tf.matmul(X, W), b)
# ReLU는 tf.nn에 존재.
# Activation function: relu(X*W)+b
L = tf.nn.relu(L)
# softmax 확률 값을 사용:
model = tf.nn.softmax(L)

# 손실함수: Cross-entropy (simple_nn.md 참고)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 기본적인 gradient descent method, learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# Tensorflow session initaialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 100번의 학습 진행
for step in range(100):
    # 학습 진행
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 10번에 한 번 꼴로 손실값 출력.
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

# 두 결과가 같은 지 확인
is_correct = tf.equal(prediction, target)
# tf.cast: boolean을 0과 1의 원하는 자료형으로 변환
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))