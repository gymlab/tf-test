import tensorflow as tf
import numpy as np

# Numpy의 text load 함수
# unpack=True인 경우 원시 데이터를 transpose함.
data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

# data.csv 형태
# 털, 날개, 기타, 포유류, 조류

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 학습에 직접 관여하지는 않고(trainable=False), 학습 횟수를 기록하고자 하는 변수를 설정.
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# global_step을 입력하면, 본 변수를 학습시마다 1씩 증가시킴.
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
# Saver를 통해 앞서 선언한 변수들을 저장하거나 불러옴.
saver = tf.train.Saver(tf.global_variables())

# 원하는 파일이 있는지 확인하고 있으면 불러오고, 없으면 초기화.
# 학습된 모델을 저장한 파일을 checkpoint파일이라고 함.
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 이전 예제와 달리 global_step은 tensor 타입의 변수이므로 session run을 통해 출력해야 함.
    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('Predictions: ', sess.run(prediction, feed_dict={X: x_data}))
print('GT: ', sess.run(target, feed_dict={Y: y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
