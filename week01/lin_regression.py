import tensorflow as tf

# 학습에 사용할 x, y좌표를 입력: (1, 1), (2, 2), (3, 3)
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# -1.~1. 사이의 uniform 분포로 변수 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 각각의 placeholder에 이름을 부여하는 것이 가능!
# 후에 tensorboard에서도 이름 정보를 출력하므로 디버깅이 수월하다.
# 선언하지 않으면 임의로 선언한다.
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W * X + b

# tf.reduce_mean(loss) 모든 데이터에 대한 평균 square_loss를 사용
# tf.reduce_max 등 다양한 함수가 있으니, https://www.tensorflow.org/api_docs 에서 확인 바람.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Optimizer의 solver를 gradient descent로 설정.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# optimizer가 cost를 minimize하도록 설정.
train_op = optimizer.minimize(cost)

# Session 블록을 자동으로 만들고 종료 하도록 with ~ as 구문 사용
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 100번의 iteration 수행
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        # Matlab의 ~ 구문을 underbar(_)를 통해 사용 가능.

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
