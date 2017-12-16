import tensorflow as tf

# placeholder 는 데이터 형테가 정해지지 않은 부분을 None 으로 설정이 가능.
X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]

# tf.random_normal([3, 2]): [3, 2] shape 의 정규분포로 행렬 생성
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1, 3]))

# tf.matmul(A, B): 행렬 곱 연산: np.dot(A,B)
# Broadcasting 기능을 통해 b를 np.dot(A,B)에 맞게 변환함.
expr = tf.matmul(W, X) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # Variable initialization

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()
