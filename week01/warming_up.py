import tensorflow as tf # tensorflow library

hello = tf.constant('Hello, TensorFlow!')   # tf의 constant 객체
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

# 위의 출력 결과
# Tensor("Const:0", shape=(), dtype=string)
# Tensor("Add:0", shape=(), dtype=int32)
#
# 출력이 예상과 다른 이유는, 텐서플로의 프로그램 방식은 지연 실행(lazy evaluation)으로,
# 1. 그래프의 생성
# 2. 그래프의 실행
# 으로 구성되기 때문이다.
# 위의 코드는 그래프의 생성 코드로 다음과 같은 그래프 생성을 수행한다.
# a=10  b=32
#    | /
#    +
#    |
#    c
#
# 따라서 원하는 출력 결과를 보기 위해서는 그래프의 실행을 해야하며, 그래프 실행은 Session 안에서 이루어진다.
# Session은 tensorflow 객체이며 이 객체의 run 메서드를 사용하여 그래프의 실행이 가능하다.

sess = tf.Session()

print(sess.run(hello))
print(sess.run([a, b, c]))

sess.close()

# b'Hello, TensorFlow!'
# [10, 32, 42]

import tensorflow as tf

a = tf.constant(5)
b = tf.constant(10)
c = tf.constant(20)

d = a*b+c

print(d)

sess = tf.Session()
result = sess.run(d)
print(result)
sess.close()
