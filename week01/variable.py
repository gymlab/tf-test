import tensorflow as tf

input_data = [1,2]
x = tf.placeholder(dtype=tf.float32)
W = tf.Variable([3],dtype=tf.float32)
y = W*x

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict={x:input_data})

print(result)