import tensorflow as tf

x = tf.placeholder(dtype=tf.float32)
y = x * 2

input_data = [1, 2]

sess = tf.Session()
result = sess.run(y, feed_dict={x: input_data})
print(result)
sess.close()