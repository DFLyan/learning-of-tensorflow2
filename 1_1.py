import tensorflow as tf


random_float = tf.random.uniform(shape=())

zero_vector = tf.zeros(shape=(2))

A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

print(A.shape)
print(A.dtype)
print(A.numpy())

C = tf.add(A, B)
D = tf.matmul(A, B)


