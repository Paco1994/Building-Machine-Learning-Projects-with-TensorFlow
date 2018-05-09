import tensorflow as tf
sess = tf.InteractiveSession()
t_matrix = tf.constant([[1,2,3],
                        [4,5,6],
                        [7,8,9]])
t_array = tf.constant([1,2,3,4,9,8,6,5])
t_array2 = tf.constant([2,3,4,5,6,7,8,9])

print(tf.slice(t_matrix, [1,1],[2,2]).eval())
print(tf.split(axis=0, num_or_size_splits=2, value=t_array))
print(tf.tile([1,2],[3]).eval())
print(tf.pad(t_matrix, [[0,1],[2,1]]).eval())
print(tf.concat(axis=0, values=[t_array, t_array2]).eval())
print(tf.stack([t_array, t_array2]).eval())
print(sess.run(tf.unstack(t_matrix)))
print(tf.reverse(t_matrix, [False,True]).eval())