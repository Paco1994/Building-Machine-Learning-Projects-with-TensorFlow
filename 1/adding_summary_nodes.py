import tensorflow as tf

sw = tf.summary.FileWriter('.', graph_def=None)

g = tf.Graph()
with g.as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32, name="primes")

  # Create another six-element vector. Each element in the vector will be
  # initialized to 1. The first argument is the shape of the tensor (more
  # on shapes below).
  ones = tf.ones([6], dtype=tf.int32, name="ddd")

  # Add the two vectors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)
  xd = tf.add(just_beyond_primes, ones)

  # Create a session to run the default graph.
  with tf.Session() as sess:
    print (just_beyond_primes.eval())
    sw.add_graph(g)
    tf.summary.merge_all()
    
    
