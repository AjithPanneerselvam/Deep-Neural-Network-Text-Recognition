import tensorflow as tf

my_graph = tf.Graph()

with tf.Session(graph=my_graph) as session:
    x = tf.constant([1,3,6])
    y = tf.constant([1,1,1])
    
    op = tf.add(x,y)
    result = session.run(fetches = op)
    print(result)

