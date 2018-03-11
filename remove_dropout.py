import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]

# read frozen graph and display nodes
graph = tf.GraphDef()
with tf.gfile.Open('out/opt_model.pb', 'rb') as f:
    data = f.read()
    graph.ParseFromString(data)
    
display_nodes(graph.node)


"""
graph.node[67].input[0] = 'dense_1/Relu'
graph.node[90].input[0] = 'dense_2/Relu'
# Remove dropout nodes
nodes = graph.node[:46] + graph.node[65:70] + graph.node[88:] 

# Save graph
output_graph = graph_pb2.GraphDef()
output_graph.node.extend(nodes)
with tf.gfile.GFile('out/skin_cancer_model.pb', 'wb') as f:
    f.write(output_graph.SerializeToString())
"""
