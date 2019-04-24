import tensorflow as tf

export_dir = './datasets/train/Logs'
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load()
