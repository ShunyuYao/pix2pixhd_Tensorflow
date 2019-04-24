import tensorflow as tf

def _extract_fn(tf_record):
    features={
        'Label':tf.FixedLenFeature([], tf.string),
        'Real':tf.FixedLenFeature([], tf.string),
    }
    sample = tf.parse_single_example(tf_record, features)

    image_label = tf.decode_raw(sample['Label'], tf.uint8)
    image_label = tf.reshape(image_label, [480, 480])

    image_real = tf.decode_raw(sample['Real'], tf.uint8)
    image_real = tf.reshape(image_real, [480, 480, 3])

    return [image_label, image_real]

# def read_and_decode(filename):
#     # filename_queue = tf.train.string_i128nput_producer([filename])
#     # reader = tf.TFRecordReader()
#     # _, serialized_example = reader.read(filename_queue)
#     dataset = tf.data.TFRecordDataset([filename])
#     dataset.map(_extract_fn)
#     iterator = dataset.make_one_shot_iterator()
#     next_data = iterator.get_next()
#
#     return next_data

filename = '/home/projects/pix2pixhd_Tensorflow/datasets/tf_train/train.tfrecords'
dataset = tf.data.TFRecordDataset([filename])
dataset = dataset.map(_extract_fn)
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = sess.run(next_data)
    print(data[1])
