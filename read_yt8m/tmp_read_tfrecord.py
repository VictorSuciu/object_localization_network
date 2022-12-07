import tensorflow.compat.v1 as tf
import os
import numpy as np
from readers import YT8MFrameFeatureReader, YT8MAggregatedFeatureReader


vid_dir = '/home/vsuciu/data/yt8m/video'
fps = [os.path.join(vid_dir, f) for f in os.listdir(vid_dir)]


# def map_tfrecord(serialized):

#     # https://github.com/google/youtube-8m/blob/e6f6bf682d20bb21904ea9c081c15e070809d914/readers.py#L66
#     feature_map = {
#         "id": tf.io.FixedLenFeature([], tf.string),
#         "labels": tf.io.VarLenFeature(tf.int64)
#     }

#     return tf.io.parse_single_example(serialized, feature=feature)

tf.disable_eager_execution()
data_reader = YT8MAggregatedFeatureReader()
filename_queue = tf.train.string_input_producer(fps, num_epochs=1, shuffle=False)

data_out = data_reader.prepare_reader(filename_queue, batch_size=1)
print(data_out)
print()
print(data_out['video_ids'])


# for fp in fps:
#     # print(fp)
#     tfrecord = tf.data.TFRecordDataset(fp)
    
#     for element in tfrecord:
#         print(tf.train.Example().ParseFromString(element.numpy()))

#     break

# for i, batch in enumerate(yt8m_dset):
#     print(i)

