'''
验证下对tfrecord数据的存储是否成功
'''
import tensorflow as tf
from matplotlib import pyplot as plt
train_tfrecord_path = './fer2013/train.tfrecord'

raw_train_dataset = tf.data.TFRecordDataset(train_tfrecord_path)
feature_description = {
    'image': tf.io.FixedLenFeature([], dtype=tf.string),
    'label': tf.io.FixedLenFeature([], dtype=tf.int64)
}
# 定义进一步解析dataset成为可用训练集的map函数
def _decode_dataset(example_string):
    # 将dataset的组成元素example_str解析为example字典格式
    example_dict = tf.io.parse_single_example(serialized=example_string, features=feature_description)
    # # 当前image属性为string，使用tf解码为jpg图片像素数组格式
    example_dict['image'] = tf.io.decode_raw(input_bytes=example_dict['image'], out_type=tf.uint8)
    # print(example_dict['image'])
    # example_dict['image'] = tf.io.decode_jpeg(contents=example_dict['image']) # 这一句就是为了把pic转为tensor，现在已经是tensor了，不需要这句了
    # 对该图片进行归一化处理
    example_dict['image'] = tf.reshape(tf.cast(example_dict['image'], tf.float32) / 255., shape=(48, 48, 1))
    return example_dict['image'], example_dict['label']

for example_string in raw_train_dataset:
    x, y = _decode_dataset(example_string)
    print(x, y)
    break