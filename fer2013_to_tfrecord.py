'''
将公开表情数据集fer2013的csv中的训练集和测试集转为tfrecord格式，以便训练使用。
fer2013人脸表情数据集由35886张人脸表情图片组成，其中，测试图（Training）28708张，公共验证图（PublicTest）和私有验证图（PrivateTest）
各3589张，每张图片是由大小固定为48×48的灰度图像组成，共有7种表情，分别对应于数字标签0-6，具体表情对应的标签和中英文如下：
0 anger 生气； 1 disgust 厌恶； 2 fear 恐惧； 3 happy 开心； 4 sad 伤心；5 surprised 惊讶； 6 normal 中性。
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 定义各种文件存储的路径
csv_path = './fer2013/fer2013.csv'
train_tfrecord_path = './fer2013/train.tfrecord'
test_tfrecord_path = './fer2013/test.tfrecord'

# 其实这里不应该这么直接读的，太占用内存，不够优雅
fer2013_df = pd.read_csv(csv_path, header=0)
# print(fer2013_df.info)

# 取出训练集和测试集数据
train_data = fer2013_df[fer2013_df['Usage'] == 'Training']
test_data = fer2013_df[fer2013_df['Usage'] == 'PublicTest']
print(train_data.shape, test_data.shape)


# 开启一个tfrecord的写入io---存训练集
with tf.io.TFRecordWriter(train_tfrecord_path) as writer:
    for row_index in range(train_data.shape[0]):
        # print(train_data.iloc[row_index, 0: 2])
        # 将字符串格式的pixels像素点，转为np格式
        image_str = train_data.iloc[row_index, 1]
        image_list = image_str.split(' ')
        image = np.array(image_list).reshape((48, 48, 1)).astype('uint8')
        # print(image)
        # plt.title(train_data.iloc[row_index, 0])
        # plt.imshow(image)
        # plt.show()
        # 构建每个样本的feature字典,将ndarray的数组转为str，以符合BytesList的输入要求
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(train_data.iloc[row_index, 0])]))
        }
        # 将feature字典转为example 格式
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # 将example序列化，并写入tfrecord文件中
        writer.write(example.SerializeToString())

# 开启一个tfrecord的写入io---存测试集
with tf.io.TFRecordWriter(test_tfrecord_path) as writer:
    for row_index in range(test_data.shape[0]):
        # print(train_data.iloc[row_index, 0: 2])
        # 将字符串格式的pixels像素点，转为np格式
        image_str = test_data.iloc[row_index, 1]
        image_list = image_str.split(' ')
        image = np.array(image_list).reshape((48, 48, 1)).astype('uint8')
        # print(image)
        # plt.title(train_data.iloc[row_index, 0])
        # plt.imshow(image)
        # plt.show()
        # 构建每个样本的feature字典,将ndarray的数组转为str，以符合BytesList的输入要求
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),  # tobytes或者tostring都可以
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(test_data.iloc[row_index, 0])]))
        }
        # 将feature字典转为example 格式
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # 将example序列化，并写入tfrecord文件中
        writer.write(example.SerializeToString())