'''
使用公开的fer2013数据集,以及keras自带的xxx模型完成预测
'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import os
import shutil

batch_size = 64
epoch_num = 30
shuffly_buffer_size = 1024
learning_rate = 0.001
train_tfrecord_path = './fer2013/train.tfrecord'
test_tfrecord_path = './fer2013/test.tfrecord'
checkpoint_path = './fer2013_models/cp.ckpt'
logs_path = './fer2013_logs/'

# 1.准备数据
# 将tfrecord格式的训练集、测试集转为dataset对象
raw_train_dataset = tf.data.TFRecordDataset(train_tfrecord_path)
raw_test_dataset = tf.data.TFRecordDataset(test_tfrecord_path)
# 告诉解析器每个example的feature结构
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

# 解析rawdataset，并进行进一步处理
decoded_train_dataset = raw_train_dataset.map(_decode_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
decoded_train_dataset = decoded_train_dataset.shuffle(buffer_size=shuffly_buffer_size)
decoded_train_dataset = decoded_train_dataset.batch(batch_size)
decoded_train_dataset = decoded_train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

decoded_test_dataset = raw_test_dataset.map(_decode_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
decoded_test_dataset = decoded_test_dataset.shuffle(buffer_size=shuffly_buffer_size)
decoded_test_dataset = decoded_test_dataset.batch(batch_size)
decoded_test_dataset = decoded_test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 2.准备模型
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_shape=(48, 48, 1), classes=7)

# 3.
class VGGNet(tf.keras.Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 当没有BN操作时时可以把激活直接写在卷积操作里 #
        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation(activation='relu')

        self.conv2 = Conv2D(filters=64, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.pool2 = MaxPool2D(2, 2, padding='same')
        self.drop2 = Dropout(0.2)

        self.conv3 = Conv2D(filters=128, kernel_size=3, padding='same')
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')

        self.conv4 = Conv2D(filters=128, kernel_size=3, padding='same')
        self.bn4 = BatchNormalization()
        self.act4 = Activation('relu')
        self.pool4 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop4 = Dropout(0.2)

        self.conv5 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.bn5 = BatchNormalization()
        self.act5 = Activation('relu')

        self.conv6 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.bn6 = BatchNormalization()
        self.act6 = Activation('relu')

        self.conv7 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.bn7 = BatchNormalization()
        self.act7 = Activation('relu')
        self.pool7 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop7 = Dropout(0.2)

        self.conv8 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn8 = BatchNormalization()
        self.act8 = Activation('relu')

        self.conv9 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn9 = BatchNormalization()
        self.act9 = Activation('relu')

        self.conv10 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn10 = BatchNormalization()
        self.act10 = Activation('relu')
        self.pool10 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop10 = Dropout(0.2)

        self.conv11 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn11 = BatchNormalization()
        self.act11 = Activation('relu')

        self.conv12 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn12 = BatchNormalization()
        self.act12 = Activation('relu')

        self.conv13 = Conv2D(filters=512, kernel_size=3, padding='same')
        self.bn13 = BatchNormalization()
        self.act13 = Activation('relu')
        self.pool13 = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop13 = Dropout(0.2)

        self.flatten = Flatten()
        self.dense14 = Dense(units=512, activation='relu')
        self.drop14 = Dropout(0.2)
        self.dense15 = Dense(units=512, activation='relu')
        self.drop15 = Dropout(0.2)
        self.dense16 = Dense(units=7, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.act7(x)
        x = self.pool7(x)
        x = self.drop7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.act8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.act9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.act10(x)
        x = self.pool10(x)
        x = self.drop10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.act11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.act12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = self.act13(x)
        x = self.pool13(x)
        x = self.drop13(x)

        x = self.flatten(x)
        x = self.dense14(x)
        x = self.drop14(x)
        x = self.dense15(x)
        x = self.drop15(x)
        y = self.dense16(x)

        return y


self_vgg = VGGNet()

model = self_vgg
# 3.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=tf.keras.metrics.sparse_categorical_accuracy)

# 设置cp
if os.path.exists(checkpoint_path+'.index'):
    print('---------------------load the model ---------------------')
    model.load_weights(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True)

# 设置logs
if os.path.exists(logs_path):
    print('删除历史logs文件夹')
    shutil.rmtree(logs_path)
else:
    os.mkdir(logs_path)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1, update_freq='batch')

# 4.
model.fit(x=decoded_train_dataset, epochs=epoch_num, validation_data=decoded_test_dataset, validation_freq=1,
                callbacks=[cp_callback, tb_callback])
