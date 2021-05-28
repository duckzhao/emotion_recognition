'''
使用训练好的model（虽然准确率不高），结合face_recognition完成人脸定位及表情预测，使用电脑摄像头完成时时的人脸检测
由于opencv自带的基于haar特征的级联人脸检测效果不太好，因此改用face_recognition（基于dlib）的方法完成人脸定位
'''
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import face_recognition

# 预测结果和实际情绪的映射字典
emotion_dict = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprised',
    6: 'normal'
}

# 绘制表情时的颜色列表
color_list = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

# cv2 puttext字体格式
font = cv2.FONT_HERSHEY_SIMPLEX

# 初始化 训练好的神经网络
model_path = './fer2013_models/cp.ckpt'
# 复现网络框架
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
model = VGGNet()
# 加载网络参数
model.load_weights(model_path)


# 使用face_recognition检测人脸位置，可能检测多张人脸，返回每个人脸的坐标位置
def detect_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 调用face_locations方法，找出img中的人脸，并返回他们的坐标（顶部，右侧，底部，左侧），其中可选model参数-'hog' 不太准、快/'cnn' 准、慢
    # face_locations = face_recognition.face_locations(gray_img, model='cnn')
    face_locations = face_recognition.face_locations(gray_img)
    # print(face_locations)
    return face_locations

# 在视频原图中画出人脸的位置，注意这里无需返回值，直接会修改传入的img
def draw_faces(img, faces, faces_emotion):
    color_index = 0
    for (top, right, bottom, left), emotion in zip(faces, faces_emotion):
        cv2.rectangle(img=img, pt1=(left, top), pt2=(right, bottom), color=color_list[color_index], thickness=2)
        cv2.putText(img, emotion, org=(int((right + left)/2), top), fontFace=font, fontScale=1,
                    color=color_list[color_index], thickness=2)
        color_index += 1


# 扣出人脸的roi区域，并送至神经网络进行检测，拿到结果，传入多个脸的坐标，返回多个表情结果
def predict_face_emotion(img, faces):
    # 遍历faces 从传入的img 根据 face扣出人脸图，转灰度，然后拼接在一起，然后送入神经网络预测，拿到预测结果
    img_list = []
    for top, right, bottom, left in faces:
        face_roi = img[top: bottom, left: right]
        gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_face_roi = cv2.resize(gray_face_roi, (48, 48)).reshape((48, 48, 1))
        img_list.append(gray_face_roi)
    x_predict = np.array(img_list).astype(np.float32) / 255.0
    result = model.predict(x_predict)
    return result

# 从返回的预测结果中解析出最大概率数值索引对应的表情
def analysis_true_emotion_with_result(faces_emotion_result):
    pred = tf.argmax(faces_emotion_result, axis=1).numpy()
    pred = list(pred)
    pred = [emotion_dict[emotion_key] for emotion_key in pred]
    # print(pred)
    return pred

def run():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('cap error,system exit')
            break
        frame = cv2.flip(frame, 1)
        # 检测人脸位置
        faces = detect_face(frame)
        # 在复制的图层进行表情绘制
        draw_frame = frame.copy()
        # 如果检测的人脸位置不为空才进来预测表情
        if len(faces) != 0:
            # 预测人脸的表情
            faces_emotion_result = predict_face_emotion(img=frame, faces=faces)
            # print(faces_emotion_result)
            faces_emotion = analysis_true_emotion_with_result(faces_emotion_result)
            draw_faces(draw_frame, faces, faces_emotion)

        # 展示视频画面
        cv2.imshow('video capture', draw_frame)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    run()