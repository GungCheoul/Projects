"""
pre-trained CNN model(Convolutional Neural Network)
recalibration the model's weights for the right purpose
needs : dataset, several packages
"""

# Making model(Training step)

"""
Collecting image dataset(sample data), using webcam
Capture an image every 4 frames
"""
import cv2

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    exit()

sampled_num = 0
captured_num = 0

while webcam.isOpened():
    _, frame = webcam.read()
    sampled_num += 1

    cv2.imshow('captured', frame)

    if sampled_num == 4:
        captured_num += 1
        cv2.imwrite('./rock/img' + str(captured_num) + '.jpg', frame)
        # cv2.imwrite('./rock/img' + str(captured_num) + '.jpg', frame)
        # cv2.imwrite('./rock/img' + str(captured_num) + '.jpg', frame)
        sampled_num = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
Generate the classifier and Train the dataset
take pre-trained "ResNet50" model
"""
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
#
# path_dir1 = './rock/'
# path_dir2 = './paper/'
# path_dir3 = './scissors/'
#
# file_list1 = os.listdir(path_dir1)
# file_list2 = os.listdir(path_dir2)
# file_list3 = os.listdir(path_dir3)
#
# # preparing train image
# num = 0
# train_img = np.float32(np.zeros((1268, 224, 224, 3)))
# train_label = np.float64(np.zeros((1268, 1)))
#
# for img_name in file_list1:
#     img_path = path_dir1 + img_name
#     img = load_img(img_path, target_size=(224,224))
#
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     train_img[num, :, :, :] = x
#
#     train_label[num] = 0 # rock
#     num += 1
# for img_name in file_list2:
#     img_path = path_dir2 + img_name
#     img = load_img(img_path, target_size=(224,224))
#
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     train_img[num, :, :, :] = x
#
#     train_label[num] = 1 # paper
#     num += 1
# for img_name in file_list3:
#     img_path = path_dir3 + img_name
#     img = load_img(img_path, target_size=(224,224))
#
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     train_img[num, :, :, :] = x
#
#     train_label[num] = 2 # scissors
#     num += 1
#
# # mixing image
# n_elem = train_label.shape[0]
# indices = np.random.choice(n_elem, size=n_elem, replace=False)
#
# train_label = train_label[indices]
# train_img = train_img[indices]
#
# IMG_SHAPE = (224, 224, 3)
#
# base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
# base_model.trainable = False
# base_model.summary()
# print('Number of layers in the base model: ', len(base_model.layers))
#
# GAP_layer = GlobalAveragePooling2D()
# dense_layer = Dense(3, activation=tf.nn.softmax)
#
# model = Sequential([
#     base_model,
#     GAP_layer,
#     dense_layer
#     ])
#
# base_learning_rate = 0.001
# model.compile(optiimzer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
#
# model.fit(train_img, train_label, epochs=5)
#
# model.save('model.h5')
# print('Saved model')

"""
Test the trained model
"""
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# # from PIL import ImageFont, ImageDraw, Image
#
# model = load_model('model.h5')
# model.summary()
#
# webcam = cv2.VideoCapture(0)
#
# while webcam.isOpened():
#     _, frame = webcam.read()
#
#     img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#
#     prediction = model.predict(x)
#     predicted_class = np.argmax(prediction[0])
#     print(prediction[0])
#     print(predicted_class)
#
#     if predicted_class == 0:
#         me = "바위"
#     elif predicted_class == 1:
#         me = "보"
#     elif predicted_class == 2:
#         me = "가위"
#
#     # fontpath = "font/gulim.ttc"
#     # font1 = ImageFont.truetype(fontpath, 100)
#     # frame_pil = Image.fromarray(frame)
#     # draw = ImageDraw.Draw(frame_pil)
#     # draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
#     # frame = np.array(frame_pil)
#     # cv2.imshow('show', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
