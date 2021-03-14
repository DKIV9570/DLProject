import os
import cv2
import numpy
import tensorflow as tf
import time

from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.applications import VGG16

TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 Error

# set the gpu
config = tf.compat.v1.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def get_generator():
    main_dir = "Data/champions/"
    train_dir = os.path.join(main_dir, "train")
    validation_dir = os.path.join(main_dir, "validation")
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(32, 32),
                                                        shuffle=True,
                                                        batch_size=10)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(32, 32),
                                                                  shuffle=True,
                                                                  batch_size=10)
    return train_generator, validation_generator


def build_model():
    model_top = keras.models.Sequential()
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(32, 32, 3))
    model_top.add(keras.layers.Flatten())
    model_top.add(keras.layers.Dropout(0.5))
    model_top.add(keras.layers.Dense(256, activation='relu'))
    model_top.add(keras.layers.Dense(153, activation='softmax'))

    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(model_top)

    conv_base.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model):
    tf.io.gfile.rmtree('./logs/')
    time_begin = time.time()
    train_generator, validation_generator = get_generator()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=700,
                                  epochs=20,
                                  validation_data=validation_generator,
                                  validation_steps=50,
                                  callbacks=[TensorBoard])

    model.save('./models/classifier_VGG.h5')
    print('action took {} seconds'.format((time.time() - time_begin)))

