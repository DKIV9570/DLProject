import os
import cv2
import numpy
import tensorflow as tf
import time
from keras.preprocessing.image import ImageDataGenerator

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
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(24, 24),
                                                        shuffle=True,
                                                        batch_size=10)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(24, 24),
                                                                  shuffle=True,
                                                                  batch_size=10)
    tags = validation_generator.class_indices
    return train_generator, validation_generator, tags


def build_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(24, 24, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='SAME'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='SAME'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='SAME'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(153, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model):
    tf.io.gfile.rmtree('./logs/')
    time_begin = time.time()
    train_generator, validation_generator = get_generator()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=700,
                                  epochs=10,
                                  validation_data=validation_generator,
                                  validation_steps=50,
                                  callbacks=[TensorBoard])

    model.save('./models/classifierCNN_2.h5')
    print('action took {} seconds'.format((time.time() - time_begin)))
    return model


def read_model():
    model = tf.keras.models.load_model('./models/classifierCNN_2.h5')
    return model


def classification_1():
    icon_list = os.listdir("Data/new/Unsorted/")
    model = tf.keras.models.load_model('./models/classifierCNN_2.h5')
    train_generator, validation_generator, tags = get_generator()
    new_tags = {v: k for k, v in tags.items()}

    count = []
    for i in range(153):
        count.append(0)

    for icon in icon_list:
        img = cv2.imread("Data/new/Unsorted/"+str(icon))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        icon_numpy = numpy.expand_dims(img, axis=0)

        result = model.predict(icon_numpy).tolist()
        max_value = max(result[0])
        index = result[0].index(max_value)
        if max_value > 0.9:
            dir = "Data/new/" + new_tags[index]
            if not os.path.exists(dir):
                os.makedirs(dir)
            os.rename("Data/new/Unsorted/" + str(icon), dir + "/" + str(count[index]) + ".jpg")
            count[index] += 1
        else:
            print(icon, max_value, index)


def init(string):
    folders = os.listdir("Data/new/")
    count = 0
    for folder in folders:
        if folder != "Unsorted":
            icons = os.listdir("Data/new/"+str(folder)+"/")
            for icon in icons:
                os.rename("Data/new/" + str(folder) + "/" + str(icon),
                          "Data/new/Unsorted/" + string + str(count)+".jpg")
                count += 1




