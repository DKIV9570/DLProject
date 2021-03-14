import os
import shutil

import cv2
import numpy
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from create_folders import get_champions
from classifier_all import init

TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 Error

# set the gpu
config = tf.compat.v1.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


class Champions10:
    tag = {}
    jungle = ''

    @staticmethod
    def init_folder():
        current_train_dir = "Data/champions/train/current_game/"
        current_validation_dir = "Data/champions/validation/current_game/"

        shutil.rmtree(current_train_dir)
        shutil.rmtree(current_validation_dir)

        os.mkdir(current_train_dir)
        os.mkdir(current_validation_dir)

    def prepare_data(self):
        champions_form = get_champions()
        champions = []

        jungle = int(input("Please input the number of the jungle champion"))
        eng_name = champions_form[jungle][0]
        eng_name = eng_name[eng_name.find(" ") + 1:]
        chi_name = champions_form[jungle][1]
        champions.append(eng_name + "-" + chi_name)
        self.jungle = eng_name + "-" + chi_name

        for i in range(9):
            order = int(input("For the rest, please input the number of the " + str(i + 1) + " champion"))
            eng_name = champions_form[order][0]
            eng_name = eng_name[eng_name.find(" ") + 1:]
            chi_name = champions_form[order][1]
            champions.append(eng_name + "-" + chi_name)

        main_train_dir = "Data/champions/train/"
        current_train_dir = "Data/champions/train/current_game/"

        main_validation_dir = "Data/champions/validation/"
        current_validation_dir = "Data/champions/validation/current_game/"

        for champion in champions:
            source_train = os.path.join(main_train_dir, champion)
            target_train = os.path.join(current_train_dir, champion)

            source_validation = os.path.join(main_validation_dir, champion)
            target_validation = os.path.join(current_validation_dir, champion)

            shutil.copytree(source_train, target_train)
            shutil.copytree(source_validation, target_validation)

    def get_generator(self):
        train_dir = "Data/champions/train/current_game/"
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(24, 24),
                                                            shuffle=True,
                                                            batch_size=100)

        validation_dir = "Data/champions/validation/current_game/"
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                      target_size=(24, 24),
                                                                      shuffle=True,
                                                                      batch_size=1000)
        self.tag = validation_generator.class_indices
        return train_generator, validation_generator

    def train_model(self):
        self.init_folder()
        self.prepare_data()

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(24, 24, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        tf.io.gfile.rmtree("./logs")

        time_begin = time.time()
        train_generator, validation_generator = self.get_generator()

        model.fit_generator(train_generator,
                            steps_per_epoch=5,
                            epochs=80,
                            validation_data=validation_generator,
                            validation_steps=1,
                            callbacks=[TensorBoard])

        model.save('./models/classifierCNN10.h5')
        print('action took {} seconds'.format((time.time() - time_begin)))

        return model

    def predict(self, array):
        model = tf.keras.models.load_model('./models/classifierCNN10.h5')
        new_tags = {v: k for k, v in self.tag.items()}
        result_number = model.predict(array)
        result_name = []
        for num in result_number:
            num = num.tolist()
            name = new_tags[num.index(max(num))]
            name = name[:name.find("-")]
            probability = round(max(num), 3)
            result_name.append([name, probability])
        return result_name


# def test_1():
#     time_begin = time.time()
#     icon_list = os.listdir("Data/new/Unsorted/")
#     model = tf.keras.models.load_model('./models/classifierCNN10.h5')
#     train_generator, validation_generator, tags = get_generator()
#
#     new_tags = {v: k for k, v in tags.items()}
#     count = []
#     print(new_tags)
#     for i in range(10):
#         count.append(0)
#
#     for icon in icon_list:
#         img = cv2.imread("Data/new/Unsorted/" + str(icon))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img / 255.0
#         icon_numpy = numpy.expand_dims(img, axis=0)
#
#         result = model.predict(icon_numpy).tolist()
#
#         max_value = max(result[0])
#         index = result[0].index(max_value)
#         dir = "Data/new/" + new_tags[index]
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#         os.rename("Data/new/Unsorted/" + str(icon), dir + "/" + str(count[index]) + ".jpg")
#         count[index] += 1
#
#     print('action on one image took {} seconds average'.format((time.time() - time_begin) / sum(count)))
#
#
# def test_2():
#     icon_list = os.listdir("Data/new/test/")
#     model = tf.keras.models.load_model('./models/classifierCNN10.h5')
#     train_generator, validation_generator, tags = get_generator()
#     new_tags = {v: k for k, v in tags.items()}
#     for icon in icon_list:
#         img = cv2.imread("Data/new/test/" + icon)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype("float") / 255.0
#         # cv2.imshow("img", img)
#         # cv2.waitKey(0)
#         icon_numpy = numpy.expand_dims(img, axis=0)
#
#         result = model.predict(icon_numpy).tolist()
#         max_value = max(result[0])
#         index = result[0].index(max_value)
#         tag = new_tags[index]
#         print(icon + " is " + tag + " with confidence level of " + str(max_value))


