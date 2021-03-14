import os
import cv2
import numpy
import tensorflow as tf
import time

TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 Error

# set the gpu
config = tf.compat.v1.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def get_image(n):
    data = []
    for i in range(n):
        data.append(cv2.imread("Data/"+str(i+1)+".jpg"))
    x_train = numpy.array(data)

    label = []

    for i in range(n):
        l = []
        for j in range(n):
            l.append(0.01)
        l[i] = 0.99
        label.append(numpy.array(l))
    y_train = numpy.array(label)

    return x_train, y_train


def build_model(n):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n, activation='sigmoid'))

    # sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, n):
    # tf.io.gfile.rmtree('./logs/')

    x_train, y_train = get_image(n)

    # test = get_test_images()

    history = model.fit(x_train, y_train, batch_size=n, epochs=12, validation_split=0, callbacks=[TensorBoard])
    model.save('./models/compareCNN.h5')

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print("loss:", score[0])
    # print("accu:", score[1])

    return model


def read_model():
    model = tf.keras.models.load_model('./models/compareCNN.h5')
    return model


def classification_1():
    icon_list = os.listdir("Data/new/Unsorted/")
    model = read_model()

    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for icon in icon_list:
        img = cv2.imread("Data/new/Unsorted/"+str(icon))
        icon_numpy = numpy.expand_dims(img, axis=0)
        result = model.predict(icon_numpy).tolist()

        max1 = max(result[0])
        index = result[0].index(max1)
        result[0][index] = -1
        max2 = max(result[0])
        if max1 > 0.5 and max1-max2 > 0.1:
            os.rename("Data/new/Unsorted/"+str(icon), "Data/new/"+str(index+1)+"/"+str(count[index])+".jpg")
            count[index] += 1


def classification_2(n):
    count = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

    folders = os.listdir("Data/new/")
    model = train_model(build_model(n), n)
    for folder in folders:
        if folder != "Unsorted":
            icons = os.listdir("Data/new/"+str(folder)+"/")
            for icon in icons:
                img = cv2.imread("Data/new/"+str(folder)+"/" + str(icon))
                icon_numpy = numpy.expand_dims(img, axis=0)
                result = model.predict(icon_numpy).tolist()
                max1 = max(result[0])
                index = result[0].index(max1)
                result[0][index] = -1
                max2 = max(result[0])
                if max1 - max2 > 0.4:
                    os.rename("Data/new/"+str(folder)+"/" + str(icon),
                              "Data/new/" + str(index + 1) + "/" + str(count[index]) + ".jpg")
                    count[index] += 1


def initialize():
    folders = os.listdir("Data/new/")
    count = 0
    for folder in folders:
        if folder != "Unsorted":
            icons = os.listdir("Data/new/"+str(folder)+"/")

            for icon in icons:
                img = cv2.imread(icon)
                cv2.imshow("img", img)
                cv2.waitKey(0)
                img = img*255.0
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite("Data/new/"+str(folder)+"/"+str(count)+".jpg")
                # os.rename("Data/new/" + str(folder) + "/" + str(icon),
                #           "Data/new/Unsorted/" + string + str(count)+".jpg")
                count += 1


def start(n):
    model = build_model(n)
    model = train_model(model, n)

    time_begin = time.time()
    classification_1()
    print('action took {} seconds'.format((time.time() - time_begin)))

    time_begin = time.time()
    classification_2(n)
    print('action took {} seconds'.format((time.time() - time_begin)))



