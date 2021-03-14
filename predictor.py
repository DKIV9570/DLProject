import os
import pickle
import pylab as pl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 Error

# set the gpu
config = tf.compat.v1.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


class Predictor:
    training_data = []  # total training data
    current_data = []  # data of a single game

    data_img = []  # img training data (mini-map)
    data_coords = []  # coordinates training data
    data_time = []  # time of training data
    data_targets = []  # the coordinate of the jungle champion

    data_img_n = []  # img after normalize
    data_coords_n = []  # coords after normalize
    data_time_n = []  # time after normalize

    model = keras.Model

    def init_lists(self):
        self.data_img = []  # img training data (mini-map)
        self.data_coords = []  # coordinates training data
        self.data_time = []  # time of training data
        self.data_targets = []  # the coordinate of the jungle champion

        self.data_img_n = []  # img after normalize
        self.data_coords_n = []  # coords after normalize
        self.data_time_n = []  # time after normalize

    def get_train_data(self):
        main_dir = "Data/BackUp/phrase2_datas/up/"
        training_datas = os.listdir(main_dir)
        for data in training_datas:
            self.init_lists()
            file_dir = os.path.join(main_dir, data)
            file = open(file_dir, "rb")
            self.current_data = pickle.load(file)
            file.close()

            self.preprocess()
            self.normalize()
            self.create_training_data()
            print("Successfully processed data:"+data)

    def preprocess(self):
        for line in self.current_data:
            if np.array([values for values in line[1].values()]).shape != (10,) and np.array(line[4]).shape != (0,):
                self.data_img.append(line[0])
                self.data_coords.append(np.array([values for values in line[1].values()]))
                self.data_time.append(line[3])
                self.data_targets.append(np.array(line[4]))

    def normalize(self):
        self.data_img_n = [img / 255.0 for img in self.data_img]  # make the value of every pixel in img between [0~1]

        coords_array = np.array(self.data_coords)
        coords_mean = coords_array.mean()
        coords_array = coords_array - coords_mean
        coords_std = coords_array.std()
        coords_array = coords_array / coords_std
        self.data_coords_n = coords_array.tolist()

        time_array = np.array(self.data_time)
        time_mean = time_array.mean()
        time_array = time_array - time_mean
        time_std = time_array.std()
        time_array = time_array / time_std
        self.data_time_n = time_array.tolist()

    def create_training_data(self):
        for i in range(5, len(self.data_coords_n)-1):
            data_line = [np.array(self.data_coords_n[i-5:i]).reshape(5, 20),
                         self.data_img_n[i], self.data_time[i]/150, self.data_targets[i+1]]
            self.training_data.append(data_line)

    def create_lstm(self, dim, regress=False):
        # define our MLP network
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(units=32, input_shape=(5, dim)))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation="relu"))

        # check to see if the regression node should be added
        if regress:
            model.add(keras.layers.Dense(2, activation="linear"))

        # return our model
        return model

    def create_dnn(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(16, input_dim=1))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation="relu"))

        return model

    def create_cnn(self, width, height, depth, filters=(16, 32, 64), regress=False):
        # initialize the input shape and channel dimension, assuming
        # TensorFlow/channels-last ordering
        input_shape = (height, width, depth)
        chan_dim = -1

        # define the model input
        inputs = keras.Input(shape=input_shape)

        # loop over the number of filters
        for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
            if i == 0:
                x = inputs

            # CONV => RELU => BN => POOL
            x = keras.layers.Conv2D(f, (3, 3), padding="same")(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.BatchNormalization(axis=chan_dim)(x)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = keras.layers.Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = keras.layers.Dense(16)(x)
        x = keras.layers.Activation("relu")(x)

        # check to see if the regression node should be added
        if regress:
            x = keras.layers.Dense(2, activation="linear")(x)

        # construct the CNN
        model = keras.Model(inputs, x)

        # return the CNN
        return model

    def build_model(self):
        model_lstm = self.create_lstm(20, regress=False)
        model_cnn = self.create_cnn(360, 360, 3, regress=False)
        model_dnn = self.create_dnn()

        x = keras.layers.concatenate([model_lstm.output, model_dnn.output])  # stub
        x = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)

        x = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
        # x = keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)

        output = keras.layers.Dense(2, activation='linear', name='output')(x)
        model = keras.Model(inputs=[model_lstm.input, model_dnn.input], outputs=output)

        self.model = model
        #model.summary()

    def my_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, "float32")
        return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true))))

    def train_model(self):
        tf.io.gfile.rmtree("./logs")

        cut = int(len(self.training_data)*0.9)
        training = self.training_data[:cut]
        validation = self.training_data[cut:]

        lstm_train_x = np.array([train[0] for train in training])
        cnn_train_x = np.array([train[1] for train in training])
        dnn_train_x = np.array([train[2] for train in training])
        train_y = np.array([train[3] for train in training])

        lstm_validation_x = np.array([va[0] for va in validation])
        cnn_validation_x = np.array([va[1] for va in validation])
        dnn_validation_x = np.array([va[2] for va in validation])
        validation_y = np.array([va[3] for va in validation])

        self.model.compile(loss=self.my_loss, optimizer="adam", metrics=['mae'])
        self.model.fit([lstm_train_x, dnn_train_x], train_y,
                       validation_data=([lstm_validation_x, dnn_validation_x], validation_y),
                       epochs=100,
                       batch_size=32,
                       callbacks=[TensorBoard])
        self.model.save('./models/predictor_v3.h5')

    def predict(self):
        model = keras.models.load_model('./models/predictor_v1.h5')

    def test(self):
        self.model = keras.models.load_model('./models/predictor_v3.h5', custom_objects={'my_loss': self.my_loss})
        cut = int(len(self.training_data)*0.9)
        training = self.training_data[:cut]
        validation = self.training_data[cut:]

        lstm_train_x = np.array([train[0] for train in training])
        cnn_train_x = np.array([train[1] for train in training])
        dnn_train_x = np.array([train[2] for train in training])
        train_y = np.array([train[3] for train in training])

        lstm_validation_x = np.array([va[0] for va in validation])
        cnn_validation_x = np.array([va[1] for va in validation])
        dnn_validation_x = np.array([va[2] for va in validation])
        validation_y = np.array([va[3] for va in validation])

        result_train = p.model.predict([lstm_train_x, dnn_train_x])
        resultx_t, resulty_t = result_train[:, 0], result_train[:, 1]
        pl.xlim(0, 360)
        pl.ylim(0, 360)

        pl.scatter(resultx_t, 360-resulty_t, alpha=1/10)
        pl.title("predict result train total points:" + str(len(result_train)))
        pl.show()

        pl.xlim(0, 360)
        pl.ylim(0, 360)
        pl.scatter(train_y[:, 0], 360-train_y[:, 1], alpha=1/10)
        pl.title("real point train total points:" + str(len(train_y)))
        pl.show()

        result_validation = p.model.predict([lstm_validation_x, dnn_validation_x])
        resultx, resulty = result_validation[:, 0], result_validation[:, 1]
        pl.xlim(0, 360)
        pl.ylim(0, 360)
        pl.scatter(resultx, 360-resulty, alpha=1/5)
        pl.title("predict result validation total points:"+str(len(result_validation)))
        pl.show()

        pl.xlim(0, 360)
        pl.ylim(0, 360)
        pl.scatter(validation_y[:, 0], 360-validation_y[:, 1], alpha=1/5)
        pl.title("real point validation total points:" + str(len(validation_y)))
        pl.show()


p = Predictor()
p.get_train_data()
p.build_model()
p.train_model()
p.test()
