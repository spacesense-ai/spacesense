import os
import sklearn
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from osgeo import ogr, gdal
from rasterio.plot import show, show_hist


class by_pixel(object):
    def __init__(self):
        """ Reconstructed Solar Induced Fluorescence

        Parameters
        ----------

        """
        self.model = None
        self.model_archi = None

    def train(self, x, y, model_architecture=gentine_lab_rsif(), test_data_size=0.10):
        '''

        :return:
        '''
        if len(x.shape) == 3:
            X = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        else:
            X = x
        y = np.ravel(y)

        # split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_size, random_state=1)

        # scale X_train and X_test
        X_train = preprocessing.scale(X_train, axis=1)
        X_test = preprocessing.scale(X_test, axis=1)

        self.model_archi = model_architecture
        self.model_archi.fit(X_train,
                             y_train)
        self.model = self.model_archi.model

        y_pred = self.model.predict(X_test)

        # to add : metrics

    def predict(self, X):
        '''

        :return:
        '''
        if len(x.shape) == 3:
            x_pred = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        else:
            x_pred = x
        x_scaled = preprocessing.scale(x_pred, axis=1)
        y_pred = self.model.predict(x_scaled)

        if len(x.shape) == 3:
            y_pred_reshaped = y_pred.reshape(x.shape[0], x.shape[1])
        else:
            y_pred_reshaped = y_pred

        return y_pred_reshaped

    def plot_performance(self):
        '''

        :return:
        '''


class cnn_custom(object):
    def __init__(self):
        """
        REF
        """
        self.model = None
        self.fit_model = None

    def build_model(self, input_shp, output_shp):
        hidden1_num_units = 200
        hidden2_num_units = 200
        hidden3_num_units = 200
        hidden4_num_units = 200

        model = keras.Sequential([
            Dense(input_dim=input_shp, kernel_regularizer=l2(0.0001), activation='relu', units=hidden1_num_units),
            Dropout(0.2),
            Dense(input_dim=hidden1_num_units, kernel_regularizer=l2(0.0001), activation='relu',
                  units=hidden2_num_units),
            Dropout(0.2),
            Dense(input_dim=hidden2_num_units, kernel_regularizer=l2(0.0001), activation='relu',
                  units=hidden3_num_units),
            Dropout(0.1),
            Dense(input_dim=hidden3_num_units, kernel_regularizer=l2(0.0001), activation='relu',
                  units=hidden4_num_units),
            Dropout(0.1),
            Dense(input_dim=hidden4_num_units, activation='softmax', units=output_shp),

        ])
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        self.model = model

    def fit(self, X_train, y_train, epochs=100, batch_size=100, validation_split=0.1, verbose=1):
        self.build_model(X_train.shape[-1], 1)
        self.fit_model = self.model.fit(X_train, y_train, epochs=100,
                                        batch_size=batch_size,
                                        validation_split=validation_split,
                                        verbose=1)

    def load_model(path):

