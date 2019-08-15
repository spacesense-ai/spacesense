from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from osgeo import ogr, gdal
from rasterio.plot import show, show_hist
from sklearn.svm import SVC
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import numpy as np
import math




""" Available model architectures"""

class SVC_by_pixel(object):
    def __init__(self):
        """
        Simple Support Vector Machine for regression
        """
        self.model = None
        self.svc_models = None

    def build_model(self):
        self.svc_models = GridSearchCV(SVC(kernel='rbf', gamma=0.1), cv=5,
                                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                                   "gamma": np.logspace(-2, 2, 5)})

    def fit(self, X_train, y_train):
        self.build_model()
        self.svc_models.fit(X_train, y_train)
        self.model = self.svc_models.best_estimator_
        self.model.fit(X_train, y_train)


class land_use(object):
    def __init__(self):
        """
        Simple Support Vector Machine for regression
        """
        self.model = None
        self.svc_models = None

    def build_model(self):
        """
        TBD

        :return: ML model to classify forest_cover
        """

    def fit(self, X_train, y_train):
        self.build_model()
        self.svc_models.fit(X_train, y_train)
        self.model = self.svc_models.best_estimator_
        self.model.fit(X_train, y_train)



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


""" Base Class"""
class by_pixel(object):
    def __init__(self):
        """ Reconstructed Solar Induced Fluorescence

        Parameters
        ----------

        """
        self.model = None
        self.model_archi = None

    def train(self, x, y, model_architecture=cnn_custom(), test_data_size=0.10):
        '''

        :return:
        '''
        if len(x.shape) == 3:
            if (x.shape[2]<x.shape[0]) and (x.shape[2]<x.shape[1]):
                X = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            elif (x.shape[0]<x.shape[1]) and (x.shape[0]<x.shape[2]):
                X = x.reshape(x.shape[1] * x.shape[2], x.shape[0])
                    
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

    def predict(self, x):
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

    def upload_model(self, model_path):
        self.model = load_model(model_path)
        print('model successfully loaded')
        print('Input shape:', self.model.input_shape[1])
        print(self.model.summary())


