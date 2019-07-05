import os
import tensorflow as tf

class RSIF(object)
    def __init__(self):
        """ Reconstructed Solar Induced Fluorescence

        Parameters
        ----------

        """
        self.model = None
        self.model_archi =None

    def train(self,x,y, model_architecture=gentine_lab_rsif(),test_data_size=0.10):
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

    def predict(self,X):
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



""" Available model architectures"""
class gentine_lab_rsif(object):
    """
    Parameters
    ----------

    Notes
    -----

    References
    ----------
    This algorithm takes reference from the work described in this paper:
    Gentine, P., & Alemohammad, S. H. (2018). Reconstructed Solar-Induced Fluorescence: A machine learning
    vegetation product based on MODIS surface reflectance to reproduce GOME-2 solar-induced fluorescence.
    Geophysical Research Letters, 45. https:// doi.org/10.1002/2017GL076294
    Same has been developed for matlab by author S. Hamed Alemohammad at https://github.com/HamedAlemo/RSIF

    """

    def __init__(self):
        """
        REF
        """
        self.model = None
        self.fit_model = None

    def build_model(self, input_shp, output_shp=1):
        model = keras.Sequential([
            layers.Dense(5, activation=tf.nn.relu, input_shape=[input_shp]),
            layers.Dense(7, activation=tf.nn.relu),
            layers.Dense(10, activation=tf.nn.relu),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        self.model = model

    def fit(self, X_train, y_train, epochs=100, batch_size=100, validation_split=0.1, verbose=1):
        self.build_model(X_train.shape[-1], 1)
        self.fit_model = self.model.fit(X_train, y_train, epochs=100,
                                        batch_size=batch_size,
                                        validation_split=validation_split,
                                        verbose=1)


class cnn_custom_rsif_example(object):
    def __init__(self):
        """
        Format for custom neural network architecture with tensorflow.
        """
        self.model = None
        self.fit_model = None

    def build_model(self, input_shp, output_shp=1):
        model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[input_shp]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        self.model = model

    def fit(self, X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=1):
        self.build_model(X_train.shape[-1], 1)
        self.fit_model = self.model.fit(X_train, y_train, epochs=100,
                                        batch_size=100,
                                        validation_split=0.2,
                                        verbose=1)


class SVR(object):

    def __init__(self):
        """
        Simple Support Vector Machine for regression:

        GridSearchCV is used to find the best performing model for given training data.
        Best model is used for predictions.
        """
        self.model = None
        self.svr_models = None

    def build_model(self):
        self.svr_models = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                                   "gamma": np.logspace(-2, 2, 5)})

    def fit(self, X_train, y_train):
        self.build_model()
        self.svr_models.fit(X_train, y_train)
        self.model = svr_models.best_estimator_
        self.model.fit(X_train, y_train)

