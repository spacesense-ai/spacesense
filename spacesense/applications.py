from spacesense import classification as cm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np


class landuse(cm.by_pixel):

    def train(self, x, y=None, model_architecture=cm.OneClassSVM(), test_data_size=0.10):
        '''

        :return:
        '''
        if len(x.shape) == 3:
            if (x.shape[2] < x.shape[0]) and (x.shape[2] < x.shape[1]):
                X = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            elif (x.shape[0] < x.shape[1]) and (x.shape[0] < x.shape[2]):
                X = x.reshape(x.shape[1] * x.shape[2], x.shape[0])

        else:
            X = x

        # split dataset into train and test
        X_train, X_test = train_test_split(X, test_size=test_data_size, random_state=1)

        # scale X_train and X_test
        X_train = preprocessing.scale(X_train, axis=1)
        X_test = preprocessing.scale(X_test, axis=1)

        self.model_archi = model_architecture
        print('fitting the model')
        self.model_archi.fit(X_train)
        self.model = self.model_archi.model

        y_pred = self.model.predict(X_test)
        print('Training metrics:')
        self.training_metrics(y_test=None,y_pred=y_pred)






