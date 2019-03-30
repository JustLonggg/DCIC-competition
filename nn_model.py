import pandas as pd
import numpy as np 
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from utils import get_data


class NNModels(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_nn_base_model(input_dim=48):

        nn_model = Sequential()

        nn_model.add(Dense(400, input_dim=input_dim, kernel_initializer='normal',activation='relu'))
        #nn_model.add(BatchNormalization())
        nn_model.add(Dropout(0.5))

        nn_model.add(Dense(200, kernel_initializer='normal',activation='relu'))
        #nn_model.add(BatchNormalization())
        nn_model.add(Dropout(0.5))

        nn_model.add(Dense(50, kernel_initializer='normal',activation='relu'))
        #nn_model.add(BatchNormalization())
        nn_model.add(Dropout(0.5))

        nn_model.add(Dense(1, kernel_initializer='normal'))
        nn_model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        return nn_model

    def nn_model(self):
        dataset = get_data('features_lgb_2.csv')

        train_data = dataset[dataset['score'] > 0.0]
        test_data = dataset[dataset['score'] < 0]
        y_data = train_data['score']
        x_data = train_data.drop(columns=['id', 'score'])

        test_data.reset_index(inplace=True, drop=True)
        x_test = test_data.drop(columns=['id', 'score'])

        baseline_model = self._get_nn_base_model()
        # estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)

        # kfold = KFold(n_splits=5)
        # mae = make_scorer(mean_absolute_error)
        # res = cross_val_score(estimator, X=x_data, y=y_data, cv=kfold, scoring=mae)
        # mae_error_cv = np.mean(res)
        # print(f'mae error cv: {mae_error_cv}')

        # estimator.fit(x_data, y_data)
        # y_pred = estimator.predict(x_data)
        # mae_error = mean_absolute_error(y_pred, y_data)

        # print(f'mae error: {mae_error}')
        # print(f'nn score: {1 / (1 + mae_error)}')

        baseline_model.fit(x_data,y_data,epochs=10,batch_size=5)
        y_pred = baseline_model.predict(x_data)
        mae_error = mean_absolute_error(y_pred, y_data)

        print(f'mae error: {mae_error}')
        print(f'nn score: {1 / (1 + mae_error)}')
        # pred = estimator.predict(x_test)
        # sub = pd.DataFrame({'id': test_data['id'], 'score': pred})
        # sub['score'] = sub['score'].apply(lambda item: int(round(item)))
        # sub.to_csv('submittion_5.csv', index=False)


nn = NNModels()
nn.nn_model()