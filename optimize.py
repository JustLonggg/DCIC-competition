import os
import warnings

import lightgbm as lgb 
import numpy as np 
import pandas as pd 
import xgboost as xgb 
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from tree_regression import TreeRegression

warnings.filterwarnings(action='ignore')

TRAIN_PATH = 'F:/DataFountain/train_dataset'
TEST_PATH = 'F:/DataFountain/test_dataset'
ETLDATA_PATH = 'F:/DataFountain/feature_dataset'

class LGBOptimize(object):
    def __init__(self):
        self.x_data, self.y_data = self._get_xy()

    @staticmethod
    def _get_data():
        data_name = os.path.join(ETLDATA_PATH,'features_lgb_3.csv')
        df = pd.read_csv(data_name,header=0)
        df = df[df['score'] > 0]
        df.reset_index(inplace=True,drop=True)
        return df

    def _get_xy(self):
        dataset = self._get_data()
        
        remove_columns = ['id','score']
        x_columns = [column for column in dataset.columns if column not in remove_columns]

        x_data = dataset[x_columns]
        y_data = dataset['score']
        return (x_data,y_data)

    def optimize(self):

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 30,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        #     'seed': 4590,
        #     'verbose': -1
        # }
        # dtrain = lgb.Dataset(x_data, y_data)
        # cv_results = lgb.cv(params, dtrain, num_boost_round=10000, nfold=5, stratified=False, shuffle=True,
        #                     metrics='mae', early_stopping_rounds=200, verbose_eval=50, show_stdv=True, seed=4590)
        # print('best n_estimators:', len(cv_results['l1-mean']))
        # print('best cv score:', cv_results['l1-mean'][-1])

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 394,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 30,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        #     'random_state': 4590
        # }
        # grid_params = {'max_depth':[7],'num_leaves':[40]}
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error',cv=5,
        #                            verbose=1,n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 394,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 40,
        #     'max_depth': 7,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        #     'random_state': 4590
        # }
        # grid_params = {'min_child_samples':[48,50,52],'min_child_weight':[0,0.001,0.01]}
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error',cv=5,
        #                            verbose=1,n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        # params = {
        #      'boosting_type': 'gbdt',
        #      'objective': 'mae',
        #      'n_estimators': 394,
        #      'metric': 'mae',
        #      'learning_rate': 0.1,
        #      'min_child_samples': 48,
        #      'min_child_weight': 0,
        #      'num_leaves': 40,
        #      'max_depth': 7,
        #      'subsample': 0.8,
        #      'colsample_bytree': 0.8,
        #      'random_state': 4590
        # }
        # grid_params = {
        #     'subsample':[0.42],
        #     'colsample_bytree':[0.48]
        #     }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error',cv=5,
        #                            verbose=1,n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        # params = {
        #      'boosting_type': 'gbdt',
        #      'objective': 'mae',
        #      'n_estimators': 394,
        #      'metric': 'mae',
        #      'learning_rate': 0.1,
        #      'min_child_samples': 48,
        #      'min_child_weight': 0,
        #      'num_leaves': 40,
        #      'max_depth': 7,
        #      'subsample': 0.42,
        #      'colsample_bytree': 0.48,
        #      'random_state': 4590
        # }
        # grid_params = {
        #     'reg_alpha':[0.15],
        #     'reg_lambda':[5]
        #     }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error',cv=5,
        #                            verbose=1,n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        # params = {
        #      'boosting_type': 'gbdt',
        #      'objective': 'mae',
        #      'n_estimators': 319,
        #      'metric': 'mae',
        #      'learning_rate': 0.1,
        #      'min_child_samples': 20,
        #      'min_child_weight': 0,
        #      'num_leaves': 28,
        #      'max_depth': 5,
        #      'subsample': 0.7,
        #      'colsample_bytree': 0.8,
        #      'reg_alpha': 0.1,
        #      'reg_lambda': 2,
        #      'random_state': 4590
        # }
        # grid_params = {
        #     'learning_rate':[0.01,0.05,0.1]
        #     }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error',cv=5,
        #                            verbose=1,n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')


        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 10000,
        #     'metric': 'mae',
        #     'learning_rate': 0.01,
        #     'min_child_samples': 46,
        #     'min_child_weight': 0.01,
        #     'subsample_freq': 1,
        #     'num_leaves': 40,
        #     'max_depth': 7,
        #     'subsample': 0.42,
        #     'colsample_bytree': 0.48,
        #     'reg_alpha': 2,
        #     'reg_lambda': 0.1,
        #     'verbose': -1,
        #     'random_state': 4590
        # }
        # grid_params = {'subsample': [0.45, 0.5, 0.6],
        #     'colsample_bytree': [0.8, 0.9, 0.95]}
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        def LGB_CV(
            max_depth,
            num_leaves,
            min_child_samples,
            min_child_weight,
            subsample,
            colsample_bytree,
            reg_alpha,
            reg_lambda
            ):
            x_data = self.x_data
            y_data = self.y_data

            folder = KFold(n_splits=5,shuffle=True,random_state=4590)
            fold = folder.split(x_data,y_data)
            oof = np.zeros(x_data.shape[0])

            for train_index,vali_index in fold:
                k_x_train = x_data.loc[train_index]
                k_y_train = y_data.loc[train_index]
                k_x_vali = x_data.loc[vali_index]
                k_y_vali = y_data.loc[vali_index]

                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'n_estimators': 10000,
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    'min_child_samples': int(min_child_samples),
                    'min_child_weight': min_child_weight,
                    'subsample_freq': 1,
                    'num_leaves': int(num_leaves),
                    'max_depth': int(max_depth),
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda,
                    'verbose': -1,
                    'random_state': 4590
                }
                gbm = lgb.LGBMRegressor(**params)
                gbm = gbm.fit(k_x_train,k_y_train,eval_set=[(k_x_train,k_y_train),(k_x_vali,k_y_vali)],
                          early_stopping_rounds=50,verbose=False)
                #iteration_kwargs = TreeRegression._get_iteration_kwargs(gbm)
                k_pred = gbm.predict(k_x_vali,num_iteration=gbm.best_iteration_)
                oof[vali_index] = k_pred
            
            mae_error = metrics.mean_absolute_error(y_data,oof)
            return ((-1) * mae_error)

        LGB_BO = BayesianOptimization(LGB_CV,{
            'max_depth': (4,10),
            'num_leaves': (5,80),
            'min_child_samples': (20,60),
            'min_child_weight': (0,0.1),
            'subsample': (0.2,1.0),
            'colsample_bytree': (0.2,1.0),
            'reg_alpha': (0,6),
            'reg_lambda': (0,6)
        })
        LGB_BO.maximize(init_points=2,n_iter=4)
        print(LGB_BO.max)



class XGBOptimize(object):
    @staticmethod
    def _get_data():
        data_name = os.path.join(ETLDATA_PATH,'features_xgb_2.csv')
        df = pd.read_csv(data_name,header=0)
        df = df[df['score'] > 0]
        df.reset_index(inplace=True,drop=True)
        return df

    def optimize(self):
        dataset = self._get_data()

        remove_columns = ['id','score']
        x_columns = [column for column in dataset.columns if column not in remove_columns]

        x_data = dataset[x_columns]
        y_data = dataset['score']

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'reg:linear',
        #     'learning_rate': 0.1,
        #     'num_leaves': 50,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # dtrain = xgb.DMatrix(x_data,y_data)
        # cv_results = xgb.cv(params,dtrain,num_boost_round=1000,nfold=5,stratified=False,
        # shuffle=True,metrics='mae',early_stopping_rounds=50,verbose_eval=50,show_stdv=True,seed=20)
        # print('best n_estimators:', len(cv_results['test-mae-mean']))
        # print('best cv score:', cv_results['test-mae-mean'][-1])

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'reg:linear',
        #     'n_estimators': 154,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 50,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # grid_params = {'max_depth':[4,5,6],'num_leaves':[40]}
        # gbm = xgb.XGBRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                             n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'reg:linear',
        #     'n_estimators': 154,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 40,
        #     'max_depth': 5,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # grid_params = {'min_child_weight': [0, 1, 2]}
        # gbm = xgb.XGBRegressor(**params)
        # grid_search = GridSearchCV(gbm,param_grid=grid_params,scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                             n_jobs=5)
        # grid_search.fit(x_data,y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        def XGB_CV(
            max_depth,
            gamma,
            min_child_weight,
            subsample,
            colsample_bytree,
            reg_alpha,
            reg_lambda
            ):

            folder = KFold(n_splits=5,shuffle=True,random_state=4590)
            fold = folder.split(x_data,y_data)
            oof = np.zeros(x_data.shape[0])

            for train_index,vali_index in fold:
                k_x_train = x_data.loc[train_index]
                k_y_train = y_data.loc[train_index]
                k_x_vali = x_data.loc[vali_index]
                k_y_vali = y_data.loc[vali_index]

                params = {
                    'boosting_type': 'gbtree',
                    'objective': 'reg:linear',
                    #'metric': 'mae',
                    'n_estimators': 10000,
                    'gamma': gamma,
                    'learning_rate': 0.01,
                    'min_child_weight': min_child_weight,
                    'max_depth': int(max_depth),
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda,
                    'verbose': -1,
                    'random_state': 4590,
                    'n_jobs': 4,
                    'silent': True
                }
                gbm = xgb.XGBRegressor(**params)
                gbm = gbm.fit(k_x_train,k_y_train,eval_set=[(k_x_train,k_y_train),(k_x_vali,k_y_vali)],
                          early_stopping_rounds=50,verbose=False)
                #iteration_kwargs = TreeRegression._get_iteration_kwargs(gbm)
                k_pred = gbm.predict(k_x_vali,ntree_limit=gbm.best_ntree_limit)
                oof[vali_index] = k_pred
            
            mae_error = metrics.mean_absolute_error(y_data,oof)
            return ((-1) * mae_error)

        XGB_BO = BayesianOptimization(XGB_CV,{
            'max_depth': (4,10),
            'gamma': (0,1),
            'min_child_weight': (0,5),
            'subsample': (0.2,1.0),
            'colsample_bytree': (0.2,1.0),
            'reg_alpha': (0,6),
            'reg_lambda': (0,6)
        })
        XGB_BO.maximize(init_points=6,n_iter=6)
        print(XGB_BO.max)

if __name__ == '__main__':
    optimizer = LGBOptimize()
    optimizer.optimize()