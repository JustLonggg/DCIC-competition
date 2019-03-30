import time
from blending import Blending
from stacking import Stacking
from tree_regression import regression_main
from utils import get_blending_score
from utils import get_combinations
from utils import get_data
from utils import get_ensemble_score
from utils import get_score_array
from utils import get_values_by_index

if __name__ == '__main__':
    t0 = time.time()

    dataset = get_data('features_lgb_2.csv')
    train_data = dataset[dataset['score'] > 0]
    test_data = dataset[dataset['score'] < 0]

    train_data.reset_index(inplace=True,drop=True)
    test_data.reset_index(inplace=True,drop=True)

    train_score_df = train_data[['id','score']]
    test_score_df = test_data[['id']]

    oof_list = list()
    prediction_list = list()

    #mode_list = ['lgb']
    features_list = ['features_lgb_1.csv','features_lgb_2.csv']
    # for mode in mode_list:
    #     mode_score_name = f'{mode}_score'
    #     if mode == 'lgb':
    #         oof,prediction = regression_main(mode=mode,file_name='features_lgb_2.csv')
    #     else:
    #         oof,prediction = regression_main(mode=mode,file_name='features_xgb_2.csv')
    #     oof_list.append(oof)
    #     prediction_list.append(prediction)

    for i,file_name in enumerate(features_list):
        mode_score_name = 'lgb_{}_score'.format(i+1)
        oof,prediction = regression_main(mode='lgb',file_name=file_name)
        oof_list.append(oof)
        prediction_list.append(prediction)

        train_score = oof.tolist()
        train_score_df[mode_score_name] = train_score
        test_score_df[mode_score_name] = prediction

    # stacking 
    mode_list = ['1','2']
    combinations_list = get_combinations(range(len(oof_list)))
    for bin_item in combinations_list:
        oof = get_values_by_index(oof_list,bin_item)
        prediction = get_values_by_index(prediction_list,bin_item)

        mode = get_values_by_index(mode_list,bin_item)
        mode.append('score')
        mode_name = '_'.join(mode)

        stacking_oof,stacking_prediction = Stacking().get_stacking(oof,prediction,train_score_df['score'])
        train_score_df[mode_name] = stacking_oof
        test_score_df[mode_name] = stacking_prediction

    # blending 
    best_weight = Blending(train_score_df).get_best_weight()
    score_array = get_score_array(test_score_df)
    test_score_df['score'] = get_blending_score(score_array,best_weight)
    #test_score_df.to_csv('all_score.csv',index=False)

    sub_df = test_score_df[['id','score']]
    sub_df['score'] = sub_df['score'].apply(lambda x:int(round(x)))
    sub_df.to_csv('submission_lgb_1_2.csv',index=False)

    usage_time = time.time() - t0
    print(f'usage time : {usage_time}')