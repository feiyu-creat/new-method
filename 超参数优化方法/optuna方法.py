import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



file1="../data/yes-no.xlsx"
# test_size=0.15
# random_state=20
aggregate_data = pd.read_excel(file1)

aggregate_feature = aggregate_data.drop(['plyindex','分档','target','Pmoments_i2','Pmoments_i4'], axis=1)  # 删除这些列的信息
aggregate_target = aggregate_data['target']
X = aggregate_feature#特征
y = aggregate_target#目标
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


def objective(trial, data=X, target=y):
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)
    param = {
        'tree_method': 'gpu_hist',
        # this parameter means using the GPU when training our model to speedup the training process
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),

        'n_estimators':trial.suggest_categorical('n_estimators',
                                                   [90, 100, 110, 115, 120, 125, 135, 145,155,160,165,170,175,180,185,190,195]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBClassifier(**param,use_label_encoder=False)

    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100, verbose=False)

    preds = model.predict(test_x)

    # rmse = mean_squared_error(test_y, preds, squared=False)
    acc=accuracy_score(test_y,preds)

    return acc
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

study.trials_dataframe()
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=['alpha',
                            #'max_depth',
                            'lambda',
                            'subsample',
                            'learning_rate',
                            'subsample'])
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_edf(study)



