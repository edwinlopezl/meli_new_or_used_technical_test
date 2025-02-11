"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
the proposed secondary metric and the performance achieved on that metrics. 
Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import warnings
warnings.filterwarnings('ignore')

#utils
import json
from ml_utils import ml
import pandas as pd 
import re
import numpy as np

#sklearn utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

#models
import lightgbm as lgb
from xgboost import XGBClassifier

#metrics
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix

#saving_model
import pickle

#os
import os
location = os.path.dirname(os.path.abspath(__file__))

#saving outputs
import sys
log_file = open(os.path.join(location, "new_or_used_output.txt"), "w")
sys.stdout = log_file


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():

    with open(os.path.join(location, '../data/inputs/MLA_100k_checked_v3.jsonlines'), 'r') as file:
        data = [json.loads(line) for line in file]

    print(' ')
    print('Starting -- Building Dataset...')
    df = pd.json_normalize(data)
    df.columns = df.columns.str.replace('.', '_')

    df = df.set_index('id')


    df = df.drop(
        [
            'site_id', 
            'listing_source', 
            'international_delivery_mode', 
            'differential_pricing', 
            'permalink',
            'descriptions'
        ]
        , axis=1
    ) 

    df = ml.expand_nested_fields(
        df, 
        [
            'non_mercado_pago_payment_methods', 
            'pictures', 
            'attributes', 
            'variations', 
            'shipping_free_methods'
        ]
    )

    df.drop(columns=['shipping_free_methods_rule_value', 'variations_attribute_combinations'], inplace = True)

    df = ml.expand_list_fields(df, ['tags', 'sub_status', 'shipping_tags'])

    df['official_store_id'] = df['official_store_id'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['video_id'] = df['video_id'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['catalog_product_id'] = df['catalog_product_id'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['deal_ids'] = df['deal_ids'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['parent_item_id'] = df['parent_item_id'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['thumbnail'] = df['thumbnail'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['secure_thumbnail'] = df['secure_thumbnail'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')
    df['seller_id'] = df['seller_id'].apply(lambda x: 'yes' if pd.notnull(x) else 'no')


    top10 = df['category_id'].value_counts().head(10).index
    df['category_id'] = df['category_id'].apply(lambda x: x if x in top10 else 'OTHERS')

    top10 = df['attributes_value_id'].value_counts().head(10).index
    df['attributes_value_id'] = df['attributes_value_id'].apply(lambda x: x if x in top10 else 'OTHERS')

    top10 = df['seller_address_city_id'].value_counts().head(10).index
    df['seller_address_city_id'] = df['seller_address_city_id'].apply(lambda x: x if x in top10 else 'OTHERS')

    df['warranty'] = df['warranty'].apply(ml.normalize_text)


    df['warranty'] = np.where(
        df['warranty'].isnull(), 
        'no', 
        np.where(
            df['warranty'] == 'si',
            'yes',
            np.where(
                df['warranty'].str.contains('sin', na=False),
                'no',
                np.where(
                    df['warranty'].str.contains('mes', na=False),
                    'meses',
                    np.where(
                        df['warranty'].str.contains('comentarios', case=False, na=False),
                        'no',
                        np.where(
                            df['warranty'].str.contains('garantia', case=False, na=False),
                            'yes',
                            'otros'
                        )
                    )
                )
            )
        )
    )


    # Transforming bools into yes/no to give all the variables the same treatments
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].applymap(lambda x: 'yes' if x else 'no')

    # Replacing empty lists for np.nan
    df = df.replace('[]', np.nan)
    df = df.applymap(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)

    #-Droping - Argument: irrelevant or very hard to understand information
    df = df.drop(
        columns = [ 
            'seller_address_country_id', 
            'seller_address_state_name', 
            'title', 
            'pictures_url', 
            'pictures_secure_url', 
            'pictures_id',
            'pictures_size',
            'pictures_max_size',
            'variations_picture_ids',
            'seller_address_city_name',
            'variations_seller_custom_field',
            'attributes_id'
        ]
    )

    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['date_created'] = pd.to_datetime(df['date_created'])

    df = ml.expand_datetime_columns(df, ['last_updated', 'date_created', 'stop_time', 'start_time'])


    for col in df.columns:
        if df[col].apply(type).nunique() > 1:
            df[col] = df[col].astype(object)

    cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=cols_to_drop)

    threshold = 300
    cat_columns = df.select_dtypes(include=['object']).columns
    cols_to_drop = [col for col in cat_columns if df[col].nunique() > threshold]
    df.drop(columns=cols_to_drop, inplace=True)


    df.drop(
        columns=['price', 'start_time_minute', 'available_quantity'],
        inplace=True
    )

    print(f'Final DF shape: {df.shape}')


    # Separating categorical and numerical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Transforming categorical into dummies
    df_cat_dummies = pd.get_dummies(df[categorical_cols], drop_first=True)

    # Scaling numerical (Choosed minmax)
    scaler = MinMaxScaler()
    df_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index
    )

    # Join the DF
    df = pd.concat([df_numeric_scaled, df_cat_dummies], axis=1)

    print(f'Used perc.: {100*df.condition_used.sum()/ df.shape[0]}')



    target = "condition_used"
    X = df.drop(columns=[target])
    y = df[target]
    porc_test = 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=porc_test, random_state=42, stratify=y
    )
    print('Building OK!')
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')
    print(' ')
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    print(' ')
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    # ...


    ### XGBOOST
    print(' ')
    print('XGB Calibration -- Starts...')

    param_grid_xgb = {
        'n_estimators': [200, 300, 400],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'subsample': [0.7, 0.8, 1.0]
    }

    xgb_clf = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )

    grid_search_xgb = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid_xgb,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search_xgb.fit(X_train, y_train)

    print(' ')
    print("Best parameters founded: XGB:")
    print(grid_search_xgb.best_params_)

    print(' ')
    print("CV XGB Train (Accuracy):")
    print(grid_search_xgb.best_score_)

    best_model_xgb = grid_search_xgb.best_estimator_

    #### TEST XGBOOST

    y_pred_xgb = best_model_xgb.predict(X_test)
    y_pred_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1] 

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_pr_xgb = average_precision_score(y_test, y_pred_proba_xgb)

    print(' ')
    print("TEST: XGB Accuracy:", acc_xgb)
    print("TEST: XGB AUC-PR:", auc_pr_xgb)

    print(' ')
    print("TEST: XGB - Classification report:")
    print(classification_report(y_test, y_pred_xgb))

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print(' ')
    print("TEST: XGB - Confussion Matrix:")
    print(cm_xgb)


    ### LGBM
    print(' ')
    print('LGBM Calibration -- Starts...')
    param_grid_lgbm = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],  
        'colsample_bytree': [0.7, 0.8, 1.0],
        'subsample': [0.7, 0.8, 1.0]
    }

    lgbm_clf = lgb.LGBMClassifier(
        random_state=42,
        metric='binary_logloss'
    )

    grid_search_lgbm = GridSearchCV(
        estimator=lgbm_clf,
        param_grid=param_grid_lgbm,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search_lgbm.fit(X_train, y_train)

    print(' ')
    print("Best parameters founded: LGBM:")
    print(grid_search_lgbm.best_params_)

    print(' ')
    print("CV LGBM Train (Accuracy):")
    print(grid_search_lgbm.best_score_)

    best_model_lgbm = grid_search_lgbm.best_estimator_

    y_pred_lgbm = best_model_lgbm.predict(X_test)
    y_pred_proba_lgbm = best_model_lgbm.predict_proba(X_test)[:, 1] 

    acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
    auc_pr_lgbm = average_precision_score(y_test, y_pred_proba_lgbm)

    print(' ')
    print("TEST: LGBM Accuracy:", acc_lgbm)
    print("TEST: LGBM AUC-PR:", auc_pr_lgbm)

    print(' ')
    print("TEST: LGBM - Classification report:")
    print(classification_report(y_test, y_pred_lgbm))

    cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
    print(' ')
    print("TEST: LGBM - Confussion Matrix:")
    print(cm_lgbm)


    ### FINAL STEP -- CHOOSING THE MODEL
    print(' ')
    if acc_xgb >= acc_lgbm:
        print('Best model: XGB')
        print(f'Accuracy: {acc_xgb}')
        print(f'AUC-PR: {auc_pr_xgb}')
        final_model = best_model_xgb
    else:
        print('Best model: LGBM')
        print(f'Accuracy: {acc_lgbm}')
        print(f'AUC-PR: {auc_pr_lgbm}')
        final_model = best_model_lgbm 

    with open(os.path.join(location, '../resources/pkl/final_model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)
    print('OK! Model saved. End of script')

sys.stdout = sys.__stdout__
log_file.close()
