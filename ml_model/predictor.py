import pickle
import pandas as pd
import re

model = pickle.load(open('ml_model//model_fin_lgb.sav', 'rb'))
MODEL_MAPE = 0.17601142926637764


def predict(sample):
    return model.predict(sample)


def create_sample(dict_user_data):
    dict_user_data = {i: j if j != 'None' else None for i, j in dict_user_data.items()}
    for ftr in ['appartment_metrage', 'kitchen_metrage', 'bathroom_metrage', 'floor']:
        # перевірка самостійно введених даних користувачем
        # у будь якій незрозумілій ситуації - значення фічі - None
        val = dict_user_data[ftr]
        new_val = re.sub('[^\d.]', '', val)
        try:
            new_val = float(new_val)
        except:
            new_val = None
        if new_val == 0:
            new_val = None
        dict_user_data[ftr] = new_val

    features = model.named_steps['preprocessor'].feature_names_in_
    sample = dict.fromkeys(features, None)
    sample.update(dict_user_data)
    return pd.Series(sample).to_frame().T
