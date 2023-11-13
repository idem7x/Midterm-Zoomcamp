#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
import xgboost as xgb

df = pd.read_csv('./data/HotelReservations.csv')
df.head()

df.columns = df.columns.str.replace(' ', '_').str.lower()
df.head()

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
df.head().T
df.avg_price_per_room = pd.to_numeric(df.avg_price_per_room, errors='coerce').astype(int)

df['arrival_year'] = df['arrival_year'].map({2017: 0, 2018: 1})
df.booking_status = (df.booking_status == 'canceled').astype(int)
df.head().T

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.booking_status.values
y_val = df_val.booking_status.values
y_test = df_test.booking_status.values

del df_train['booking_status']
del df_val['booking_status']
del df_test['booking_status']

df_full_train = df_full_train.reset_index(drop=True)
df_full_train.isnull().sum()
df_full_train.booking_status.value_counts(normalize=True)
df_full_train.dtypes

numerical = list(df.select_dtypes(include=[np.int64]).drop(columns=['booking_status']).columns)
numerical
categorical = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
global_booking_status = df_full_train.booking_status.mean()
global_booking_status

for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).booking_status.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_booking_status
    df_group['risk'] = df_group['mean'] / global_booking_status
    print()
    print()

correlation_matrix = df_train[numerical].corr().abs()

plt.figure(figsize=(12, 10)) 
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

df_full_train[numerical].corrwith(df_full_train.booking_status).abs()

# Create an empty list to store the results
results = []

# Define the interval size
interval = 10

# Iterate through the intervals
for start_day in range(0, 366, interval):
    end_day = start_day + interval

    # Filter the DataFrame for the current interval
    filtered_df = df_full_train[(df_full_train['lead_time'] > start_day) & (df_full_train['lead_time'] <= end_day)]

    # Calculate the mean of the 'booking_status' for the current interval
    mean_booking_status = round(filtered_df['booking_status'].mean(), 2)

    # Count the '0' (not canceled) and '1' (canceled) values for the current interval
    not_canceled_count = (filtered_df['booking_status'] == 0).sum()
    canceled_count = (filtered_df['booking_status'] == 1).sum()

    # Append the results to the list
    results.append([start_day, end_day, mean_booking_status, not_canceled_count, canceled_count])

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results, columns=['Start Day', 'End Day', 'Mean', 'not_canceled_count', 'canceled_count'])

# Print the results DataFrame
print(results_df)

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]
cancel_decision = (y_pred >= 0.5)
round((y_val == cancel_decision).mean(), 2)


fpr, tpr, thresholds = roc_curve(y_val, y_pred)

plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()

auc(fpr, tpr)

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []
    rmse_scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.booking_status.values
        y_val = df_val.booking_status.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
        
        rmse = sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(round(rmse, 3))

    print('C=%s %.3f +- %.3f %.3f' % (C, np.mean(scores), np.std(scores), np.mean(rmse_scores)))


dv, model = train(df_full_train, df_full_train.booking_status.values, C=0.5)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
    
print("Logistic regression candidate with C=0.5 AUC=%.3f RMSE=%.3f" % (auc, rmse))

client = {
"no_of_adults": 2,
"no_of_children":  0,
"no_of_weekend_nights": 1,
"no_of_week_nights": 2,
"type_of_meal_plan": "meal_plan_1",
"required_car_parking_space": 0,
"room_type_reserved": "room_type_1",
"lead_time": 224,
"arrival_year": 0,
"arrival_month": 10,
"arrival_date": 2,
"market_segment_type": "offline",
"repeated_guest": 0,
"no_of_previous_cancellations": 0,
"no_of_previous_bookings_not_canceled": 0,
"avg_price_per_room": 65,
"no_of_special_requests": 0
}

df.head()
X_client = dv.transform([client])
model.predict_proba(X_client)[0, 1]


with open('./data/model-logistic.bin', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump(model, f_out)


with open('./data/dv-logistic.bin', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump(dv, f_out)

def train(df_train, y_train, alpha=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = Ridge(alpha=alpha, solver='sag', random_state=42)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

alpha_values = [0, 0.01, 0.1, 1, 10]
rmse_scores = []
auc_scores = []

for alpha in tqdm(alpha_values):
    y_train = df_train.booking_status.values
    y_val = df_val.booking_status.values

    dv, model = train(df_train, y_train, alpha=alpha)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)
    
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(round(rmse, 3))

for alpha, rmse, auc in zip(alpha_values, rmse_scores, auc_scores):
    print(f"Alpha = {alpha}: RMSE = {rmse} : AUC = {auc}")

dv, model = train(df_full_train, df_full_train.booking_status.values, alpha=1)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
    
print("Ridge regression candidate with alpha=1.0 AUC=%.3f RMSE=%.3f" % (auc, rmse))

X_client = dv.transform([client])
model.predict(X_client)[0]

with open('./data/model-ridge.bin', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump(model, f_out)

with open('./data/dv-ridge.bin', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump(dv, f_out)

dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(df_train[categorical + numerical].to_dict(orient='records'))
X_val = dv.transform(df_val[categorical + numerical].to_dict(orient='records'))
X_test = dv.transform(df_test[categorical + numerical].to_dict(orient='records'))


model = RandomForestRegressor()
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
feature_names = dv.get_feature_names_out()
feature_importance_list = list(zip(feature_names, feature_importances))

# Sort the list by importance in descending order
feature_importance_list.sort(key=lambda x: x[1], reverse=True)

# Print the feature names and their importances
for feature, importance in feature_importance_list:
    print(f"{feature}: {importance}")

rmse_scores = []
auc_scores = []

for n in tqdm(range(10, 201, 20)):
    model = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    rmse_scores.append((n, rmse))
    auc_scores.append((n, auc))


df_scores = pd.DataFrame(rmse_scores, columns=['n_estimators', 'rmse'])

plt.plot(df_scores['n_estimators'], df_scores['rmse'])
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.title('RMSE vs. Number of Estimators')
plt.show()


# In[203]:


df_scores = pd.DataFrame(auc_scores, columns=['n_estimators', 'auc'])

plt.plot(df_scores['n_estimators'], df_scores['auc'])
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('AUC')
plt.title('AUC vs. Number of Estimators')
plt.show()


# In[204]:


# scores = []

# for d in tqdm([1, 5, 10, 15, 20, 25]):
#     mean_rmse_values = []
#     for n in tqdm(range(10, 201, 20)):
#         rf = RandomForestRegressor(n_estimators=n,
#                                     max_depth=d,
#                                     random_state=1)
#         rf.fit(X_train, y_train)

#         y_pred = rf.predict(X_val)
#         rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#         mean_rmse_values.append(rmse)
        
#     mean_rmse = np.mean(mean_rmse_values)
#     scores.append((d, mean_rmse))


# In[205]:


# df_scores = pd.DataFrame(scores, columns=['max_depth', 'mean_rmse'])
# df_scores


model = RandomForestRegressor(n_estimators=200,
                                    max_depth=20,
                                    random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
    
print("RandomForest regression candidate AUC=%.3f RMSE=%.3f" % (auc, rmse))

bad_client = {
"no_of_adults": 2,
"no_of_children":  0,
"no_of_weekend_nights": 0,
"no_of_week_nights": 2,
"type_of_meal_plan": "meal_plan_1",
"required_car_parking_space": 0,
"room_type_reserved": "room_type_1",
"lead_time": 211,
"arrival_year": 1,
"arrival_month": 5,
"arrival_date": 20,
"market_segment_type": "online",
"repeated_guest": 0,
"no_of_previous_cancellations": 0,
"no_of_previous_bookings_not_canceled": 0,
"avg_price_per_room": 100,
"no_of_special_requests": 0
}


X_client = dv.transform(client)
print(X_client.toarray())
model.predict(X_client)[0]

with open('./data/model-randomforest.pkl', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump(model, f_out)

with open('./data/dv-randomforest.pkl', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump(dv, f_out)


dtrain = xgb.DMatrix(X_train, label=y_train)

dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, 'train'), (dval, 'validation')]

# Train models with different eta values
num_round = 100  # Number of boosting rounds
scores = []
for eta in tqdm([0.1, 0.3, 0.5, 0.7, 0.9]):
    mean_rmse_values = []
    for depth in tqdm([3, 5, 7, 9]):
        xgb_params = {
           'eta': eta,
           'max_depth': depth,
           'min_child_weight': 1,
           'objective': 'reg:squarederror',
           'nthread': 8,
           'seed': 1,
           'verbosity': 0
        }
        model = xgb.train(xgb_params, dtrain, num_round, watchlist)
        rmse = float(model.eval(dval).split(":")[1])
        mean_rmse_values.append(round(rmse,3))
        print(f"ETA = {eta}, depth = {depth} :  RMSE = {rmse}")
    print(mean_rmse_values)
    mean_rmse = np.mean(mean_rmse_values)
    scores.append((eta, mean_rmse))

df_scores = pd.DataFrame(scores, columns=['eta', 'mean_rmse'])
df_scores


dtest = xgb.DMatrix(X_test, label=y_test)
watchlist_test = [(dtrain, 'train'), (dtest, 'validation')]

xgb_params_test = {
           'eta': 0.1,
           'max_depth': 9,
           'min_child_weight': 1,
           'objective': 'reg:squarederror',
           'nthread': 8,
           'seed': 1,
           'verbosity': 0
        }

model = xgb.train(xgb_params, dtrain, num_round, watchlist_test)
rmse = float(model.eval(dtest).split(":")[1])
rmse
