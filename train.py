import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score

df_full = pd.read_csv('data/gym_members_dataset.csv')
df = df_full.drop(columns=['Member_ID', 'Name', 'Gender', 'Address', 'Phone_Number', 'Avg_Calories_Burned'])

df['Join_Date'] = pd.to_datetime(df['Join_Date'])
df['Join_Day'] = df['Join_Date'].dt.day
df['Join_Month'] = df['Join_Date'].dt.month
df['Join_Year'] = df['Join_Date'].dt.year

df['Last_Visit_Date'] = pd.to_datetime(df['Last_Visit_Date'])
df['Last_Visit_Day'] = df['Last_Visit_Date'].dt.day
df['Last_Visit_Month'] = df['Last_Visit_Date'].dt.month
df['Last_Visit_Year'] = df['Last_Visit_Date'].dt.year

df = df.drop(columns=['Last_Visit_Date', 'Join_Date'])

X = df.drop(columns=['Churn']).values
y = df['Churn'].values

y = pd.Series(y).map({'No': 0, 'Yes': 1}).values

X_encoded = X.copy()
le = LabelEncoder()

X_encoded[:, 1] = le.fit_transform(X[:, 1])
X_encoded[:, 2] = le.fit_transform(X[:, 2])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
parameters = {
    'max_depth': [1, 2, 3],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

clf = GridSearchCV(estimator=model,
                   param_grid=parameters,
                   cv=5,
                   scoring='accuracy',
                   refit=True,
                   return_train_score=True
                   )
clf.fit(X_train, y_train)

print(f"Лучшие параметры: {clf.best_params_}")
print(f"Лучшая точность: {clf.best_score_:.4f}")

y_pred = clf.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Выявленно: {y_pred.mean() * 100:.0f}%')
print(f'Precision: {precision * 100:.0f}%')
print(f'Recall: {recall * 100:.0f}%')

joblib.dump(clf, 'model.pkl')
