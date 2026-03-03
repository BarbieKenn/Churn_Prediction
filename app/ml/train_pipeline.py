import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from app.ml.transformer import Transformer


df_full = pd.read_csv('../../data/gym_members_dataset.csv')
df = df_full.drop(columns=['Member_ID', 'Name', 'Gender', 'Address', 'Phone_Number', 'Avg_Calories_Burned', 'Total_Weight_Lifted_kg'])

ref_train = pd.to_datetime(df["Last_Visit_Date"]).max().normalize()
df["Reference_Date"] = ref_train

X = df.drop(columns=['Churn'])
y = df['Churn'].values
y = pd.Series(y).map({'No': 0, 'Yes': 1}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.columns)

num_columns = ['Age', 'Avg_Workout_Duration_Min', 'Visits_Per_Month', 'Days_Since_Last_Visit']
cat_columns = ['Membership_Type', 'Favorite_Exercise']

model = RandomForestClassifier(n_estimators=100, random_state=42)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num', num_pipe, num_columns),
    ('cat', cat_pipe, cat_columns)
])

pipe = Pipeline([
    ('feat', Transformer()),
    ('preprocess', preprocess),
    ('model', model)
])


parameters = {
    'model__max_depth': [1, 3, 5, 8, 12],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

grid = GridSearchCV(pipe,
                   param_grid=parameters,
                   cv=5,
                   scoring='f1',
                   refit=True,
                   return_train_score=True
    )

grid.fit(X_train, y_train)

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучшая точность: {grid.best_score_:.2f}")

model_artifact = {
    "model": grid.best_estimator_,
    "features": list(X_train.columns),
    "best_params": grid.best_params_,
    "cv_score": grid.best_score_,
}

joblib.dump(model_artifact, "../../model.joblib")

importance = grid.best_estimator_.named_steps['model'].feature_importances_
features = grid.best_estimator_.named_steps['preprocess'].get_feature_names_out()
robustness = pd.DataFrame({
    "feature": features,
    "importance": importance
})
print(robustness.nlargest(10, ['importance']))

feat_out = pipe.named_steps["feat"].transform(X.copy())
print("FEAT OUT:", feat_out.iloc[0].to_dict())

# 2) что идёт в preprocess после отбора колонок
print("COLUMNS AFTER FEAT:", feat_out.columns.tolist())

print(df.groupby("Churn")["Visits_Per_Month"].describe())
