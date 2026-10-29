import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 3. Class distribution
# --------------------------------------------------

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 6. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 7. SMOTE
# --------------------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())
# --------------------------------------------------
# 8. Logistic Regression
# --------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

y_pred = logistic_model.predict(X_test_scaled)

y_prob = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 9. Precision-Recall Curve
# --------------------------------------------------

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure()

plt.plot(
    recall,
    precision
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig("precision_recall_curve.png")

plt.close()

print("\nPrecision-Recall curve saved.")


# --------------------------------------------------
# 10. Find best threshold using F1 score
# --------------------------------------------------

f1_scores = (
    2 * (precision * recall)
    / (precision + recall + 1e-10)
)

best_index = f1_scores[:-1].argmax()

best_threshold = thresholds[best_index]

print("\nBest threshold:", best_threshold)
print("Best F1 score:", f1_scores[best_index])


# --------------------------------------------------
# 11. Logistic Regression with best threshold
# --------------------------------------------------

y_pred_best = (
    y_prob >= best_threshold
).astype(int)

print("\nLogistic Regression - Best Threshold")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))


# --------------------------------------------------
# 12. Decision Tree
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)
tree_model.fit(
    X_train_scaled,
    y_train
)

tree_pred = tree_model.predict(X_test_scaled)


print("\n" + "=" * 50)
print("DECISION TREE")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))


# --------------------------------------------------
# 13. Random Forest
# --------------------------------------------------

forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

forest_model.fit(
    X_train_scaled,
    y_train
)

forest_pred = forest_model.predict(X_test_scaled)


print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, forest_pred))

print("\nClassification Report:")
print(classification_report(y_test, forest_pred))


# --------------------------------------------------
# 14. Random Forest with SMOTE
# --------------------------------------------------

forest_smote_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_smote_model.fit(
    X_train_smote,
    y_train_smote
)

forest_smote_pred = forest_smote_model.predict(
    X_test_scaled
)


print("\n" + "=" * 50)
print("RANDOM FOREST + SMOTE")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    forest_smote_pred
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    forest_smote_pred
))


# --------------------------------------------------
# 15. Model comparison
# --------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score


models = {
    "Logistic Regression": y_pred_best,
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred,
    "Random Forest + SMOTE": forest_smote_pred
}


results = []

for model_name, predictions in models.items():

    precision_value = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall_value = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1_value = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Precision": precision_value,
        "Recall": recall_value,
        "F1": f1_value
    })


comparison_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison_df)

# # Hyperparameter Tuning

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     "n_estimators": [50, 100],
#     "max_depth": [10, None]
# }

# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(
#         random_state=42,
#         class_weight="balanced"
#     ),
#     param_grid=param_grid,
#     scoring="f1",
#     cv=2,
#     n_jobs=-1,
#     verbose=1
# )

# grid_search.fit(X_train_smote, y_train_smote)

# print("Best Parameters:", grid_search.best_params_)
# print("Best F1 Score:", grid_search.best_score_)
# Save the trained model and scaler

os.makedirs("models", exist_ok=True)

joblib.dump(forest_smote_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")
