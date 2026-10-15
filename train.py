import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset/tourism.csv").drop_duplicates().dropna()          # clean
CAT = ["category", "season"]
NUM = ["rating", "reviews", "avg_cost", "temperature", "accessibility", "festival"]
X, y = df[CAT + NUM], df["popularity"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

pipe = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
acc = accuracy_score(yte, pred)
cv = cross_val_score(pipe, X, y, cv=5).mean()
print(f"Test accuracy: {acc:.3f} | 5-fold CV: {cv:.3f}\n", classification_report(yte, pred))
labels = ["Low", "Medium", "High"]
names = pipe["prep"].get_feature_names_out()
imp = dict(sorted(zip([n.split("__")[1] for n in names], pipe["rf"].feature_importances_.round(4).tolist()), key=lambda t: -t[1]))
joblib.dump(pipe, "model/model.pkl")
json.dump({"accuracy": acc, "cv_accuracy": cv, "labels": labels, "importance": imp,
           "confusion": confusion_matrix(yte, pred, labels=labels).tolist()}, open("model/metrics.json", "w"))
print("Saved model/model.pkl")
