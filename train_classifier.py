import pandas as pd, joblib, json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)
df = pd.read_csv("dataset/hand_gesture_landmarks.csv")
X, y = df.drop("label",axis=1), df["label"]
le = LabelEncoder(); y_enc = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(
    X,y_enc,test_size=0.2,stratify=y_enc,random_state=42)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=300,random_state=42))
])
pipe.fit(X_train,y_train)
print("Test Acc:", pipe.score(X_test,y_test))
joblib.dump(pipe,"models/gesture_rf.pkl")
inv = {i:lbl for i,lbl in enumerate(le.inverse_transform(range(len(le.classes_))))}
with open("models/labels_inv.json","w") as f:
    json.dump(inv,f)
