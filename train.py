import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "data"
# Only train on these signs for now, even though data/ has more recorded
# (Please, Help, Sorry, Stop, Water, Food) -- add them back here later.
actions = ["Hello", "Yes", "No", "Thanks", "ILoveYou", "Good"]

X, y = [], []

for label, action in enumerate(actions):
    for seq in os.listdir(os.path.join(DATA_PATH, action)):
        sequence = []
        for frame in range(30):
            file_path = os.path.join(DATA_PATH, action, seq, f"{frame}.npy")
            if os.path.exists(file_path):
                sequence.extend(np.load(file_path))
        if len(sequence) == 63*30:
            X.append(sequence)
            y.append(label)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(actions, "labels.pkl")

print("Training complete!")