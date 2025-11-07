import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Download latest version
path = kagglehub.dataset_download("rohanrao/sudoku")
#print("Path to dataset files:", path)

df = pd.read_csv(os.path.join(path,"sudoku.csv"))
os.makedirs("data",exist_ok=True)
index_path = "data/sample_indices.npy"

if os.path.exists(index_path):
    indices = np.load(index_path)
else:
    indices = np.random.RandomState(42).choice(len(df),size=1000,replace=False)
    np.save(index_path,indices)

df_sample = df.iloc[indices].reset_index(drop=True)

def str_to_grid(s):
    return np.array(list(map(int,s))).reshape(9,9)

puzzles = np.stack([str_to_grid(p) for p in df_sample["puzzle"]])
solutions = np.stack([str_to_grid(s) for s in df_sample["solution"]])

X_train, X_test,y_train,y_test = train_test_split(puzzles, solutions, test_size=0.2, random_state=42)
X_train,X_val,y_train,y_val = train_test_split(X_train, y_train,test_size=0.1, random_state=42)

np.save("data/train_puzzles.npy",X_train)
np.save("data/train_solutions.npy",y_train)
np.save("data/val_puzzles.npy",X_val)
np.save("data/val_solutions.npy",y_val)
np.save("data/test_puzzles.npy",X_test)
np.save("data/test_solutions.npy",y_test)

print("Dataset split saved in /data/")
print(f"Train: {len(X_train)}, Val:{len(X_val)}, Test: {len(X_test)}")