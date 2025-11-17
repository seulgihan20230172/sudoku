import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import os

# ==============================================================================
# 1. CNN Part (from Sudoku_CNN_Stan.py using PyTorch)
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_input(grid):
    """CNN 모델에 맞게 스도쿠 퍼즐을 10x9x9 텐서로 인코딩합니다."""
    x = np.zeros((10, 9, 9), np.float32)
    for d in range(1, 10):
        x[d - 1] = (grid == d)
    x[9] = (grid > 0)
    return x

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(32, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SudokuCNNx15(nn.Module):
    def __init__(self, ch=512, depth=15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, ch), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[ConvBlock(ch) for _ in range(depth - 1)])
        self.head = nn.Conv2d(ch, 9, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

def load_cnn_model(path="sudoku_cnn.pth", ch=512, depth=15):
    """학습된 PyTorch CNN 모델을 로드합니다."""
    model = SudokuCNNx15(ch, depth).to(DEVICE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CNN model weights not found at {path}. Please train and save the CNN model first.")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

# ==============================================================================
# 2. DQN Part (from DQN.py using TensorFlow)
# ==============================================================================

def box_indices(r, c):
    br = (r // 3) * 3
    bc = (c // 3) * 3
    return br, bc

def is_valid_move(board, r, c, k):
    if board[r, c] != 0: return False
    if (k in board[r, :]) or (k in board[:, c]): return False
    br, bc = box_indices(r, c)
    if k in board[br:br + 3, bc:bc + 3]: return False
    return True

def is_solved(board):
    return np.all(board > 0) and all(set(board[i, :]) == set(range(1, 10)) for i in range(9))

def id_to_action(aid):
    i = aid // 81
    j = (aid % 81) // 9
    k = (aid % 9) + 1
    return i, j, k

def action_to_id(i, j, k):
    return i * 81 + j * 9 + (k - 1)

class SudokuEnv:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.state = None
        self.steps = 0
        self.max_steps = 81 - np.count_nonzero(puzzle)

    def reset(self):
        self.state = self.puzzle.copy()
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        i, j, k = id_to_action(action)
        
        if not is_valid_move(self.state, i, j, k):
            return self.state.copy(), -1.0, self.steps >= self.max_steps, {}

        self.state[i, j] = k
        if is_solved(self.state):
            return self.state.copy(), 1.0, True, {}
        
        done = self.steps >= self.max_steps
        return self.state.copy(), 0.0, done, {}

def build_qnet():
    inputs = keras.layers.Input(shape=(81,), dtype=tf.float32)
    x = keras.layers.Dense(256, activation="elu")(inputs)
    x = keras.layers.Dense(256, activation="elu")(x)
    outputs = keras.layers.Dense(729, activation=None)(x)
    return keras.Model(inputs, outputs)

def load_dqn_model(path="sudoku_dqn.h5"):
    """학습된 TensorFlow DQN 모델을 로드합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"DQN model not found at {path}. Please train and save the DQN model first.")
    return keras.models.load_model(path)

def legal_action_mask(board):
    mask = np.zeros(729, dtype=np.float32)
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0: continue
            for k in range(1, 10):
                if is_valid_move(board, i, j, k):
                    mask[action_to_id(i, j, k)] = 1.0
    return mask

def greedy_action(model, state):
    mask = legal_action_mask(state)
    s = state.reshape(1, -1) / 9.0
    q = model.predict(s, verbose=0)[0]
    q[mask == 0.0] = -1e9
    return int(np.argmax(q))

# ==============================================================================
# 3. Hybrid Solver
# ==============================================================================

def solve_with_hybrid_model(puzzle, cnn_model, dqn_model, blanks_for_dqn=6):
    """
    CNN으로 퍼즐을 풀다가 남은 빈칸이 특정 개수 이하일 때 DQN으로 전환합니다.
    """
    current_puzzle = puzzle.copy()
    print("--- Starting with CNN ---")

    while True:
        num_blanks = np.count_nonzero(current_puzzle == 0)
        if num_blanks <= blanks_for_dqn:
            print(f"\nSwitching to DQN. {num_blanks} blanks remaining.")
            break

        # CNN 예측
        with torch.no_grad():
            x = torch.from_numpy(encode_input(current_puzzle)).unsqueeze(0).to(DEVICE)
            logits = cnn_model(x)[0]  # (9, 9, 9)
            probs = torch.softmax(logits, dim=0)

        # 가장 확률 높은 빈칸 찾기
        best_prob, best_r, best_c, best_k = -1, -1, -1, -1
        
        blank_indices = np.argwhere(current_puzzle == 0)
        for r, c in blank_indices:
            
            prob_dist = probs[:, r, c]
            max_prob, k_idx = torch.max(prob_dist, dim=0)
            
            if max_prob.item() > best_prob:
                best_prob = max_prob.item()
                best_r, best_c, best_k = r, c, k_idx.item() + 1
        
        if best_r == -1:
            print("CNN could not find a valid move. Stopping.")
            break

        print(f"CNN fills ({best_r}, {best_c}) with {best_k} (Prob: {best_prob:.4f})")
        current_puzzle[best_r, best_c] = best_k

    # DQN으로 나머지 풀기
    print("\n--- Starting with DQN ---")
    
    dqn_env = SudokuEnv(current_puzzle)
    state = dqn_env.reset()
    
    while True:
        num_blanks = np.count_nonzero(state == 0)
        if num_blanks == 0:
            print("DQN finished.")
            break

        action = greedy_action(dqn_model, state)
        i, j, k = id_to_action(action)
        print(f"DQN fills ({i}, {j}) with {k}")
        
        state, reward, done, _ = dqn_env.step(action)

        if done:
            if not is_solved(state):
                 print("DQN could not solve the puzzle.")
            break
            
    return current_puzzle, state

# ==============================================================================
# 4. Main Execution
# ==============================================================================

def load_dataset(root="../data"):
    try:
        X_test = np.load(os.path.join(root, "test_puzzles.npy"))
        y_test = np.load(os.path.join(root, "test_solutions.npy"))
        return X_test, y_test
    except FileNotFoundError:
        print(f"Test data not found in {root}. Please check the path.")
        return None, None

def main():
    # 모델 로드
    try:
        cnn_model = load_cnn_model(path="sudoku_cnn.pth")
        dqn_model = load_dqn_model(path="sudoku_dqn.h5")
    except FileNotFoundError as e:
        print(e)
        print("Please run the training scripts for both models first.")
        return

    # 데이터셋 로드
    X_test, y_test = load_dataset()
    if X_test is None:
        return

    # 테스트할 퍼즐 선택
    puzzle_idx = np.random.randint(len(X_test))
    puzzle = X_test[puzzle_idx]
    solution = y_test[puzzle_idx]

    print("="*30)
    print(f"Solving Puzzle #{puzzle_idx}")
    print("="*30)
    print("Initial Puzzle:\n", puzzle)
    
    # 하이브리드 모델로 풀기
    intermediate_puzzle, final_puzzle = solve_with_hybrid_model(puzzle, cnn_model, dqn_model)
    
    print("\n" + "="*30)
    print("Results")
    print("="*30)
    print("Intermediate Puzzle (after CNN):\n", intermediate_puzzle)
    print("\nFinal Puzzle (after DQN):\n", final_puzzle)
    print("\nGround Truth Solution:\n", solution)

    if np.array_equal(final_puzzle, solution):
        print("\nSuccess! The final puzzle matches the solution.")
    else:
        print("\nFailure. The final puzzle does not match the solution.")

if __name__ == "__main__":
    main()
