import os, math, random
from collections import deque
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="train or eval"
    )
    return parser.parse_args()

illegal_action_count = 0#불법 액션 탐지
# =========================
# 1) 데이터 로드 (공통 세트)
# =========================
def load_dataset(root="data"):
    X_train = np.load(os.path.join(root, "../data/train_puzzles.npy"))
    y_train = np.load(os.path.join(root, "../data/train_solutions.npy"))  # RL에선 안 씀(검증용 가능)
    X_val   = np.load(os.path.join(root, "../data/val_puzzles.npy"))
    y_val   = np.load(os.path.join(root, "../data/val_solutions.npy"))
    X_test  = np.load(os.path.join(root, "../data/test_puzzles.npy"))
    y_test  = np.load(os.path.join(root, "../data/test_solutions.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test

# =========================
# 2) 스도쿠 유틸/검증 함수
# =========================
def box_indices(r, c):
    br = (r // 3) * 3
    bc = (c // 3) * 3
    return br, bc

def is_valid_move(board, r, c, k):
    """보드에서 (r,c)에 k를 놓아도 되는지(규칙 위반 여부)"""
    if board[r, c] != 0:  # 이미 채워진 칸
        return False
    # 행/열 중복 체크
    if (k in board[r, :]) or (k in board[:, c]):
        return False
    # 3x3 박스 중복 체크
    br, bc = box_indices(r, c)
    if k in board[br:br+3, bc:bc+3]:
        return False
    return True

def is_solved(board):
    """빈칸 없고 모든 규칙 만족하면 True"""
    if np.any(board == 0):
        return False
    # 행/열/박스 각각 1~9 집합 확인
    target = set(range(1, 10))
    for i in range(9):
        if set(board[i, :]) != target: return False
        if set(board[:, i]) != target: return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(board[br:br+3, bc:bc+3].reshape(-1)) != target: return False
    return True

# =========================
# 3) 액션 인코딩/마스킹
# =========================
# action id ∈ [0, 728]  ↔  (i, j, k) with i,j∈[0..8], k∈[1..9]
def id_to_action(aid):
    i = aid // 81
    j = (aid % 81) // 9
    k = (aid % 9) + 1
    return i, j, k

def action_to_id(i, j, k):
    return i * 81 + j * 9 + (k - 1)

def legal_action_mask(board):
    """729 길이의 0/1 마스크 반환: 합법이면 1, 불법이면 0"""
    mask = np.zeros(729, dtype=np.float32)
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:  # 이미 채워진 칸이면 해당 칸 관련 액션 모두 불가
                continue
            for k in range(1, 10):
                aid = action_to_id(i, j, k)
                if is_valid_move(board, i, j, k):
                    mask[aid] = 1.0
    return mask

# =========================
# 4) 환경 (Gym 스타일)
# =========================
class SudokuEnv:
    def __init__(self, puzzles):
        self.puzzles = puzzles
        self.state = None
        self.steps = 0
        self.max_steps = 80

    def reset(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.puzzles))
        self.state = self.puzzles[idx].copy()
        self.steps = 0
        return self.state

    def step(self, action):
        """action: 0..728 -> (i,j,k)"""
        self.steps += 1
        i, j, k = id_to_action(action)
        

        # 불법 행동
        if not is_valid_move(self.state, i, j, k):
            return self.state.copy(), -2.0, False, {}

        # 합법 → 놓기
        prev_filled = np.count_nonzero(self.state)
        self.state[i, j] = k
        new_filled = np.count_nonzero(self.state)
        '''
        fill_bonus = 0.05* (new_filled - prev_filled)#수정 보드에 칸 채워질때
        step_reward = 0.03
        reward = fill_bonus + step_reward
        '''
        reward = -0.01
        # 완성 보상/종료
        if is_solved(self.state):
            reward += 5.0
            return self.state.copy(), reward, True, {}

        # 스텝 초과로 에피소드 중단
        done = self.steps >= self.max_steps
        return self.state.copy(), (reward + 0.1) , done, {}

# =========================
# 5) DQN 네트워크 (CNN 없이)
# =========================
def build_qnet():
    # 입력: 81차원(0~9). 간단히 정규화해서 사용.
    inputs = keras.layers.Input(shape=(81,), dtype=tf.float32)
    x = inputs  #정규화는 이미 함
    x = keras.layers.Dense(256, activation="elu")(x)
    x = keras.layers.Dense(256, activation="elu")(x)
    outputs = keras.layers.Dense(729, activation=None)(x)  # 각 액션의 Q값
    model = keras.Model(inputs, outputs)
    return model

# =========================
# 6) 에이전트: ε-탐욕 정책 + 리플레이 버퍼
# =========================
def epsilon_greedy_action(model, state, mask, epsilon):
    """마스크 기반 ε-탐욕. 불법 액션은 Q=-1e9로 막음."""
    if np.random.rand() < epsilon:
        legal_ids = np.where(mask > 0.0)[0]
        if len(legal_ids) == 0:  # 움직일 수 없으면 무작위 반환(실제로 거의 없음)
            return np.random.randint(729)
        return int(np.random.choice(legal_ids))
    # greedy
    s = state.reshape(1, -1) / 9.0
    q = model.predict(s, verbose=0)[0]  # (729,)
    q_masked = q.copy()
    q_masked[mask == 0.0] = -1e9
    return int(np.argmax(q_masked))

# =========================
# 7) 학습 구성요소
# =========================
batch_size = 64
gamma = 0.95
optimizer = keras.optimizers.Nadam(learning_rate=1e-4)
loss_fn = keras.losses.MeanSquaredError()

replay_buffer = deque(maxlen=8000)

def play_one_step(env, model, state, epsilon):
    mask = legal_action_mask(state)
    action = epsilon_greedy_action(model, state.reshape(-1), mask, epsilon)
    next_state, reward, done, info = env.step(action)
    # 경험 저장: (state, action, reward, next_state, done, mask_next)
    next_mask = legal_action_mask(next_state)
    replay_buffer.append((
        state.copy(), action, reward, next_state.copy(), done, next_mask
    ))
    return next_state, reward, done

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[i] for i in indices]
    states      = np.array([b[0].reshape(-1)/9.0 for b in batch], dtype=np.float32)  # (B,81)
    actions     = np.array([b[1] for b in batch], dtype=np.int32)                    # (B,)
    rewards     = np.array([b[2] for b in batch], dtype=np.float32)                  # (B,)
    next_states = np.array([b[3].reshape(-1)/9.0 for b in batch], dtype=np.float32)  # (B,81)
    dones       = np.array([b[4] for b in batch], dtype=np.float32)                  # (B,)
    next_masks  = np.array([b[5] for b in batch], dtype=np.float32)                  # (B,729)
    return states, actions, rewards, next_states, dones, next_masks

def training_step(model, target_model):
    states, actions, rewards, next_states, dones, next_masks = sample_experiences(batch_size)

    # 다음 상태의 Q값 (target net)
    next_q = target_model(next_states, training=False).numpy()  # (B,729)
    # 불법 액션 마스킹
    for i in range(len(next_q)):
        next_q[i][next_masks[i] == 0.0] = -1e9
    max_next_q = np.max(next_q, axis=1)  # (B,)

    # DQN 타깃
    target_q_for_actions = rewards + (1.0 - dones) * gamma * max_next_q  # (B,)

    with tf.GradientTape() as tape:
        all_q = model(states, training=True)                  # (B,729)
        # 선택한 action의 Q만 추출 (one-hot gather)
        action_onehot = tf.one_hot(actions, 729, dtype=tf.float32)  # (B,729)
        q_selected = tf.reduce_sum(all_q * action_onehot, axis=1)   # (B,)
        loss = tf.reduce_mean(loss_fn(target_q_for_actions, q_selected))
        '''
        if loss > 1e6:
            print("LOSS EXPLODED:", loss)
            print("targets:", target_q_for_actions[:10])
            print("q_selected:", q_selected[:10])
        '''

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g,1.0) for g in grads]#클리핑 추가함#exploding gd하기 않기 위함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)

# =========================
# 8) 학습 루프
# =========================
def train_rl_only(num_episodes=200, warmup_episodes=30, target_sync_every=15):
    X_train, *_ = load_dataset()
    env = SudokuEnv(X_train)

    qnet = build_qnet()
    target_qnet = build_qnet()
    target_qnet.set_weights(qnet.get_weights())  # 초기 동기화

    os.makedirs("checkpoints", exist_ok=True)

    eps_start, eps_end = 1.0, 0.05
    eps_decay_ep = max(1, num_episodes // 2)

    reward_hist, loss_hist = [], []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0

        # 에피소드마다 epsilon 선형 감소
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (episode / num_episodes)) #epsilon 너무 빨리 감소하지 않게 바꿈

        for step in range(env.max_steps):
            state, r, done = play_one_step(env, qnet, state, epsilon)
            ep_reward += r
            if done:
                break

        reward_hist.append(ep_reward)

        # 워밍업 이후부터 학습
        if episode > warmup_episodes and len(replay_buffer) >= batch_size:
            if episode%3 ==0:#3번마다 1번 업데이트로 바꿈
                loss = training_step(qnet, target_qnet)
                loss_hist.append(loss)

        # 타깃 네트워크 주기적 동기화
        if episode % target_sync_every == 0:
            target_qnet.set_weights(qnet.get_weights())

        if episode % 50 == 0:
            ckpt_path = f"checkpoints/DQN_ep{episode}.weights.h5"
            qnet.save_weights(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if episode % 20 == 0:
            print(f"[Ep {episode:4d}] reward={ep_reward:7.3f}  eps={epsilon:.3f}  buffer={len(replay_buffer)}")

    return qnet, reward_hist, loss_hist

# =========================
# 9) 평가 (탐욕 정책, 마스크 적용)
# =========================
'''
def greedy_action(model, state):
    global illegal_action_count
    mask = legal_action_mask(state)
    s = state.reshape(1, -1) / 9.0
    q = model.predict(s, verbose=0)[0]
    
    q[mask == 0.0] = -1e9
    a = int(np.argmax(q))

    illegal = (mask[a] == 0 or q[a] <= -1e8)

    if illegal:
        illegal_action_count += 1
        legal_q = q.copy()
        legal_q[mask == 0] = -1e9
        a = int(np.argmax(legal_q))
        #print(f"[EVAL] ILLEGAL ACTION SELECTED! action={a}, q={q[a]:.2f}")

    return a, illegal
'''
def greedy_action(model, state):
    global illegal_action_count
    
    mask = legal_action_mask(state)
    s = state.reshape(1, -1) / 9.0
    q = model.predict(s, verbose=0)[0]

    # 모델이 원래 선택하려 했던 action (raw argmax)
    raw_a = int(np.argmax(q))
    raw_illegal = (mask[raw_a] == 0)

    # 실제 사용할 action = 무조건 legal 중에서 선택
    legal_ids = np.where(mask > 0)[0]

    if len(legal_ids) == 0:
        # 모든 액션이 illegal인 비정상 상황 방지용
        illegal_action_count += 1
        return np.random.randint(729), True

    # legal Q만 고려해서 argmax
    legal_a = legal_ids[np.argmax(q[legal_ids])]

    # illegal 카운트 증가 (raw 선택이 illegal이었는가?)
    if raw_illegal:
        illegal_action_count += 1

    return int(legal_a), raw_illegal

def evaluate(model, puzzles, max_episodes=100, save_dir="best_sudoku"):
    env = SudokuEnv(puzzles)
    solved = 0
    illegal_streak = 0

    os.makedirs("output", exist_ok=True)

    best_reward = -1e9
    best_board = None
    best_ep = -1

    for ep in range(min(max_episodes, len(puzzles))):
        state = env.reset(idx=ep)
        done = False
        total_r = 0

        while True:
            a,illegal = greedy_action(model, state)
            next_state, r, done, _ = env.step(a)

            if illegal:
                illegal_streak +=1
                if illegal_streak >= 5:
                    break
            else: 
                illegal_streak=0

            total_r += r
            state = next_state
            if done:
                break

        if is_solved(state):
            solved += 1

        if total_r > best_reward:
            best_reward = total_r
            best_board = state.copy()
            best_ep = ep
    
    save_path = f"{save_dir}/best_reward_ep{best_ep}.png"
    save_sudoku_image(best_board, save_path)

    with open(f"{save_dir}/best_reward_info.txt", "w") as f:
        f.write(f"episode: {best_ep}\nreward: {best_reward}\nboard:\n{best_board}")

    print(f"Total illegal: {illegal_action_count}")
    print(f"Solved: {solved}/{max_episodes}")
    
    return solved

# =========================
# 10) 학습 결과 시각화 및 저장
# =========================
def save_sudoku_image(board, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(np.zeros((9, 9)), cmap="gray_r")  # 그냥 배경
    plt.xticks(range(9))
    plt.yticks(range(9))
    plt.grid(True)

    for i in range(9):
        for j in range(9):
            num = board[i, j]
            if num != 0:
                plt.text(j, i, str(num), ha="center", va="center", fontsize=16)

    plt.title("Sudoku Board")
    plt.savefig(path)
    plt.close()


def plot_and_save_curves(rewards, losses, save_dir="output"):
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 자동 생성

    # ---- 보상 그래프 ----
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color="tab:blue", label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Sudoku RL - Episode Reward Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_curve.png"))
    plt.close()

    # ---- 손실 그래프 ----
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color="tab:red", label="Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Sudoku RL - Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    print(f"그래프 저장 완료: {save_dir}/reward_curve.png, {save_dir}/loss_curve.png")

# =========================
# 11) 실행
# =========================
if __name__ == "__main__":
    args = parse_args()
    import glob
    import os

    # load test set
    _, _, _, _, X_test, _ = load_dataset()
    
    if args.mode == "train":
        print("=== TRAINING START ===")
        qnet, rewards, losses = train_rl_only(num_episodes=200)

        plot_and_save_curves(rewards, losses, save_dir="output")
        print("=== TRAINING DONE ===")

    elif args.mode == "eval":
        print("=== EVALUATION START ===")
        ckpt_list = sorted(glob.glob("checkpoints/DQN_ep*.weights.h5"))
        print("Found checkpoints:", ckpt_list)

        for ckpt in ckpt_list:
            print(f"\n============================")
            print(f" Evaluating checkpoint: {ckpt}")
            print(f"============================")
            qnet = build_qnet()
            qnet.load_weights(ckpt)
            print("Loaded checkpoint")
            # 평가 결과 저장 폴더 분리
            ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
            save_dir = f"best_sudoku/{ckpt_name}"
            os.makedirs(save_dir, exist_ok=True)
            evaluate(qnet, X_test, max_episodes=100, save_dir=save_dir)
        print("=== EVALUATION DONE ===")
