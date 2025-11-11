# 15-Conv Pure CNN for Sudoku (Kyubyong-style)
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH   = 64
EPOCHS  = 40
CH, DEPTH = 512, 15   # <- 3x3 conv 15층

# --- data ---
Xtr = np.load("data/train_puzzles.npy");   Ytr = np.load("data/train_solutions.npy")
Xva = np.load("data/val_puzzles.npy");     Yva = np.load("data/val_solutions.npy")

def encode_input(grid):
    x = np.zeros((10,9,9), np.float32)
    for d in range(1,10): x[d-1] = (grid==d)
    x[9] = (grid>0)
    return x

class SudokuDS(Dataset):
    def __init__(self, puzzles, sols, augment=False):
        self.p, self.s, self.aug = puzzles, sols, augment
    def __len__(self): return len(self.p)
    def _permute(self, puz, sol):
        perm = np.random.permutation(9)+1
        p2, s2 = puz.copy(), sol.copy()
        for d in range(1,10): p2[puz==d]=perm[d-1]; s2[sol==d]=perm[d-1]
        return p2, s2
    def __getitem__(self, i):
        puz, sol = self.p[i], self.s[i]
        if self.aug and np.random.rand()<0.9: puz, sol = self._permute(puz, sol)  # 숫자 치환 증강
        x = encode_input(puz)
        y = (sol-1).astype(np.int64)
        m = (puz==0).astype(np.float32)  # blank mask
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

train_dl = DataLoader(SudokuDS(Xtr,Ytr,augment=True), batch_size=BATCH, shuffle=True, num_workers=0)
val_dl   = DataLoader(SudokuDS(Xva,Yva,augment=False), batch_size=128, shuffle=False, num_workers=0)

# --- model: 3x3 conv x15, no pooling, GroupNorm for small batch ---
class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(32, ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class SudokuCNNx15(nn.Module):
    def __init__(self, ch=512, depth=15):
        super().__init__()
        # stem = 1 conv, blocks = depth-1 conv  -> 총 3x3 conv = depth
        self.stem = nn.Sequential(
            nn.Conv2d(10, ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, ch), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[ConvBlock(ch) for _ in range(depth-1)])
        self.head   = nn.Conv2d(ch, 9, 1)  # 1x1 head (클래스 9)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        return self.head(x)                # (B,9,9,9)  (N,C,H,W)

model = SudokuCNNx15(CH, DEPTH).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit  = nn.CrossEntropyLoss(reduction='none')  # blank 마스킹용

# --- train / eval ---
def train_epoch():
    model.train()
    for xb,yb,mb in train_dl:
        xb,yb,mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        logits = model(xb)
        loss_map = crit(logits, yb)                  # (B,9,9)
        loss = (loss_map*mb).sum() / mb.sum().clamp_min(1)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()

@torch.no_grad()
def blank_acc(dl):
    model.eval()
    corr=tot=0
    for xb,yb,mb in dl:
        xb,yb,mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        pred = model(xb).argmax(1)
        corr += ((pred==yb)*mb.bool()).sum().item()
        tot  += mb.sum().item()
    return corr/max(1,tot)

for ep in range(EPOCHS):
    train_epoch()
    va = blank_acc(val_dl)
    print(f"ep {ep:02d} | blank-acc val {va:.4f}")
