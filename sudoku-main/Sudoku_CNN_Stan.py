# 15-Conv (ResNet-style) CNN for Sudoku (Kyubyong-style, with improvements 2/3/4/5)
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# --- default hyperparams ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64
EPOCHS = 60  # (3) 조금 더 길게 학습
CH, DEPTH = 512, 15  # conv depth "의도"는 3x3 conv 15층 정도

# ======================
# data & encoding
# ======================
Xtr = np.load("data/train_puzzles.npy")
Ytr = np.load("data/train_solutions.npy")
Xva = np.load("data/val_puzzles.npy")
Yva = np.load("data/val_solutions.npy")


def encode_input(grid):
    """10채널 인코딩: 1~9 원핫 + nonzero mask"""
    x = np.zeros((10, 9, 9), np.float32)
    for d in range(1, 10):
        x[d - 1] = grid == d
    x[9] = grid > 0
    return x


class SudokuDS(Dataset):
    def __init__(self, puzzles, sols, augment=False):
        self.p = puzzles
        self.s = sols
        self.aug = augment

    def __len__(self):
        return len(self.p)

    # -------- (5) Sudoku symmetry augmentations --------
    def _permute_digits(self, puz, sol):
        """1~9 숫자 치환"""
        perm = np.random.permutation(9) + 1
        p2, s2 = puz.copy(), sol.copy()
        for d in range(1, 10):
            p2[puz == d] = perm[d - 1]
            s2[sol == d] = perm[d - 1]
        return p2, s2

    def _permute_rows_within_bands(self, puz, sol):
        """각 band(3행 묶음) 안에서 행 permute"""
        p2, s2 = puz.copy(), sol.copy()
        for band in range(3):
            rows = np.arange(band * 3, band * 3 + 3)
            perm = np.random.permutation(rows)
            p2[rows] = p2[perm]
            s2[rows] = s2[perm]
        return p2, s2

    def _permute_cols_within_stacks(self, puz, sol):
        """각 stack(3열 묶음) 안에서 열 permute"""
        p2, s2 = puz.copy(), sol.copy()
        for stack in range(3):
            cols = np.arange(stack * 3, stack * 3 + 3)
            perm = np.random.permutation(cols)
            p2[:, cols] = p2[:, perm]
            s2[:, cols] = s2[:, perm]
        return p2, s2

    def _permute_bands(self, puz, sol):
        """3개 band 순서 permute (행 방향)"""
        p2 = puz.copy()
        s2 = sol.copy()
        perm = np.random.permutation(3)
        for b in range(3):
            src = perm[b]
            p2[b * 3 : (b + 1) * 3] = puz[src * 3 : (src + 1) * 3]
            s2[b * 3 : (b + 1) * 3] = sol[src * 3 : (src + 1) * 3]
        return p2, s2

    def _permute_stacks(self, puz, sol):
        """3개 stack 순서 permute (열 방향)"""
        p2 = puz.copy()
        s2 = sol.copy()
        perm = np.random.permutation(3)
        for s in range(3):
            src = perm[s]
            p2[:, s * 3 : (s + 1) * 3] = puz[:, src * 3 : (src + 1) * 3]
            s2[:, s * 3 : (s + 1) * 3] = sol[:, src * 3 : (src + 1) * 3]
        return p2, s2

    def _transpose(self, puz, sol):
        """전치 (행/열 교환)"""
        return puz.T.copy(), sol.T.copy()

    def _augment(self, puz, sol):
        """조합된 증강 파이프라인"""
        if np.random.rand() < 0.9:
            puz, sol = self._permute_digits(puz, sol)
        if np.random.rand() < 0.5:
            puz, sol = self._permute_rows_within_bands(puz, sol)
        if np.random.rand() < 0.5:
            puz, sol = self._permute_cols_within_stacks(puz, sol)
        if np.random.rand() < 0.3:
            puz, sol = self._permute_bands(puz, sol)
        if np.random.rand() < 0.3:
            puz, sol = self._permute_stacks(puz, sol)
        if np.random.rand() < 0.3:
            puz, sol = self._transpose(puz, sol)
        return puz, sol

    def __getitem__(self, i):
        puz, sol = self.p[i], self.s[i]
        if self.aug:
            puz, sol = self._augment(puz, sol)

        x = encode_input(puz)  # (10,9,9)
        y = (sol - 1).astype(np.int64)  # target 0~8
        m_blank = (puz == 0).astype(np.float32)  # blank mask (blank-acc용)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m_blank),
        )


def make_loaders(batch, num_workers, device):
    pin = device.type == "cuda"
    train_dl = DataLoader(
        SudokuDS(Xtr, Ytr, augment=True),
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        SudokuDS(Xva, Yva, augment=False),
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_dl, val_dl


# ======================
# model (4) ResBlock
# ======================


class ResBlock(nn.Module):
    """2x 3x3 conv + GroupNorm + ReLU + skip"""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + x
        return self.act(out)


class SudokuCNNx15(nn.Module):
    """
    depth ≈ 총 3x3 conv 개수 (stem 1개 + ResBlock당 2개 기준으로 맞춤)
    depth=15 -> stem(1) + ResBlock 7개(14) = 15 conv
    """

    def __init__(self, ch=512, depth=15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, ch),
            nn.ReLU(inplace=True),
        )
        # depth-1개의 conv를 2개씩 쓰는 ResBlock에 배분
        n_blocks = max(1, (depth - 1) // 2)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)])
        self.head = nn.Conv2d(ch, 9, 1)  # 1x1 conv -> 9 클래스

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)  # (B, 9, 9, 9) = (N,C,H,W)


# ======================
# train / eval
# ======================


def train_epoch(model, opt, crit, train_dl, device, sched=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_w = 0.0

    # 셀/퍼즐 메트릭 누적 변수
    corr_blank = 0  # blank 셀 정답 개수
    tot_blank = 0  # blank 셀 전체 개수
    corr_all = 0  # 전체 셀 정답 개수
    tot_all = 0  # 전체 셀 개수
    puzzle_full = 0  # 81칸 전부 맞은 퍼즐 개수
    puzzle_blank = 0  # 빈칸만 모두 맞은 퍼즐 개수
    n_boards = 0  # 퍼즐 개수

    for xb, yb, mb in train_dl:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

        logits = model(xb)  # (B,9,9,9)
        loss_map = crit(logits, yb)  # (B,9,9)

        # (2) 모든 칸을 학습에 사용하되,
        #     blank=1.0, givens=0.3 정도로 가중치
        w = mb + 0.3 * (1.0 - mb)  # mb: blank mask (0 or 1)
        weighted = loss_map * w
        batch_w = w.sum().clamp_min(1.0)
        loss = weighted.sum() / batch_w
        total_loss += weighted.sum().item()
        total_w += batch_w.item()

        # 메트릭 계산
        with torch.no_grad():
            pred = logits.argmax(1)  # (B,9,9)
            correct = pred == yb  # (B,9,9) bool
            blank_mask = mb.bool()  # (B,9,9)

            # 셀 단위
            corr_blank += (correct & blank_mask).sum().item()
            tot_blank += blank_mask.sum().item()

            corr_all += correct.sum().item()
            tot_all += correct.numel()

            B = xb.size(0)
            n_boards += B

            # 퍼즐 단위 (1) 81칸 전부 맞은 퍼즐
            full_solved = correct.view(B, -1).all(dim=1)  # (B,)
            puzzle_full += full_solved.sum().item()

            # 퍼즐 단위 (2) "빈칸"만 모두 맞은 퍼즐
            # -> 빈칸은 correct여야 하고, givens는 상관 없음
            blank_ok_grid = (~blank_mask) | correct  # (B,9,9)
            blank_solved = blank_ok_grid.view(B, -1).all(dim=1)
            puzzle_blank += blank_solved.sum().item()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    if sched is not None:
        sched.step()

    avg_loss = total_loss / max(1.0, total_w)
    blank_acc = corr_blank / max(1, tot_blank)
    cell_acc = corr_all / max(1, tot_all)
    puzzle_full_acc = puzzle_full / max(1, n_boards)
    puzzle_blank_acc = puzzle_blank / max(1, n_boards)

    return avg_loss, blank_acc, cell_acc, puzzle_full_acc, puzzle_blank_acc


@torch.no_grad()
def evaluate(model, dl, device, crit):
    """loss + blank-acc + cell-acc + puzzle-level acc"""
    model.eval()
    total_loss = 0.0
    total_w = 0.0

    corr_blank = 0
    tot_blank = 0
    corr_all = 0
    tot_all = 0
    puzzle_full = 0
    puzzle_blank = 0
    n_boards = 0

    for xb, yb, mb in dl:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        logits = model(xb)
        loss_map = crit(logits, yb)
        w = mb + 0.3 * (1.0 - mb)
        weighted = loss_map * w
        total_loss += weighted.sum().item()
        total_w += w.sum().item()

        pred = logits.argmax(1)
        correct = pred == yb
        blank_mask = mb.bool()

        # 셀 단위
        corr_blank += (correct & blank_mask).sum().item()
        tot_blank += blank_mask.sum().item()

        corr_all += correct.sum().item()
        tot_all += correct.numel()

        B = xb.size(0)
        n_boards += B

        # 퍼즐 단위 (1) 81칸 전부 맞은 퍼즐
        full_solved = correct.view(B, -1).all(dim=1)
        puzzle_full += full_solved.sum().item()

        # 퍼즐 단위 (2) 빈칸만 모두 맞은 퍼즐
        blank_ok_grid = (~blank_mask) | correct
        blank_solved = blank_ok_grid.view(B, -1).all(dim=1)
        puzzle_blank += blank_solved.sum().item()

    avg_loss = total_loss / max(1.0, total_w)
    blank_acc = corr_blank / max(1, tot_blank)
    cell_acc = corr_all / max(1, tot_all)
    puzzle_full_acc = puzzle_full / max(1, n_boards)
    puzzle_blank_acc = puzzle_blank / max(1, n_boards)

    return avg_loss, blank_acc, cell_acc, puzzle_full_acc, puzzle_blank_acc


def save_checkpoint(path, model, opt, sched, epoch, best_metric, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict() if sched else None,
            "best_val_puzzle": best_metric,
            "args": vars(args),
        },
        path,
    )
    print(f"checkpoint saved: {path}")


# ======================
# CLI
# ======================


def parse_args():
    p = argparse.ArgumentParser(description="Train Sudoku CNN")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH, help="train batch size")
    p.add_argument("--channels", type=int, default=CH, help="channels (original=512)")
    p.add_argument("--depth", type=int, default=DEPTH, help="conv depth (original=15)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument(
        "--device", type=str, default=None, help="force device: cpu/cuda/mps"
    )
    p.add_argument(
        "--fast", action="store_true", help="lighter config for quick CPU runs"
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--full",
        action="store_true",
        help="skip CPU auto-downsizing; keep provided hyperparams",
    )
    p.add_argument(
        "--log-csv", type=str, default="", help="CSV path to save per-epoch metrics"
    )
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="PNG path to save curves (requires matplotlib)",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="runs/ckpt",
        help="directory to save checkpoints",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="save checkpoint every N epochs (0 to disable)",
    )
    p.add_argument(
        "--name",
        type=str,
        default="stan",
        help="prefix for checkpoint filenames",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else DEVICE

    # Auto-downsize for CPU unless explicitly asked not to.
    ch, depth, epochs, batch = args.channels, args.depth, args.epochs, args.batch
    user_overrode = any(
        [
            args.channels != CH,
            args.depth != DEPTH,
            args.epochs != EPOCHS,
            args.batch != BATCH,
        ]
    )
    if args.fast:
        ch, depth, epochs, batch = 128, 6, min(args.epochs, 5), min(args.batch, 128)
        print(f"fast mode -> ch={ch}, depth={depth}, epochs={epochs}, batch={batch}")
    elif device.type == "cpu" and not args.full and not user_overrode:
        ch, depth, epochs, batch = 128, 9, min(args.epochs, 12), min(args.batch, 128)
        print(
            f"cpu detected; using lighter config ch={ch}, depth={depth}, "
            f"epochs={epochs}, batch={batch} (use --full for full model)"
        )

    print(f"device={device}, ch={ch}, depth={depth}, epochs={epochs}, batch={batch}")

    train_dl, val_dl = make_loaders(batch, args.num_workers, device)
    model = SudokuCNNx15(ch, depth).to(device)

    # (3) lr를 1e-3로 상향
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(reduction="none")  # per-cell loss (masking/가중치용)

    history = []
    best_val_puzzle = -1.0
    save_dir = Path(args.save_dir)
    try:
        for ep in range(epochs):
            tr_loss, tr_blank, tr_cell, tr_puzzle, tr_puzzle_blank = train_epoch(
                model, opt, crit, train_dl, device, sched, grad_clip=args.grad_clip
            )
            va_loss, va_blank, va_cell, va_puzzle, va_puzzle_blank = evaluate(
                model, val_dl, device, crit
            )

            history.append(
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "train_blank": tr_blank,
                    "train_cell": tr_cell,
                    "train_puzzle": tr_puzzle,
                    "train_puzzle_blank": tr_puzzle_blank,
                    "val_loss": va_loss,
                    "val_blank": va_blank,
                    "val_cell": va_cell,
                    "val_puzzle": va_puzzle,
                    "val_puzzle_blank": va_puzzle_blank,
                }
            )
            print(
                f"ep {ep:02d} | "
                f"train_loss {tr_loss:.4f} | "
                f"train_blank {tr_blank:.4f} | train_cell {tr_cell:.4f} | "
                f"train_puzzle {tr_puzzle:.4f} | "
                f"val_loss {va_loss:.4f} | "
                f"val_blank {va_blank:.4f} | val_cell {va_cell:.4f} | "
                f"val_puzzle {va_puzzle:.4f}"
            )

            # checkpointing
            ep_num = ep + 1
            if va_puzzle > best_val_puzzle:
                best_val_puzzle = va_puzzle
                save_checkpoint(
                    save_dir / f"{args.name}_best.pt",
                    model,
                    opt,
                    sched,
                    ep_num,
                    best_val_puzzle,
                    args,
                )
            if args.save_every > 0 and ep_num % args.save_every == 0:
                save_checkpoint(
                    save_dir / f"{args.name}_ep{ep_num:03d}.pt",
                    model,
                    opt,
                    sched,
                    ep_num,
                    best_val_puzzle,
                    args,
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Returning current model.")
    finally:
        if history:
            save_checkpoint(
                save_dir / f"{args.name}_last.pt",
                model,
                opt,
                sched,
                len(history),
                best_val_puzzle,
                args,
            )
        if args.log_csv and history:
            path = Path(args.log_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "train_blank",
                        "train_cell",
                        "train_puzzle",
                        "train_puzzle_blank",
                        "val_loss",
                        "val_blank",
                        "val_cell",
                        "val_puzzle",
                        "val_puzzle_blank",
                    ],
                )
                writer.writeheader()
                writer.writerows(history)
            print(f"metrics written to {path}")

        if args.plot and history:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not installed; skip plotting")
            else:
                path = Path(args.plot)
                path.parent.mkdir(parents=True, exist_ok=True)
                epochs_ax = [h["epoch"] for h in history]
                tr_loss = [h["train_loss"] for h in history]
                va_loss = [h["val_loss"] for h in history]
                tr_blank = [h["train_blank"] for h in history]
                va_blank = [h["val_blank"] for h in history]
                va_puzzle = [h["val_puzzle"] for h in history]

                plt.figure(figsize=(9, 4))
                plt.subplot(1, 2, 1)
                plt.plot(epochs_ax, tr_loss, label="train")
                plt.plot(epochs_ax, va_loss, label="val")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("Loss")
                plt.legend()
                plt.grid(True, alpha=0.2)

                plt.subplot(1, 2, 2)
                plt.plot(epochs_ax, tr_blank, label="train blank-acc")
                plt.plot(epochs_ax, va_blank, label="val blank-acc")
                plt.plot(epochs_ax, va_puzzle, label="val puzzle-full", linestyle="--")
                plt.xlabel("epoch")
                plt.ylabel("accuracy")
                plt.title("Accuracy")
                plt.legend()
                plt.grid(True, alpha=0.2)

                plt.tight_layout()
                plt.savefig(path, dpi=150)
                plt.close()
                print(f"plot saved to {path}")


if __name__ == "__main__":
    main()
