"""
train_networks.py — Train Auction Value Network + Opponent Range Estimator
===========================================================================
Run on your GPU server:
  python train_networks.py --data ./data/dataset.pkl --out ./models/

Trains two networks:
  1. AuctionNet: state → optimal bid amount
     Architecture: 33 → 256 → 256 → 128 → 1
     Loss: Huber loss on normalized bid + win probability head
     
  2. RangeNet: (state + opp_action) → opp hand strength [0,1]
     Architecture: 35 → 256 → 256 → 128 → 1
     Loss: MSE on hand strength percentile

Both networks are tiny — train in minutes on GPU, inference in <0.1ms.
"""

import os, sys, pickle, argparse, time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_OK = True
except ImportError:
    print("PyTorch not found. Install with: pip install torch")
    sys.exit(1)


# ── Hyperparameters ────────────────────────────────────────────────────────

CFG = {
    'state_dim':        33,
    'hidden':           256,
    'batch_size':       512,
    'lr':               3e-4,
    'epochs_auction':   60,
    'epochs_range':     60,
    'dropout':          0.15,
    'weight_decay':     1e-5,
    'val_split':        0.15,
    'patience':         10,      # early stopping
    'grad_clip':        1.0,
}


# ── Model architectures ────────────────────────────────────────────────────

class AuctionNet(nn.Module):
    """
    Predicts: optimal bid as fraction of my_chips [0, 1].
    Also has auxiliary head: probability of winning auction with this bid.
    
    Input:  state_vec (33 dims)
    Output: bid_fraction (1 dim, sigmoid → [0,1])
            win_prob    (1 dim, sigmoid → [0,1])  ← auxiliary, helps training
    """
    def __init__(self, state_dim=33, hidden=256, dropout=0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
        )
        self.bid_head = nn.Sequential(
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()   # [0, 1] fraction of chips
        )
        self.win_head = nn.Sequential(
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()   # win probability
        )
    
    def forward(self, x):
        h = self.backbone(x)
        return self.bid_head(h), self.win_head(h)
    
    def predict_bid(self, x):
        """Inference only: returns bid fraction."""
        with torch.no_grad():
            h = self.backbone(x)
            return self.bid_head(h)


class RangeNet(nn.Module):
    """
    Predicts opponent hand strength given state + their action.
    
    Input:  [state_vec (33), action_onehot (5), bet_ratio (1)] = 39 dims
    Output: hand_strength [0, 1]  (0=trash, 1=monster)
    """
    def __init__(self, state_dim=33, n_actions=5, hidden=256, dropout=0.15):
        super().__init__()
        input_dim = state_dim + n_actions + 1  # +1 for bet_ratio
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()   # [0, 1] hand strength
        )
    
    def forward(self, state, action_onehot, bet_ratio):
        x = torch.cat([state, action_onehot, bet_ratio.unsqueeze(-1)], dim=-1)
        return self.net(x)


# ── Training utilities ─────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float('inf')
    
    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False   # continue
        self.counter += 1
        return self.counter >= self.patience   # True = stop


def train_epoch(model, loader, optimizer, loss_fn, device, clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(model, batch, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            loss = loss_fn(model, batch, device)
            total_loss += loss.item()
    return total_loss / len(loader)


# ── Auction Net training ───────────────────────────────────────────────────

def build_auction_dataset(samples, device):
    """
    Build tensors for auction training.
    
    Target construction:
    - Primary target: "what bid fraction maximizes EV?"
    - Proxy: we know who won (1/0) and the outcome.
    - If we WON the auction → our bid was high enough (bid=our_bid is OK)
    - If we LOST → our bid was too low (optimal bid >= opp_bid)
    - EV target: bid just above opp_bid when possible
    
    We regress on: normalized bid + BCE on win + outcome.
    """
    states    = []
    bid_norms = []   # bid / my_chips (what they actually bid, normalized)
    won       = []
    outcomes  = []
    chips     = []
    
    for s in samples:
        states.append(s['state'])
        # Approximate my_chips from state vector (index 6 = my_chip_n * 5000)
        my_chips_approx = max(s['state'][6] * 5000, 20)
        bid_norm        = min(s['my_bid'] / my_chips_approx, 1.0)
        bid_norms.append(bid_norm)
        won.append(s['won'])
        outcomes.append(s['outcome'])
        chips.append(my_chips_approx)
    
    states    = torch.tensor(np.array(states),    dtype=torch.float32)
    bid_norms = torch.tensor(bid_norms,           dtype=torch.float32)
    won_t     = torch.tensor(won,                 dtype=torch.float32)
    outcomes  = torch.tensor(outcomes,            dtype=torch.float32)
    
    return TensorDataset(states, bid_norms, won_t, outcomes)


def auction_loss_fn(model, batch, device):
    states, bid_targets, won_targets, outcomes = [b.to(device) for b in batch]
    
    pred_bid, pred_win = model(states)
    pred_bid = pred_bid.squeeze(-1)
    pred_win = pred_win.squeeze(-1)
    
    # Huber loss on bid (robust to outliers from crazy bidders)
    bid_loss = nn.functional.huber_loss(pred_bid, bid_targets, delta=0.1)
    
    # BCE on win probability
    win_loss = nn.functional.binary_cross_entropy(pred_win, won_targets)
    
    # Outcome correlation: push bids that led to positive outcomes higher
    # This is a soft reward signal
    outcome_weight = outcomes.clamp(-1, 1)
    outcome_loss   = -(pred_bid * outcome_weight).mean() * 0.1
    
    return bid_loss + 0.3 * win_loss + outcome_loss


def train_auction_net(samples, cfg, device, save_path):
    print("\n" + "=" * 60)
    print("TRAINING AUCTION VALUE NETWORK")
    print("=" * 60)
    print(f"  Samples: {len(samples):,}")
    print(f"  Device:  {device}")
    
    dataset = build_auction_dataset(samples, device)
    n_val   = int(len(dataset) * cfg['val_split'])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model = AuctionNet(cfg['state_dim'], cfg['hidden'], cfg['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs_auction'])
    stopper   = EarlyStopping(patience=cfg['patience'])
    
    best_val  = float('inf')
    best_state= None
    
    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>8}")
    print("-" * 50)
    
    for epoch in range(1, cfg['epochs_auction'] + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, auction_loss_fn, device, cfg['grad_clip'])
        vl = val_epoch(model, val_loader, auction_loss_fn, device)
        scheduler.step()
        
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:6d} {tr:10.5f} {vl:10.5f} {lr:10.6f} {time.time()-t0:7.1f}s")
        
        if stopper.step(vl):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Load best weights
    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'cfg': cfg, 'type': 'auction'}, save_path)
    print(f"\nAuction net saved to {save_path}  (best val loss: {best_val:.5f})")
    return model


# ── Range Net training ─────────────────────────────────────────────────────

def build_range_dataset(samples):
    states    = []
    actions   = []   # integer 0-4
    bet_ratios= []
    strengths = []   # target: opp hand strength [0,1]
    
    for s in samples:
        states.append(s['state'])
        actions.append(s['action'])
        bet_ratios.append(s['bet_ratio'])
        strengths.append(s['opp_strength'])
    
    states     = torch.tensor(np.array(states),  dtype=torch.float32)
    actions_t  = torch.zeros(len(actions), 5,    dtype=torch.float32)
    for i, a in enumerate(actions):
        actions_t[i, a] = 1.0
    bet_ratios = torch.tensor(bet_ratios,         dtype=torch.float32)
    strengths  = torch.tensor(strengths,          dtype=torch.float32)
    
    return TensorDataset(states, actions_t, bet_ratios, strengths)


def range_loss_fn(model, batch, device):
    states, actions, bet_ratios, targets = [b.to(device) for b in batch]
    preds = model(states, actions, bet_ratios).squeeze(-1)
    return nn.functional.mse_loss(preds, targets)


def train_range_net(samples, cfg, device, save_path):
    print("\n" + "=" * 60)
    print("TRAINING OPPONENT RANGE ESTIMATOR")
    print("=" * 60)
    print(f"  Samples: {len(samples):,}")
    print(f"  Device:  {device}")
    
    dataset = build_range_dataset(samples)
    n_val   = int(len(dataset) * cfg['val_split'])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model = RangeNet(cfg['state_dim'], dropout=cfg['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs_range'])
    stopper   = EarlyStopping(patience=cfg['patience'])
    
    best_val  = float('inf')
    best_state= None
    
    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>8}")
    print("-" * 50)
    
    for epoch in range(1, cfg['epochs_range'] + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, range_loss_fn, device, cfg['grad_clip'])
        vl = val_epoch(model, val_loader, range_loss_fn, device)
        scheduler.step()
        
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:6d} {tr:10.5f} {vl:10.5f} {lr:10.6f} {time.time()-t0:7.1f}s")
        
        if stopper.step(vl):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'cfg': cfg, 'type': 'range'}, save_path)
    print(f"\nRange net saved to {save_path}  (best val loss: {best_val:.5f})")
    return model


# ── Multi-GPU wrapper ──────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"Found {n} GPU(s):")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
        return torch.device('cuda:0')   # primary GPU; DataParallel handles multi-GPU
    print("No GPU found, using CPU")
    return torch.device('cpu')


def wrap_multi_gpu(model, device):
    """Wrap with DataParallel if multiple GPUs available."""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model.to(device)


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_auction_net(model, samples, device):
    """Show how well the auction net predicts bids."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for s in samples[:1000]:  # sample for speed
            state   = torch.tensor(s['state'], dtype=torch.float32).unsqueeze(0).to(device)
            my_chips= max(s['state'][6] * 5000, 20)
            actual_bid_norm = min(s['my_bid'] / my_chips, 1.0)
            
            pred_bid_norm, pred_win = model(state)
            pred_bid_norm = pred_bid_norm.item()
            pred_bid_chips= pred_bid_norm * my_chips
            
            errors.append(abs(pred_bid_chips - s['my_bid']))
    
    print(f"\nAuction Net Evaluation (n={len(errors)}):")
    print(f"  Mean absolute error: {np.mean(errors):.1f} chips")
    print(f"  Median error:        {np.median(errors):.1f} chips")
    print(f"  95th pctile error:   {np.percentile(errors, 95):.1f} chips")


def evaluate_range_net(model, samples, device):
    """Evaluate range net accuracy."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        batch_states  = []
        batch_actions = []
        batch_bets    = []
        batch_targets = []
        
        for s in samples[:2000]:
            batch_states.append(s['state'])
            act_oh = [0.0] * 5; act_oh[s['action']] = 1.0
            batch_actions.append(act_oh)
            batch_bets.append(s['bet_ratio'])
            batch_targets.append(s['opp_strength'])
        
        states  = torch.tensor(np.array(batch_states),  dtype=torch.float32).to(device)
        actions = torch.tensor(batch_actions,            dtype=torch.float32).to(device)
        bets    = torch.tensor(batch_bets,               dtype=torch.float32).to(device)
        targets = np.array(batch_targets)
        
        preds = model(states, actions, bets).squeeze(-1).cpu().numpy()
        errors = np.abs(preds - targets)
    
    print(f"\nRange Net Evaluation (n={len(errors)}):")
    print(f"  Mean absolute error: {np.mean(errors):.4f}")
    print(f"  Correlation:         {np.corrcoef(preds, targets)[0,1]:.4f}")
    
    # By action type
    action_names = ['fold', 'check', 'call', 'bet', 'raise']
    for ai, aname in enumerate(action_names):
        mask = [s['action'] == ai for s in samples[:2000]]
        if sum(mask) < 10: continue
        errs = errors[mask]
        print(f"  {aname:8s}: n={sum(mask):4d}  mae={np.mean(errs):.4f}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='./data/dataset.pkl')
    parser.add_argument('--out',     default='./models/')
    parser.add_argument('--epochs_auction', type=int, default=CFG['epochs_auction'])
    parser.add_argument('--epochs_range',   type=int, default=CFG['epochs_range'])
    parser.add_argument('--batch',   type=int, default=CFG['batch_size'])
    parser.add_argument('--lr',      type=float, default=CFG['lr'])
    parser.add_argument('--hidden',  type=int, default=CFG['hidden'])
    args = parser.parse_args()
    
    CFG['epochs_auction'] = args.epochs_auction
    CFG['epochs_range']   = args.epochs_range
    CFG['batch_size']     = args.batch
    CFG['lr']             = args.lr
    CFG['hidden']         = args.hidden
    
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    
    print(f"\nLoading dataset from {args.data}")
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
    
    auction_samples = dataset['auction_samples']
    range_samples   = dataset['range_samples']
    opp_profiles    = dataset['opp_profiles']
    
    print(f"Auction samples: {len(auction_samples):,}")
    print(f"Range samples:   {len(range_samples):,}")
    
    # Save opponent profiles (used by bot at runtime for fast init)
    profiles_path = os.path.join(args.out, 'opp_profiles.pkl')
    with open(profiles_path, 'wb') as f:
        pickle.dump(opp_profiles, f)
    print(f"Opponent profiles saved to {profiles_path}")
    
    # Train auction net
    auction_path = os.path.join(args.out, 'auction_net.pt')
    auction_model = train_auction_net(auction_samples, CFG, device, auction_path)
    if torch.cuda.device_count() > 1:
        evaluate_auction_net(auction_model.module, auction_samples[-500:], device)
    else:
        evaluate_auction_net(auction_model, auction_samples[-500:], device)
    
    # Train range net
    range_path = os.path.join(args.out, 'range_net.pt')
    range_model = train_range_net(range_samples, CFG, device, range_path)
    if torch.cuda.device_count() > 1:
        evaluate_range_net(range_model.module, range_samples[-1000:], device)
    else:
        evaluate_range_net(range_model, range_samples[-1000:], device)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Models saved to: {args.out}")
    print("Next step: python train_cfr.py --models ./models/")
    print("=" * 60)


if __name__ == '__main__':
    main()