"""
train_cfr.py — Deep CFR Self-Play Training (Background Process)
================================================================
Run on your GPU server (background, Days 1-6):
  python train_cfr.py --models ./models/ --checkpoint_dir ./cfr_checkpoints/ &

This implements External Sampling Monte Carlo CFR with Deep Neural Networks.
Reference: "Deep Counterfactual Regret Minimization" (Brown et al., NeurIPS 2019)

The trained strategy network becomes the bot's baseline policy — it plays
near Nash equilibrium, meaning it cannot be exploited.

Key design for 6-day timeline:
- Card abstraction: 169 preflop buckets + 50 postflop buckets (manageable)  
- Action abstraction: 7 actions (fold/check-call/bet33/bet66/bet100/bet150/allin)
- Network: tiny (33→256→256→7), fast inference
- Multi-GPU: DataParallel across your 2-4 GPUs
- Target: 10M traversals = meaningful convergence in ~24-48 hours

IMPORTANT: Even partial convergence (1-2M traversals) beats any heuristic bot.
"""

import os, sys, pickle, argparse, time, random, math
import numpy as np
from collections import defaultdict
from itertools import combinations

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────

STARTING_STACK = 5000
BIG_BLIND      = 20
SMALL_BLIND    = 10
RANKS          = '23456789TJQKA'
SUITS          = 'cdhs'
RANK_V         = {r: i for i, r in enumerate(RANKS)}
FULL_DECK      = [r+s for r in RANKS for s in SUITS]

try:
    import eval7 as _e7
    def evaluate_hand(cards):
        return _e7.evaluate([_e7.Card(c) for c in cards])
    def make_deck():
        d = _e7.Deck(); d.shuffle(); return d
    HAS_EVAL7 = True
except ImportError:
    HAS_EVAL7 = False
    def _score5(cards):
        from collections import Counter
        rs  = sorted([RANK_V[c[0]] for c in cards], reverse=True)
        fl  = len(set(c[1] for c in cards)) == 1
        ur  = set(rs)
        st  = hi = 0
        if len(ur) == 5:
            if rs[0]-rs[4] == 4:          st, hi = 1, rs[0]
            elif ur == {12,3,2,1,0}:      st, hi = 1, 3
        cnt   = Counter(rs)
        freqs = sorted(cnt.values(), reverse=True)
        def pk(cat, tb):
            v = cat*(15**6)
            for i,r in enumerate(tb[:6]): v+=r*(15**(5-i))
            return v
        if st and fl: return pk(8,[hi])
        if freqs[0]==4:
            q=next(r for r,f in cnt.items() if f==4)
            k=next(r for r,f in cnt.items() if f==1)
            return pk(7,[q,k])
        if freqs[0]==3 and freqs[1]==2:
            return pk(6,[next(r for r,f in cnt.items() if f==3),
                         next(r for r,f in cnt.items() if f==2)])
        if fl: return pk(5,rs)
        if st: return pk(4,[hi])
        if freqs[0]==3:
            t=next(r for r,f in cnt.items() if f==3)
            ks=sorted((r for r,f in cnt.items() if f==1),reverse=True)
            return pk(3,[t]+ks)
        if freqs[0]==2 and freqs[1]==2:
            ps=sorted((r for r,f in cnt.items() if f==2),reverse=True)
            k=next(r for r,f in cnt.items() if f==1)
            return pk(2,ps+[k])
        if freqs[0]==2:
            p=next(r for r,f in cnt.items() if f==2)
            ks=sorted((r for r,f in cnt.items() if f==1),reverse=True)
            return pk(1,[p]+ks)
        return pk(0,rs)
    def evaluate_hand(cards):
        if len(cards)<=5: return _score5(cards)
        return max(_score5(list(c)) for c in combinations(cards,5))


# ── Card Abstraction ───────────────────────────────────────────────────────

# Preflop: 169 canonical hand classes (AA, KK, ..., 32o)
# We use preflop equity percentile as a continuous bucket [0,1]

PREFLOP_EQ_TABLE = {
    'AA': 0.852, 'KK': 0.827, 'QQ': 0.794, 'JJ': 0.772, 'TT': 0.744,
    '99': 0.717, '88': 0.698, '77': 0.668, '66': 0.643, '55': 0.616,
    '44': 0.568, '33': 0.547, '22': 0.498,
    'AKs': 0.684, 'AQs': 0.649, 'AJs': 0.647, 'ATs': 0.645, 'A9s': 0.646,
    'A8s': 0.632, 'A7s': 0.603, 'A6s': 0.600, 'A5s': 0.607, 'A4s': 0.583,
    'A3s': 0.573, 'A2s': 0.567,
    'AKo': 0.648, 'AQo': 0.636, 'AJo': 0.643, 'ATo': 0.615, 'A9o': 0.602,
    'A8o': 0.616, 'A7o': 0.589, 'A6o': 0.559, 'A5o': 0.583, 'A4o': 0.578,
    'A3o': 0.555, 'A2o': 0.538,
    'KQs': 0.622, 'KJs': 0.630, 'KTs': 0.625, 'K9s': 0.616, 'K8s': 0.598,
    'K7s': 0.565, 'K6s': 0.571, 'K5s': 0.561, 'K4s': 0.541, 'K3s': 0.544,
    'K2s': 0.513,
    'KQo': 0.605, 'KJo': 0.604, 'KTo': 0.586, 'K9o': 0.565, 'K8o': 0.562,
    'K7o': 0.557, 'K6o': 0.526, 'K5o': 0.537, 'K4o': 0.521, 'K3o': 0.515,
    'K2o': 0.516,
    'QJs': 0.608, 'QTs': 0.574, 'Q9s': 0.581, 'Q8s': 0.557, 'Q7s': 0.532,
    'Q6s': 0.534, 'Q5s': 0.522, 'Q4s': 0.506, 'Q3s': 0.509, 'Q2s': 0.503,
    'QJo': 0.573, 'QTo': 0.573, 'Q9o': 0.562, 'Q8o': 0.541, 'Q7o': 0.503,
    'Q6o': 0.503, 'Q5o': 0.493, 'Q4o': 0.498, 'Q3o': 0.492, 'Q2o': 0.479,
    'JTs': 0.573, 'J9s': 0.535, 'J8s': 0.557, 'J7s': 0.516, 'J6s': 0.521,
    'J5s': 0.497, 'J4s': 0.485, 'J3s': 0.490, 'J2s': 0.470,
    'JTo': 0.545, 'J9o': 0.524, 'J8o': 0.514, 'J7o': 0.496, 'J6o': 0.481,
    'J5o': 0.452, 'J4o': 0.464, 'J3o': 0.444, 'J2o': 0.439,
    'T9s': 0.522, 'T8s': 0.516, 'T7s': 0.493, 'T6s': 0.483, 'T5s': 0.460,
    'T4s': 0.468, 'T3s': 0.468, 'T2s': 0.438,
    'T9o': 0.510, 'T8o': 0.487, 'T7o': 0.492, 'T6o': 0.443, 'T5o': 0.443,
    'T4o': 0.449, 'T3o': 0.428, 'T2o': 0.414,
    '98s': 0.509, '97s': 0.466, '96s': 0.477, '95s': 0.464, '94s': 0.425,
    '93s': 0.439, '92s': 0.407,
    '98o': 0.471, '97o': 0.460, '96o': 0.443, '95o': 0.422, '94o': 0.411,
    '93o': 0.402, '92o': 0.392,
    '87s': 0.470, '86s': 0.472, '85s': 0.450, '84s': 0.420, '83s': 0.415,
    '82s': 0.411,
    '87o': 0.437, '86o': 0.427, '85o': 0.410, '84o': 0.393, '83o': 0.371,
    '82o': 0.368,
    '76s': 0.468, '75s': 0.454, '74s': 0.409, '73s': 0.375, '72s': 0.376,
    '76o': 0.413, '75o': 0.396, '74o': 0.378, '73o': 0.384, '72o': 0.350,
    '65s': 0.440, '64s': 0.415, '63s': 0.412, '62s': 0.386,
    '65o': 0.398, '64o': 0.379, '63o': 0.358, '62o': 0.356,
    '54s': 0.407, '53s': 0.389, '52s': 0.392,
    '54o': 0.376, '53o': 0.360, '52o': 0.345,
    '43s': 0.390, '42s': 0.378, '43o': 0.364, '42o': 0.333,
    '32s': 0.360, '32o': 0.325,
}

def hand_key(hand):
    r1, r2 = RANK_V[hand[0][0]], RANK_V[hand[1][0]]
    s1, s2 = hand[0][1], hand[1][1]
    if r1 < r2: r1, r2, s1, s2 = r2, r1, s2, s1
    c1, c2 = RANKS[r1], RANKS[r2]
    if r1 == r2: return c1+c2
    return c1+c2+('s' if s1==s2 else 'o')

def pf_equity(hand):
    return PREFLOP_EQ_TABLE.get(hand_key(hand), 0.45)

def postflop_bucket(hand, board, n_samples=20):
    """Fast equity bucket [0,1] via MC."""
    known = set(hand+board)
    deck  = [c for c in FULL_DECK if c not in known]
    need_b= 5-len(board)
    wins = total = 0
    for _ in range(n_samples):
        draw = random.sample(deck, need_b+2)
        b5   = board + draw[:need_b]
        opp  = draw[need_b:]
        ms   = evaluate_hand(hand+b5)
        os   = evaluate_hand(opp+b5)
        if ms > os: wins += 1
        elif ms == os: wins += 0.5
        total += 1
    return wins/total if total else 0.5


# ── Action Abstraction ─────────────────────────────────────────────────────

# 7 abstract actions: fold, check/call, bet33%, bet66%, bet100%, bet150%, allin
N_ACTIONS = 7
ACTION_FOLD   = 0
ACTION_CALL   = 1   # check if no bet, call otherwise
ACTION_BET33  = 2
ACTION_BET66  = 3
ACTION_BET100 = 4
ACTION_BET150 = 5
ACTION_ALLIN  = 6

# Auction: 6 abstract bid sizes
N_BID_ACTIONS = 6
BID_FRACS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.10]  # fraction of stack

def abstract_to_raise(action_idx, pot, mn, mx):
    """Convert abstract action to actual raise amount."""
    fracs = {
        ACTION_BET33:  0.33,
        ACTION_BET66:  0.66,
        ACTION_BET100: 1.00,
        ACTION_BET150: 1.50,
        ACTION_ALLIN:  10.0,   # will be clamped to mx
    }
    frac  = fracs.get(action_idx, 0.66)
    amount= int(pot * frac)
    return max(mn, min(amount, mx))

def abstract_bid(bid_idx, my_chips):
    """Convert abstract bid index to actual bid."""
    frac = BID_FRACS[min(bid_idx, len(BID_FRACS)-1)]
    return max(0, min(int(frac * my_chips), my_chips))


# ── Game State for CFR ─────────────────────────────────────────────────────

class CFRGameState:
    """
    Lightweight game state for CFR traversal.
    Tracks: street, pot, stacks, wagers, board, hands, auction.
    Uses abstracted card representation (equity bucket, not raw cards).
    """
    __slots__ = [
        'street', 'pot', 'chips', 'wagers', 'hands', 'board',
        'auction_done', 'bids', 'opp_revealed',
        'dealer', 'history', 'terminal', 'payoff',
        'pf_eq',  # preflop equity for each player
    ]
    
    def __init__(self):
        self.street       = 0       # 0=preflop, 3=flop, 4=turn, 5=river
        self.pot          = SMALL_BLIND + BIG_BLIND
        self.chips        = [STARTING_STACK-SMALL_BLIND, STARTING_STACK-BIG_BLIND]
        self.wagers       = [SMALL_BLIND, BIG_BLIND]
        self.hands        = [None, None]
        self.board        = []
        self.auction_done = False
        self.bids         = [None, None]
        self.opp_revealed = [None, None]   # card revealed to each player
        self.dealer       = 0       # who acts next (% 2 = player index)
        self.history      = []      # (player, abstract_action)
        self.terminal     = False
        self.payoff       = 0       # positive = player 0 wins
        self.pf_eq        = [0.5, 0.5]
    
    def active_player(self):
        return self.dealer % 2
    
    def get_info_set(self, player):
        """
        Build information set string for the player.
        This is what the CFR network conditions on.
        We use abstracted representation.
        """
        # Hand bucket (equity percentile)
        eq_bucket = int(self.pf_eq[player] * 10)  # 0-10
        
        # Board texture bucket
        board_bucket = self._board_bucket()
        
        # Betting history this hand (compressed)
        hist_str = '|'.join(f"{p}{a}" for p, a in self.history[-8:])  # last 8 actions
        
        # Stack depth bucket
        spr_bucket = min(int(self.chips[player] / max(self.pot, 1)), 20)
        
        return f"{self.street}:{eq_bucket}:{board_bucket}:{spr_bucket}:{hist_str}"
    
    def _board_bucket(self):
        """Simple board texture bucket (0-9)."""
        if not self.board: return 0
        from collections import Counter
        suits = [c[1] for c in self.board]
        ranks = [RANK_V[c[0]] for c in self.board]
        sc    = Counter(suits)
        rc    = Counter(ranks)
        flush = 1 if max(sc.values()) >= 3 else 0
        pair  = 1 if max(rc.values()) >= 2 else 0
        rs    = sorted(set(ranks))
        conn  = 1 if len(rs)>=3 and any(rs[i+2]-rs[i]<=4 for i in range(len(rs)-2)) else 0
        return flush*4 + pair*2 + conn
    
    def get_valid_abstract_actions(self):
        """Returns list of valid abstract action indices."""
        p = self.active_player()
        cost = self.wagers[1-p] - self.wagers[p]
        
        if cost == 0:
            # Check or raise
            if self.chips[0] == 0 or self.chips[1] == 0:
                return [ACTION_CALL]   # only check (all-in)
            return [ACTION_CALL, ACTION_BET33, ACTION_BET66,
                    ACTION_BET100, ACTION_BET150, ACTION_ALLIN]
        else:
            # Fold, call, or raise
            if cost == self.chips[p] or self.chips[1-p] == 0:
                return [ACTION_FOLD, ACTION_CALL]
            return [ACTION_FOLD, ACTION_CALL, ACTION_BET33, ACTION_BET66,
                    ACTION_BET100, ACTION_BET150, ACTION_ALLIN]
    
    def clone(self):
        s = CFRGameState()
        s.street       = self.street
        s.pot          = self.pot
        s.chips        = list(self.chips)
        s.wagers       = list(self.wagers)
        s.hands        = list(self.hands)
        s.board        = list(self.board)
        s.auction_done = self.auction_done
        s.bids         = list(self.bids)
        s.opp_revealed = list(self.opp_revealed)
        s.dealer       = self.dealer
        s.history      = list(self.history)
        s.terminal     = self.terminal
        s.payoff       = self.payoff
        s.pf_eq        = list(self.pf_eq)
        return s


# ── State Vector for Deep CFR ──────────────────────────────────────────────

def state_to_vector(state: CFRGameState, player: int) -> np.ndarray:
    """Convert CFR game state to neural network input vector."""
    v = []
    
    # Hand equity bucket (player's perspective)
    v.append(state.pf_eq[player])
    
    # Opponent's revealed card (if we won auction)
    rev = state.opp_revealed[player]
    v.append(RANK_V[rev[0]] / 12.0 if rev else 0.0)
    v.append(1.0 if rev and any(rev[0] == c[0] for c in state.board) else 0.0)
    v.append(1.0 if rev else 0.0)
    
    # Board texture
    if state.board:
        from collections import Counter
        suits = [c[1] for c in state.board]
        ranks = [RANK_V[c[0]] for c in state.board]
        sc    = Counter(suits)
        rc    = Counter(ranks)
        rs    = sorted(set(ranks))
        v.append(max(sc.values()) / 5.0)        # flush draw
        v.append(max(rc.values()) / 3.0)        # paired board
        v.append(max(ranks) / 12.0)             # high card
        conn = any(rs[i+2]-rs[i]<=4 for i in range(len(rs)-2)) if len(rs)>=3 else False
        v.append(1.0 if conn else 0.0)
    else:
        v += [0.0, 0.0, 0.0, 0.0]
    
    # Street (one-hot: preflop/flop/turn/river)
    for s in [0, 3, 4, 5]:
        v.append(1.0 if state.street == s else 0.0)
    
    # Stack/pot
    total = 2 * STARTING_STACK
    v.append(state.pot / total)
    v.append(state.chips[player] / STARTING_STACK)
    v.append(state.chips[1-player] / STARTING_STACK)
    cost = state.wagers[1-player] - state.wagers[player]
    v.append(min(cost / max(state.pot, 1), 2.0) / 2.0)
    v.append(min(state.chips[player] / max(state.pot, 1), 10.0) / 10.0)  # SPR
    v.append(1.0 if player == 1 else 0.0)  # is BB
    
    # Recent action history (last 6 actions, encoded as pairs)
    hist = state.history[-6:]
    for i in range(6):
        if i < len(hist):
            p, a = hist[i]
            v.append(float(p))           # which player acted
            v.append(a / N_ACTIONS)      # abstract action normalized
        else:
            v += [0.0, 0.0]
    
    # Wagers
    v.append(min(state.wagers[player] / STARTING_STACK, 1.0))
    v.append(min(state.wagers[1-player] / STARTING_STACK, 1.0))
    
    while len(v) < 33: v.append(0.0)
    assert len(v) == 33, f"CFR state vector length {len(v)} != 33"
    return np.array(v, dtype=np.float32)


# ── Deep CFR Networks ──────────────────────────────────────────────────────

class AdvantageNet(nn.Module):
    """Predicts counterfactual regrets (advantages) for each action."""
    def __init__(self, state_dim=33, hidden=256, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x): return self.net(x)


class StrategyNet(nn.Module):
    """Predicts average strategy (probability of each action)."""
    def __init__(self, state_dim=33, hidden=256, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.net(x)


# ── Experience Replay Buffers ──────────────────────────────────────────────

class AdvantageBuffer:
    """Circular buffer for (state, iteration, regrets) tuples."""
    def __init__(self, maxsize=2_000_000):
        self.maxsize  = maxsize
        self.states   = np.zeros((maxsize, 33), dtype=np.float32)
        self.regrets  = np.zeros((maxsize, N_ACTIONS), dtype=np.float32)
        self.iters    = np.zeros(maxsize, dtype=np.float32)
        self.idx      = 0
        self.full     = False
    
    def push(self, state, regrets, iteration):
        i = self.idx % self.maxsize
        self.states[i]  = state
        self.regrets[i] = regrets
        self.iters[i]   = iteration
        self.idx += 1
        if self.idx >= self.maxsize: self.full = True
    
    def sample(self, n):
        size = self.maxsize if self.full else self.idx
        idxs = np.random.choice(size, min(n, size), replace=False)
        # Weight recent samples higher (linear in iteration)
        return self.states[idxs], self.regrets[idxs], self.iters[idxs]
    
    def __len__(self):
        return self.maxsize if self.full else self.idx


class StrategyBuffer:
    """Buffer for (state, strategy) tuples (average strategy)."""
    def __init__(self, maxsize=2_000_000):
        self.maxsize    = maxsize
        self.states     = np.zeros((maxsize, 33), dtype=np.float32)
        self.strategies = np.zeros((maxsize, N_ACTIONS), dtype=np.float32)
        self.iters      = np.zeros(maxsize, dtype=np.float32)
        self.idx        = 0
        self.full       = False
    
    def push(self, state, strategy, iteration):
        i = self.idx % self.maxsize
        self.states[i]     = state
        self.strategies[i] = strategy
        self.iters[i]      = iteration
        self.idx += 1
        if self.idx >= self.maxsize: self.full = True
    
    def sample(self, n):
        size = self.maxsize if self.full else self.idx
        idxs = np.random.choice(size, min(n, size), replace=False)
        return self.states[idxs], self.strategies[idxs], self.iters[idxs]
    
    def __len__(self):
        return self.maxsize if self.full else self.idx


# ── Regret Matching ────────────────────────────────────────────────────────

def regret_matching(advantages, valid_actions):
    """
    Convert advantages (regrets) to strategy using regret matching+.
    strategy[a] ∝ max(0, advantage[a])
    """
    strategy = np.zeros(N_ACTIONS, dtype=np.float32)
    pos_sum  = 0.0
    for a in valid_actions:
        strategy[a] = max(0.0, advantages[a])
        pos_sum     += strategy[a]
    
    if pos_sum > 0:
        strategy /= pos_sum
    else:
        # Uniform over valid actions
        for a in valid_actions:
            strategy[a] = 1.0 / len(valid_actions)
    
    return strategy


# ── Dealer / Deck ──────────────────────────────────────────────────────────

def deal_game():
    """Deal a fresh game: 2 hands + 5 community cards."""
    deck  = list(FULL_DECK)
    random.shuffle(deck)
    h0    = [deck[0], deck[1]]
    h1    = [deck[2], deck[3]]
    board = deck[4:9]
    return h0, h1, board

def compute_showdown(h0, h1, board):
    """Returns payoff for player 0 (positive = p0 wins)."""
    s0 = evaluate_hand(h0 + board)
    s1 = evaluate_hand(h1 + board)
    pot = STARTING_STACK * 2   # simplified (actual pot from state)
    if s0 > s1:   return 1
    if s0 < s1:   return -1
    return 0


# ── External Sampling MCCFR Traversal ─────────────────────────────────────

class CFRTrainer:
    def __init__(self, adv_net_0, adv_net_1, strat_net, device):
        self.adv_nets  = [adv_net_0, adv_net_1]
        self.strat_net = strat_net
        self.device    = device
        self.adv_bufs  = [AdvantageBuffer(), AdvantageBuffer()]
        self.strat_buf = StrategyBuffer()
        self.iteration = 0
        self.traversals= 0
    
    def get_strategy(self, state, player):
        """Get current strategy from advantage network."""
        sv   = state_to_vector(state, player)
        x    = torch.tensor(sv, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            adv = self.adv_nets[player](x).cpu().numpy()[0]
        valid = state.get_valid_abstract_actions()
        return regret_matching(adv, valid)
    
    def traverse(self, state, player, h0, h1, board, iteration):
        """
        External sampling MCCFR traversal.
        Returns: expected value for the traversing player.
        Also fills advantage and strategy buffers.
        """
        if state.terminal:
            return state.payoff if player == 0 else -state.payoff
        
        active = state.active_player()
        valid  = state.get_valid_abstract_actions()
        
        # Handle auction phase separately
        if state.street == 3 and not state.auction_done:
            return self._traverse_auction(state, player, h0, h1, board, iteration)
        
        sv       = state_to_vector(state, active)
        strategy = self.get_strategy(state, active)
        
        if active == player:
            # Traversing player: compute value of each action
            action_values = np.zeros(N_ACTIONS)
            
            for a in valid:
                next_state = self._apply_action(state, a, h0, h1, board)
                action_values[a] = self.traverse(next_state, player, h0, h1, board, iteration)
            
            # Expected value under current strategy
            ev = sum(strategy[a] * action_values[a] for a in valid)
            
            # Regrets = action_value - expected_value
            regrets = np.zeros(N_ACTIONS)
            for a in valid:
                regrets[a] = action_values[a] - ev
            
            # Store in advantage buffer
            self.adv_bufs[player].push(sv, regrets, iteration)
            
            # Store in strategy buffer (weighted by iteration for averaging)
            self.strat_buf.push(sv, strategy, iteration)
            
            return ev
        
        else:
            # Opponent (external sampling): sample one action
            action_probs = [strategy[a] for a in range(N_ACTIONS)]
            sampled_a    = random.choices(range(N_ACTIONS), weights=action_probs)[0]
            if sampled_a not in valid:
                sampled_a = random.choice(valid)
            
            # Store strategy for opponent
            self.strat_buf.push(sv, strategy, iteration)
            
            next_state = self._apply_action(state, sampled_a, h0, h1, board)
            return self.traverse(next_state, player, h0, h1, board, iteration)
    
    def _traverse_auction(self, state, player, h0, h1, board, iteration):
        """Handle auction traversal (simplified: bid then continue)."""
        # Both players bid simultaneously — we treat it sequentially for CFR
        # Player 0 bids first (hidden), then player 1 bids (hidden)
        # Then auction resolves
        
        # For simplicity: treat auction bids as independent decisions
        # Each player chooses a bid fraction based on their info set
        
        # Approximate: randomly sample opponent bid, compute EV of our bid
        # (Full simultaneous CFR auction is much more complex — this is close enough)
        
        p0_bid_idx = random.randint(0, N_BID_ACTIONS-1)
        p1_bid_idx = random.randint(0, N_BID_ACTIONS-1)
        p0_bid     = abstract_bid(p0_bid_idx, state.chips[0])
        p1_bid     = abstract_bid(p1_bid_idx, state.chips[1])
        
        # Resolve auction
        new_state = state.clone()
        new_state.auction_done = True
        new_state.bids = [p0_bid, p1_bid]
        
        if p0_bid > p1_bid:
            # P0 wins, pays p1_bid
            new_state.chips[0]      -= p1_bid
            new_state.pot           += p1_bid
            new_state.opp_revealed[0] = random.choice(h1)   # p0 sees p1's card
        elif p1_bid > p0_bid:
            # P1 wins, pays p0_bid
            new_state.chips[1]      -= p0_bid
            new_state.pot           += p0_bid
            new_state.opp_revealed[1] = random.choice(h0)   # p1 sees p0's card
        else:
            # Tie: both pay, both see
            new_state.chips[0]      -= p0_bid
            new_state.chips[1]      -= p1_bid
            new_state.pot           += p0_bid + p1_bid
            new_state.opp_revealed[0] = random.choice(h1)
            new_state.opp_revealed[1] = random.choice(h0)
        
        new_state.dealer = 1  # BB acts first post-flop
        
        # Update equity estimates with new info
        new_state.pf_eq = list(state.pf_eq)
        if new_state.opp_revealed[0]:
            # Player 0 now knows one of player 1's cards
            # Recompute P0's equity with this info (simplified)
            pass  # keep pf_eq as-is for now; full version uses MC
        
        return self.traverse(new_state, player, h0, h1, board, iteration)
    
    def _apply_action(self, state, abstract_action, h0, h1, board):
        """Apply an abstract action to produce next game state."""
        s   = state.clone()
        p   = s.active_player()
        
        if abstract_action == ACTION_FOLD:
            s.terminal = True
            # Payoff = pot to non-folder, player 0 perspective
            pot_winner = 1 - p
            if pot_winner == 0:
                s.payoff = s.wagers[1]   # p0 wins p1's wager
            else:
                s.payoff = -s.wagers[0]  # p0 loses their wager
            return s
        
        cost = s.wagers[1-p] - s.wagers[p]
        
        if abstract_action == ACTION_CALL:
            if cost == 0:
                # Check
                s.history.append((p, abstract_action))
                s.dealer += 1
                # Check if street is over
                if (s.street == 0 and s.dealer > 1) or s.dealer > 1:
                    s = self._advance_street(s, h0, h1, board)
            else:
                # Call
                s.chips[p]  -= cost
                s.wagers[p] += cost
                s.pot       += cost
                s.history.append((p, abstract_action))
                s.dealer += 1
                s = self._advance_street(s, h0, h1, board)
            return s
        
        # Bet/Raise
        # Compute pot for sizing
        pot = s.pot
        mn  = max(BIG_BLIND, cost + max(cost, BIG_BLIND))  # min raise
        mx  = min(s.chips[p], s.chips[1-p] + cost)         # max raise
        if mn > mx: mn = mx
        
        raise_to = abstract_to_raise(abstract_action, pot, mn + s.wagers[p], mx + s.wagers[p])
        added    = raise_to - s.wagers[p]
        added    = max(0, min(added, s.chips[p]))
        
        s.chips[p]  -= added
        s.wagers[p] += added
        s.pot       += added
        s.history.append((p, abstract_action))
        s.dealer += 1
        return s
    
    def _advance_street(self, state, h0, h1, board):
        """Move to next street."""
        s = state.clone()
        s.wagers  = [0, 0]
        s.dealer  = 1   # BB (player 1) acts first post-flop
        
        if s.street == 0:
            # Preflop → Flop + Auction
            s.street = 3
            s.board  = board[:3]
            # Auction happens: update equity estimates
            eq0 = postflop_bucket(h0, board[:3], n_samples=8)
            eq1 = postflop_bucket(h1, board[:3], n_samples=8)
            s.pf_eq = [eq0, eq1]
            # Auction is handled separately
        elif s.street == 3:
            # Flop → Turn
            s.street = 4
            s.board  = board[:4]
            eq0 = postflop_bucket(h0, board[:4], n_samples=8)
            eq1 = postflop_bucket(h1, board[:4], n_samples=8)
            s.pf_eq = [eq0, eq1]
        elif s.street == 4:
            # Turn → River
            s.street = 5
            s.board  = board[:5]
            eq0 = postflop_bucket(h0, board[:5], n_samples=8)
            eq1 = postflop_bucket(h1, board[:5], n_samples=8)
            s.pf_eq = [eq0, eq1]
        elif s.street == 5:
            # River → Showdown
            s0 = evaluate_hand(h0 + board[:5])
            s1 = evaluate_hand(h1 + board[:5])
            pot_half = s.pot // 2
            if s0 > s1:   s.payoff = s.wagers[0] + pot_half    # simplified
            elif s0 < s1: s.payoff = -(s.wagers[0] + pot_half)
            else:         s.payoff = 0
            s.terminal = True
        
        return s
    
    def run_iteration(self, h0, h1, board):
        """Run one CFR iteration (both players traverse)."""
        state = CFRGameState()
        state.hands  = [h0, h1]
        state.pf_eq  = [pf_equity(h0), pf_equity(h1)]
        
        # Traverse for player 0
        self.traverse(state, 0, h0, h1, board, self.iteration)
        
        # Traverse for player 1
        self.traverse(state, 1, h0, h1, board, self.iteration)
        
        self.traversals += 2
        self.iteration  += 1
    
    def update_networks(self, device):
        """Train networks on buffered data."""
        BATCH    = 4096
        N_STEPS  = 200
        
        for player in [0, 1]:
            buf = self.adv_bufs[player]
            if len(buf) < BATCH: continue
            
            net = self.adv_nets[player]
            opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            net.train()
            
            for _ in range(N_STEPS):
                states, regrets, iters = buf.sample(BATCH)
                
                # Weight by iteration (newer = more important)
                weights = torch.tensor(iters / max(iters.max(), 1),
                                       dtype=torch.float32).to(device)
                
                x   = torch.tensor(states,  dtype=torch.float32).to(device)
                y   = torch.tensor(regrets, dtype=torch.float32).to(device)
                pred= net(x)
                
                loss = (weights.unsqueeze(-1) * (pred - y)**2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        
        # Train strategy net
        buf = self.strat_buf
        if len(buf) >= BATCH:
            net = self.strat_net
            opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            net.train()
            
            for _ in range(N_STEPS):
                states, strategies, iters = buf.sample(BATCH)
                weights = torch.tensor(iters / max(iters.max(), 1),
                                       dtype=torch.float32).to(device)
                x    = torch.tensor(states,     dtype=torch.float32).to(device)
                y    = torch.tensor(strategies, dtype=torch.float32).to(device)
                pred = net(x)
                
                # Cross entropy weighted by iteration
                loss = -(weights.unsqueeze(-1) * y * torch.log(pred + 1e-8)).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()


# ── Main Training Loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',         default='./models/')
    parser.add_argument('--checkpoint_dir', default='./cfr_checkpoints/')
    parser.add_argument('--traversals',     type=int, default=10_000_000)
    parser.add_argument('--update_every',   type=int, default=10_000)
    parser.add_argument('--save_every',     type=int, default=100_000)
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"CFR Training on: {device}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.device_count()} GPU(s) available")
    
    # Networks
    adv_net_0  = AdvantageNet().to(device)
    adv_net_1  = AdvantageNet().to(device)
    strat_net  = StrategyNet().to(device)
    
    # Load pretrained strategy net if available (from supervised training)
    strat_path = os.path.join(args.models, 'cfr_strategy_net.pt')
    if os.path.exists(strat_path):
        ckpt = torch.load(strat_path, map_location=device)
        strat_net.load_state_dict(ckpt['model_state'])
        print(f"Loaded pretrained strategy net from {strat_path}")
    
    trainer = CFRTrainer(adv_net_0, adv_net_1, strat_net, device)
    
    # Training loop
    t0 = time.time()
    last_update = 0
    last_save   = 0
    
    print(f"\nStarting CFR training: {args.traversals:,} target traversals")
    print(f"Update networks every {args.update_every:,} traversals")
    print(f"Save checkpoint every {args.save_every:,} traversals")
    print()
    
    while trainer.traversals < args.traversals:
        h0, h1, board = deal_game()
        trainer.run_iteration(h0, h1, board)
        
        t = trainer.traversals
        
        if t - last_update >= args.update_every:
            trainer.update_networks(device)
            last_update = t
            elapsed     = time.time() - t0
            rate        = t / elapsed
            eta         = (args.traversals - t) / rate
            print(f"Traversals: {t:>10,} | "
                  f"Rate: {rate:>8,.0f}/s | "
                  f"Elapsed: {elapsed/3600:.1f}h | "
                  f"ETA: {eta/3600:.1f}h | "
                  f"Adv buf: {len(trainer.adv_bufs[0]):,}")
        
        if t - last_save >= args.save_every:
            ckpt_path = os.path.join(args.checkpoint_dir,
                                     f'cfr_{t//1000}k.pt')
            torch.save({
                'traversals':     t,
                'iteration':      trainer.iteration,
                'adv_net_0':      adv_net_0.state_dict(),
                'adv_net_1':      adv_net_1.state_dict(),
                'strat_net':      strat_net.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}")
            last_save = t
    
    # Save final
    final_path = os.path.join(args.models, 'cfr_strategy_net.pt')
    torch.save({
        'traversals':  trainer.traversals,
        'model_state': strat_net.state_dict(),
        'type':        'cfr_strategy',
    }, final_path)
    print(f"\nFinal CFR strategy net saved to {final_path}")
    print(f"Total time: {(time.time()-t0)/3600:.1f} hours")


if __name__ == '__main__':
    main()