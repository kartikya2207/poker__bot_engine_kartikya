import time
import random
from collections import Counter, deque
from itertools import combinations
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

STARTING_STACK = 5000
BB = 20
RANKS = '23456789TJQKA'
RV = {r: i for i, r in enumerate(RANKS)}
DECK = [r+s for r in RANKS for s in 'cdhs']

PF = {
    'AA':0.852,'KK':0.827,'QQ':0.794,'JJ':0.772,'TT':0.744,'99':0.717,
    '88':0.698,'77':0.668,'66':0.643,'55':0.616,'44':0.568,'33':0.547,'22':0.498,
    'AKs':0.684,'AQs':0.649,'AJs':0.647,'ATs':0.645,'A9s':0.646,'A8s':0.632,
    'A7s':0.603,'A6s':0.600,'A5s':0.607,'A4s':0.583,'A3s':0.573,'A2s':0.567,
    'AKo':0.648,'AQo':0.636,'AJo':0.643,'ATo':0.615,'A9o':0.602,'A8o':0.616,
    'A7o':0.589,'A6o':0.559,'A5o':0.583,'A4o':0.578,'A3o':0.555,'A2o':0.538,
    'KQs':0.622,'KJs':0.630,'KTs':0.625,'K9s':0.616,'K8s':0.598,'K7s':0.565,
    'K6s':0.571,'K5s':0.561,'K4s':0.541,'K3s':0.544,'K2s':0.513,
    'KQo':0.605,'KJo':0.604,'KTo':0.586,'K9o':0.565,'K8o':0.562,'K7o':0.557,
    'K6o':0.526,'K5o':0.537,'K4o':0.521,'K3o':0.515,'K2o':0.516,
    'QJs':0.608,'QTs':0.574,'Q9s':0.581,'Q8s':0.557,'Q7s':0.532,'Q6s':0.534,
    'Q5s':0.522,'Q4s':0.506,'Q3s':0.509,'Q2s':0.503,
    'QJo':0.573,'QTo':0.573,'Q9o':0.562,'Q8o':0.541,'Q7o':0.503,'Q6o':0.503,
    'Q5o':0.493,'Q4o':0.498,'Q3o':0.492,'Q2o':0.479,
    'JTs':0.573,'J9s':0.535,'J8s':0.557,'J7s':0.516,'J6s':0.521,'J5s':0.497,
    'J4s':0.485,'J3s':0.490,'J2s':0.470,
    'JTo':0.545,'J9o':0.524,'J8o':0.514,'J7o':0.496,'J6o':0.481,'J5o':0.452,
    'J4o':0.464,'J3o':0.444,'J2o':0.439,
    'T9s':0.522,'T8s':0.516,'T7s':0.493,'T6s':0.483,'T5s':0.460,'T4s':0.468,
    'T3s':0.468,'T2s':0.438,'T9o':0.510,'T8o':0.487,'T7o':0.492,'T6o':0.443,
    'T5o':0.443,'T4o':0.449,'T3o':0.428,'T2o':0.414,
    '98s':0.509,'97s':0.466,'96s':0.477,'95s':0.464,'94s':0.425,'93s':0.439,'92s':0.407,
    '98o':0.471,'97o':0.460,'96o':0.443,'95o':0.422,'94o':0.411,'93o':0.402,'92o':0.392,
    '87s':0.470,'86s':0.472,'85s':0.450,'84s':0.420,'83s':0.415,'82s':0.411,
    '87o':0.437,'86o':0.427,'85o':0.410,'84o':0.393,'83o':0.371,'82o':0.368,
    '76s':0.468,'75s':0.454,'74s':0.409,'73s':0.375,'72s':0.376,
    '76o':0.413,'75o':0.396,'74o':0.378,'73o':0.384,'72o':0.350,
    '65s':0.440,'64s':0.415,'63s':0.412,'62s':0.386,
    '65o':0.398,'64o':0.379,'63o':0.358,'62o':0.356,
    '54s':0.407,'53s':0.389,'52s':0.392,'54o':0.376,'53o':0.360,'52o':0.345,
    '43s':0.390,'42s':0.378,'43o':0.364,'42o':0.333,'32s':0.360,'32o':0.325,
}

def score5(cards):
    rs = sorted([RV[c[0]] for c in cards], reverse=True)
    fl = len(set(c[1] for c in cards)) == 1
    ur = set(rs); st = 0; st_hi = 0
    if len(ur) == 5:
        if rs[0] - rs[4] == 4: st, st_hi = 1, rs[0]
        elif ur == {12,3,2,1,0}: st, st_hi = 1, 3
    cn = Counter(rs); fr = sorted(cn.values(), reverse=True)
    def pk(cat, tb):
        v = cat * (15**6)
        for i,r in enumerate(tb[:6]): v += r * (15**(5-i))
        return v
    if st and fl: return pk(8, [st_hi])
    if fr[0]==4:
        q = next(r for r,f in cn.items() if f==4)
        k = next(r for r,f in cn.items() if f==1)
        return pk(7, [q,k])
    if fr[0]==3 and len(fr)>1 and fr[1]==2:
        return pk(6, [next(r for r,f in cn.items() if f==3), next(r for r,f in cn.items() if f==2)])
    if fl: return pk(5, rs)
    if st: return pk(4, [st_hi])
    if fr[0]==3:
        t = next(r for r,f in cn.items() if f==3)
        ks = sorted((r for r,f in cn.items() if f==1), reverse=True)
        return pk(3, [t]+ks)
    if fr[0]==2 and len(fr)>1 and fr[1]==2:
        ps = sorted((r for r,f in cn.items() if f==2), reverse=True)
        k = next(r for r,f in cn.items() if f==1)
        return pk(2, ps+[k])
    if fr[0]==2:
        p = next(r for r,f in cn.items() if f==2)
        ks = sorted((r for r,f in cn.items() if f==1), reverse=True)
        return pk(1, [p]+ks)
    return pk(0, rs)

def best_hand(cards):
    if len(cards) <= 5: return score5(cards)
    return max(score5(list(c)) for c in combinations(cards, 5))

def exact_equity(my_hand, opp_card, board, samples=60):
    known = set(my_hand + [opp_card] + board)
    deck = [c for c in DECK if c not in known]
    need = 5 - len(board)
    wins = ties = total = 0
    for _ in range(samples):
        if len(deck) < need + 1: break
        draw = random.sample(deck, need + 1)
        opp_hand = [opp_card, draw[0]]
        full_board = board + draw[1:]
        ms = best_hand(my_hand + full_board)
        os = best_hand(opp_hand + full_board)
        if ms > os: wins += 1
        elif ms == os: ties += 1
        total += 1
    return (wins + 0.5*ties) / total if total else 0.5

def no_info_equity(my_hand, board, samples=60):
    known = set(my_hand + board)
    deck = [c for c in DECK if c not in known]
    need = 5 - len(board)
    wins = ties = total = 0
    for _ in range(samples):
        if len(deck) < need + 2: break
        draw = random.sample(deck, need + 2)
        opp_hand = draw[:2]
        full_board = board + draw[2:]
        ms = best_hand(my_hand + full_board)
        os = best_hand(opp_hand + full_board)
        if ms > os: wins += 1
        elif ms == os: ties += 1
        total += 1
    return (wins + 0.5*ties) / total if total else 0.5

def preflop_equity(h):
    r1,r2 = RV[h[0][0]], RV[h[1][0]]
    s1,s2 = h[0][1], h[1][1]
    if r1 < r2: r1,r2,s1,s2 = r2,r1,s2,s1
    key = RANKS[r1]+RANKS[r2]
    if r1 != r2: key += ('s' if s1==s2 else 'o')
    return PF.get(key, 0.45)

class OppStats:
    def __init__(self):
        self.hands = 0; self.vpip = 0
        self.agg_acts = 0; self.all_acts = 0
        self.bids = deque(maxlen=300)
    def new_hand(self): self.hands += 1
    @property
    def n(self): return max(self.hands, 1)
    @property
    def vpip_r(self): return self.vpip / self.n
    @property
    def af(self): return self.agg_acts / max(self.all_acts - self.agg_acts, 1)
    @property
    def avg_bid(self): return sum(self.bids)/len(self.bids) if self.bids else 80.0
    @property
    def calibrated(self): return self.hands >= 10
    @property
    def is_fish(self): return self.calibrated and self.vpip_r > 0.60
    @property
    def is_nit(self): return self.calibrated and self.vpip_r < 0.35
    @property
    def is_agg(self): return self.calibrated and self.af > 0.6
    @property
    def is_passive(self): return self.calibrated and self.af < 0.25

class Player(BaseBot):
    def __init__(self):
        self.opp = OppStats()
        self._eq_cache = {}
        self._opp_card = None
        self._pf_eq = 0.5
        self._raised_pf = False
        self._street_bets = 0

    def on_hand_start(self, gi, cs):
        self.opp.new_hand()
        try:
            self._pf_eq = preflop_equity(cs.my_hand)
        except Exception:
            self._pf_eq = 0.50
        self._eq_cache = {}
        self._opp_card = None
        self._raised_pf = False
        self._street_bets = 0
        self._we_lost_auction = False
        self._last_bid = 0

    def on_hand_end(self, gi, cs): pass

    def get_move(self, gi, cs):
        try:
            rv = cs.opp_revealed_cards
            my_hand_set = set(cs.my_hand) if hasattr(cs, 'my_hand') else set()
            if rv:
                opp_cards = [c for c in rv if c not in my_hand_set]  # CRITICAL: ignore our own cards
                if self._opp_card is None and opp_cards:
                    self._opp_card = opp_cards[0]
                    self._eq_cache = {}
                    if self._last_bid > 0:
                        self.opp.bids.append(max(0, self._last_bid - 5))
            elif cs.street not in ('pre-flop', 'auction') and self._opp_card is None:
                self._we_lost_auction = True
                if self._last_bid > 0:
                    self.opp.bids.append(self._last_bid + 10)
        except: pass
        s = cs.street
        if s == 'auction':  return self._auction(cs)
        if s == 'pre-flop': return self._preflop(cs)
        return self._postflop(cs, s)

    def _equity(self, cs):
        street = cs.street
        if street not in self._eq_cache:
            board = cs.board
            t0 = time.time()
            try:
                if self._opp_card:
                    eq = exact_equity(cs.my_hand, self._opp_card, board, samples=60)
                else:
                    eq = no_info_equity(cs.my_hand, board, samples=60)
                # Fallback if calc took too long
                if time.time() - t0 > 0.8:
                    eq = self._pf_eq
            except Exception:
                eq = self._pf_eq
            self._eq_cache[street] = eq
        return self._eq_cache[street]

    def _auction(self, cs):
        eq = self._pf_eq
        avg = self.opp.avg_bid
        pot = max(cs.pot, 20)

        # FIX: use pot-based cap, NOT my_chips (which can be 0 after all-in)
        # Max bid = 20% of pot is reasonable; never more than 1000 unless premium
        hi = max(int(pot * 2.0), 50)   # generous upper bound
        lo = 0

        # Scale aggressively - we were losing every auction to bids of 100-1500
        # Winning auction on strong hand = massive EV edge
        if eq >= 0.72:
            bid = max(int(avg * 2.0) + 60, 120)
        elif eq >= 0.62:
            bid = max(int(avg * 1.6) + 30, 70)
        elif eq >= 0.52:
            bid = max(int(avg * 1.2) + 15, 40)
        elif eq >= 0.42:
            bid = max(int(avg * 0.7), 20)
        else:
            # Weak hand: low bid, let them waste chips
            bid = max(int(avg * 0.3), 5)

        bid = max(lo, min(bid, hi))
        # Track our own bid history to adapt next time
        self._last_bid = bid
        return ActionBid(bid)

    def _preflop(self, cs):
        eq = self._pf_eq
        ctc = cs.cost_to_call
        pot = max(cs.pot, 40)
        mn, mx = cs.raise_bounds if cs.can_act(ActionRaise) else (0,0)
        facing_raise = ctc > 20
        if facing_raise: self.opp.vpip += 1

        def raise_to(mult):
            return ActionRaise(max(mn, min(int(pot * mult), mx)))
        def shove():
            return ActionRaise(mx) if cs.can_act(ActionRaise) else ActionCall()

        if facing_raise:
            po = ctc / (pot + ctc)
            raise_size = ctc / 20.0  # how many BBs is the raise?
            if eq >= 0.68:
                # Premium: 3-bet or shove
                self._raised_pf = True
                return shove() if raise_size > 5 else raise_to(3.0)
            elif eq >= 0.58:
                # Strong: 3-bet small raises, call medium
                self._raised_pf = True
                return raise_to(2.5) if raise_size < 3 else ActionCall()
            elif eq >= max(0.44, po + 0.04):
                # Decent hand getting good price: call
                return ActionCall()
            else:
                return ActionFold()

        if cs.is_bb and ctc == 0:
            if eq >= 0.60: self._raised_pf = True; return raise_to(3.0)
            return ActionCheck()

        if eq >= 0.62:
            self._raised_pf = True; return raise_to(2.5)
        if eq >= 0.44:
            return ActionCall()
        return ActionFold()

    def _postflop(self, cs, street):
        eq = self._equity(cs)
        pot = max(cs.pot, 40)
        ctc = cs.cost_to_call
        mn, mx = cs.raise_bounds if cs.can_act(ActionRaise) else (0, 0)
        def shove(): return ActionRaise(mx) if cs.can_act(ActionRaise) else ActionCall()
        def bet(r): return ActionRaise(max(mn, min(int(pot * r), mx)))
        def chk(): return ActionCheck() if cs.can_act(ActionCheck) else ActionFold()
        def fold(): return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
        facing_bet = ctc > 0
        po = ctc / (pot + ctc) if facing_bet else 0

        if facing_bet:
            self.opp.agg_acts += 1
            self.opp.all_acts += 1
            # RAISE OR FOLD — never passively call postflop.
            # Data: postflop call win rate = 35% across 695 pots = massive bleed.
            # Pure strategy: if we're ahead, get all the chips in. If behind, lose 0.
            if eq >= 0.65:
                return shove() if cs.can_act(ActionRaise) else ActionFold()
            elif eq >= 0.52 and po < 0.18 and ctc < 60:
                # Exception: tiny bet into big pot, pot odds too good to fold
                return shove() if cs.can_act(ActionRaise) else ActionFold()
            else:
                return fold()

        # Not facing a bet — bet for value, check/give up otherwise
        self.opp.all_acts += 1
        if not cs.can_act(ActionRaise):
            return chk()
        if eq >= 0.65:
            if street == 'river': return shove()
            return bet(0.80)
        if eq >= 0.55:
            return bet(0.50)
        return chk()


if __name__ == '__main__':
    run_bot(Player(), parse_args())