"""
Run this AFTER downloading cfr_table.pkl from Colab.
Embeds the CFR table into the bot as a single uploadable file.

Usage: python make_cfr_bot.py
Output: bot_final.py  ← upload this to competition
"""
import pickle, base64, zlib, os

# Load CFR table
with open('./models/cfr_table.pkl', 'rb') as f:
    data = pickle.load(f)

table = data['table']
iters = data['iterations']
print(f"Loaded CFR table: {len(table):,} states, {iters:,} iterations")

# Compress and encode
raw       = pickle.dumps(table)
compressed= zlib.compress(raw, level=9)
encoded   = base64.b85encode(compressed).decode()
print(f"Compressed: {len(raw)/1024:.0f} KB → {len(compressed)/1024:.0f} KB")

bot_code = f'''# GHOST BOT — CFR Edition
# CFR table: {len(table):,} states, {iters:,} iterations trained
import random, pickle, base64, zlib, os, tempfile
from collections import defaultdict, Counter
from itertools import combinations
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

# ── Load embedded CFR table ──────────────────────────────────
_CFR_DATA = {repr(encoded)}

def _load_table():
    try:
        raw = zlib.decompress(base64.b85decode(_CFR_DATA.encode()))
        return pickle.loads(raw)
    except Exception:
        return {{}}

_CFR_TABLE = _load_table()
print(f"[GhostBot] CFR table loaded: {{len(_CFR_TABLE):,}} states")

# ── Constants ────────────────────────────────────────────────
STARTING_STACK = 5000
BB = 20
RANKS = '23456789TJQKA'
RV = {{r: i for i, r in enumerate(RANKS)}}
DECK = [r+s for r in RANKS for s in 'cdhs']
FOLD=0; CHECK=1; CALL=2; BET33=3; BET66=4; BET100=5; ALLIN=6
N_ACTIONS = 7

PF = {{
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
}}

def hk(h):
    r1,r2=RV[h[0][0]],RV[h[1][0]]
    s1,s2=h[0][1],h[1][1]
    if r1<r2: r1,r2,s1,s2=r2,r1,s2,s1
    b=RANKS[r1]+RANKS[r2]
    return b if r1==r2 else b+('s' if s1==s2 else 'o')

def pfeq(h): return PF.get(hk(h), 0.45)

def postflop_eq(my_hand, board, opp_card=None):
    all_c = my_hand+board
    ranks = [RV[c[0]] for c in all_c]
    suits = [c[1] for c in all_c]
    rc=Counter(ranks); sc=Counter(suits)
    freq=sorted(rc.values(),reverse=True)
    if freq[0]>=4: hs=0.97
    elif freq[0]==3 and len(freq)>1 and freq[1]>=2: hs=0.94
    elif max(sc.values())>=5: hs=0.91
    elif freq[0]==3: hs=0.83
    elif freq[0]==2 and len(freq)>1 and freq[1]==2:
        p=sorted([r for r,n in rc.items() if n==2],reverse=True)
        hs=0.68+p[0]/60.
    elif freq[0]==2:
        pr=max(r for r,n in rc.items() if n==2)
        hs=0.50+pr/36.
    else: hs=0.28+max(ranks)/20.
    my_s=[c[1] for c in my_hand]; bd_s=[c[1] for c in board]
    for s in set(my_s):
        if bd_s.count(s)+my_s.count(s)>=4: hs=max(hs,hs+0.07)
    ur=sorted(set(ranks))
    for i in range(len(ur)-4):
        if ur[i+4]-ur[i]==4: hs=max(hs,0.89); break
    if set([12,0,1,2,3]).issubset(set(ranks)): hs=max(hs,0.89)
    if opp_card:
        or_=RV[opp_card[0]]; br=[RV[c[0]] for c in board]
        if or_ in br: hs-=0.10
        elif or_>=10 and freq[0]<2: hs-=0.06
        elif or_<=5 and freq[0]>=2: hs+=0.04
    return max(0.05,min(0.97,hs))

def eq_bucket(eq):
    for i,t in enumerate([0.30,0.38,0.46,0.54,0.64,0.76,0.88]):
        if eq<t: return i
    return 7

def board_bucket(board):
    if not board: return 0
    sc=Counter(c[1] for c in board)
    rk=sorted(set(RV.get(c[0],0) for c in board))
    w=0
    if max(sc.values())>=3: w+=2
    elif max(sc.values())>=2: w+=1
    if len(rk)>=3 and any(rk[i+2]-rk[i]<=4 for i in range(len(rk)-2)): w+=1
    return min(w,2)

def info_set_key(street, my_eq_b, board_b, is_bb, has_opp_card, facing_bet, history):
    hist=tuple(history[-2:]) if len(history)>=2 else tuple(history)
    return (street, my_eq_b, board_b, int(is_bb), int(has_opp_card), int(facing_bet), hist)

def cfr_action(key, valid, fallback_eq, facing_bet):
    """Look up CFR strategy, fall back to equity-based if not found."""
    if key in _CFR_TABLE:
        probs = _CFR_TABLE[key]
        valid_probs = [(a, probs[a]) for a in valid if a < len(probs)]
        total = sum(p for _,p in valid_probs)
        if total > 0:
            roll = random.random(); cumul = 0.
            for a, p in valid_probs:
                cumul += p/total
                if roll <= cumul: return a
    # Fallback: simple equity thresholds
    eq = fallback_eq
    if facing_bet:
        if eq>=0.72 and BET66 in valid: return BET66
        if eq>=0.54 and CALL in valid: return CALL
        if eq>=0.44 and CALL in valid: return CALL
        return FOLD if FOLD in valid else CHECK
    else:
        if eq>=0.76 and BET66 in valid: return BET66
        if eq>=0.60 and BET33 in valid: return BET33
        if eq>=0.50 and BET33 in valid and random.random()<0.4: return BET33
        return CHECK if CHECK in valid else CALL

def to_action(a, cs):
    pot=max(cs.pot,40)
    mn,mx=cs.raise_bounds if cs.can_act(ActionRaise) else (0,0)
    def b(r): return ActionRaise(max(mn,min(int(pot*r),mx)))
    if a==FOLD:    return ActionFold() if cs.can_act(ActionFold) else ActionCheck()
    if a==CHECK:   return ActionCheck() if cs.can_act(ActionCheck) else ActionFold()
    if a==CALL:    return ActionCall() if cs.can_act(ActionCall) else ActionCheck()
    if a==BET33:   return b(0.33)
    if a==BET66:   return b(0.66)
    if a==BET100:  return b(1.00)
    if a==ALLIN:
        return ActionRaise(max(mn,min(cs.my_chips,mx))) if cs.can_act(ActionRaise) else ActionCall()
    return ActionCheck()

def get_valid(cs, facing_bet):
    v=[]
    if cs.can_act(ActionFold):  v.append(FOLD)
    if cs.can_act(ActionCheck): v.append(CHECK)
    if cs.can_act(ActionCall):  v.append(CALL)
    if cs.can_act(ActionRaise):
        v+=[BET33,BET66]
        if cs.street=='river': v.append(BET100)
        if cs.my_chips<=max(cs.pot,40)*1.5: v.append(ALLIN)
    return v if v else [CHECK]

from collections import deque
class Opp:
    def __init__(self):
        self.h=0; self.vp=0; self.fp=0
        self.bets=0; self.acts=0
        self.bids=deque(maxlen=100)
        self.ftb=0; self.fb=0
    def new(self): self.h+=1
    @property
    def n(self): return max(self.h,1)
    @property
    def vpr(self): return self.vp/self.n
    @property
    def fpr(self): return self.fp/self.n
    @property
    def af(self): return self.bets/max(self.acts,1)
    @property
    def fold_to_bet(self): return self.ftb/max(self.fb,1)
    @property
    def abid(self): return sum(self.bids)/len(self.bids) if self.bids else 15.
    @property
    def cal(self): return self.h>=20
    @property
    def fish(self): return self.cal and self.vpr>0.62
    @property
    def nit(self): return self.cal and self.fpr>0.45
    @property
    def agg(self): return self.cal and self.af>0.55

class Player(BaseBot):
    def __init__(self):
        self.opp=Opp()
        self.pf=0.5; self.eqc={{}}
        self.oc=[]; self.hi=False
        self.rpf=False; self.lb=15
        self.hist=(); self.hb=0

    def on_hand_start(self,gi,cs):
        self.opp.new()
        self.pf=pfeq(cs.my_hand)
        self.eqc={{}}; self.oc=[]; self.hi=False
        self.rpf=False; self.lb=15
        self.hist=(); self.hb=0

    def on_hand_end(self,gi,cs):
        try:
            self.opp.bids.append(self.lb)
            if cs.opp_revealed_cards: pass
        except: pass

    def get_move(self,gi,cs):
        try:
            rv=cs.opp_revealed_cards
            if rv and not self.oc:
                self.oc=list(rv); self.hi=True; self.eqc={{}}
        except: pass
        s=cs.street
        if s=='auction': return self._bid(cs)
        if s=='pre-flop': return self._pf(cs)
        return self._post(cs,s)

    def _bid(self,cs):
        avg=self.opp.abid; pf=self.pf
        iv=1.0-abs(2*pf-1.0)
        bid=int((avg*1.35+5)*(0.80+iv*0.65))
        bid=min(bid,int(cs.my_chips*0.15),int(max(cs.pot,30)*0.30))
        self.lb=max(bid,5)
        return ActionBid(self.lb)

    def _geq(self,street,cs):
        if street not in self.eqc:
            self.eqc[street]=postflop_eq(
                cs.my_hand,cs.board,
                self.oc[0] if self.hi and self.oc else None)
        return self.eqc[street]

    def _decide(self,cs,street,eq,facing_bet):
        board_b=board_bucket(cs.board)
        eq_b=eq_bucket(eq)
        key=info_set_key(street,eq_b,board_b,cs.is_bb,
                         self.hi,facing_bet,self.hist)
        valid=get_valid(cs,facing_bet)
        a=cfr_action(key,valid,eq,facing_bet)

        # Opponent exploitation override
        o=self.opp
        if o.cal:
            if o.fish and not facing_bet and eq<0.46:
                a=CHECK if CHECK in valid else a  # no bluffs vs fish
            elif o.nit and not facing_bet and eq<0.44 and street in ('flop','turn'):
                if BET33 in valid and random.random()<0.35:
                    a=BET33  # bluff vs nit
            elif o.agg and facing_bet and eq<0.42:
                a=FOLD if FOLD in valid else a   # tighten vs aggro

        self.hist=self.hist+(a,)
        return to_action(a,cs)

    def _pf(self,cs):
        facing=cs.cost_to_call>BB
        if facing:
            self.opp.vp+=1; self.opp.bets+=1; self.opp.acts+=1
        act=self._decide(cs,'pre-flop',self.pf,facing)
        if isinstance(act,ActionRaise): self.rpf=True
        elif isinstance(act,ActionFold) and facing: self.opp.fp+=1
        return act

    def _post(self,cs,street):
        eq=self._geq(street,cs)
        facing=cs.cost_to_call>0
        if facing:
            self.opp.bets+=1; self.opp.acts+=1
            self.opp.fb+=1; self.hb+=1
        else:
            self.opp.acts+=1
        act=self._decide(cs,street,eq,facing)
        if isinstance(act,ActionFold) and facing:
            self.opp.ftb+=1
        return act

if __name__==\'__main__\':
    run_bot(Player(),parse_args())
'''

# Write bot file
with open('bot_final.py', 'w', encoding='utf-8') as f:   
    f.write(bot_code)

print(f"bot_final.py written ({os.path.getsize('bot_final.py')/1024:.0f} KB)")
print("Upload bot_final.py to competition!")