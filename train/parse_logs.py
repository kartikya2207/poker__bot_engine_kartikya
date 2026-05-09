"""
parse_logs.py - Fixed for actual log format
Format: BotA vs BotB, Round #N, BotX received [cards], etc.
"""

import os, re, gzip, pickle, argparse
import numpy as np
from collections import defaultdict, Counter

RANKS     = '23456789TJQKA'
SUITS     = 'cdhs'
RANK_V    = {r: i for i, r in enumerate(RANKS)}
FULL_DECK = [r+s for r in RANKS for s in SUITS]
STARTING_STACK = 5000
BIG_BLIND      = 20

PREFLOP_EQ = {
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

def hand_key(cards):
    if len(cards) < 2: return 'XX'
    r1,r2 = RANK_V.get(cards[0][0],6), RANK_V.get(cards[1][0],6)
    s1,s2 = cards[0][1], cards[1][1]
    if r1 < r2: r1,r2,s1,s2 = r2,r1,s2,s1
    c1,c2 = RANKS[r1], RANKS[r2]
    if r1==r2: return c1+c2
    return c1+c2+('s' if s1==s2 else 'o')

def preflop_eq(cards):
    return PREFLOP_EQ.get(hand_key(cards), 0.45)

def board_wetness(board):
    if not board: return 0.0
    sc = Counter(c[1] for c in board)
    ranks = sorted(set(RANK_V.get(c[0],0) for c in board))
    w = 0.0
    if max(sc.values()) >= 3: w += 0.4
    elif max(sc.values()) >= 2: w += 0.2
    if len(ranks)>=3 and any(ranks[i+2]-ranks[i]<=4 for i in range(len(ranks)-2)): w+=0.3
    return min(1.0, w)

def build_state_vector(hand, board, pot, my_chips, opp_chips,
                        my_wager, opp_wager, street, is_bb,
                        auction_won, round_num=1):
    v = []
    pf_eq  = preflop_eq(hand) if hand else 0.45
    suited = 1.0 if (hand and len(hand)>=2 and hand[0][1]==hand[1][1]) else 0.0
    is_pair= 1.0 if (hand and len(hand)>=2 and hand[0][0]==hand[1][0]) else 0.0
    hi_rank= (RANK_V.get(hand[0][0],6) if hand else 6)/12.0
    v += [pf_eq, suited, is_pair, hi_rank]
    bf_wet = board_wetness(board)
    board_ranks = [RANK_V.get(c[0],0) for c in board]
    board_suits = [c[1] for c in board]
    sc = Counter(board_suits) if board_suits else Counter({'x':0})
    rc = Counter(board_ranks) if board_ranks else Counter({0:0})
    v += [
        bf_wet,
        1.0 if max(rc.values())>=2 else 0.0,
        1.0 if board_suits and max(sc.values())==len(board) else 0.0,
        1.0 if board_suits and max(sc.values())>=2 else 0.0,
        max(board_ranks)/12.0 if board_ranks else 0.0,
        bf_wet,
    ]
    street_enc = {'pre-flop':0,'flop':1,'auction':2,'turn':3,'river':4}
    se = [0.0]*5; se[street_enc.get(street,0)] = 1.0
    v += se
    v += [
        min(pot/(2*STARTING_STACK),1.0),
        my_chips/STARTING_STACK,
        opp_chips/STARTING_STACK,
        min((opp_wager-my_wager)/max(pot,1),1.0),
        min(my_chips/max(pot,1),10.0)/10.0,
        1.0 if is_bb else 0.0,
        1.0 if auction_won else 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
        min(round_num/1000.0,1.0),
        0.0,
        min(my_wager/STARTING_STACK,1.0),
        min(opp_wager/STARTING_STACK,1.0),
        min(pot/STARTING_STACK,1.0),
    ]
    while len(v) < 33: v.append(0.0)
    return np.array(v[:33], dtype=np.float32)


def read_log(path):
    # Try plain text first (logs are NOT gzipped)
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        pass
    # Fallback: try gzip
    try:
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        pass
    return ''


def parse_log(content):
    """
    Parse actual log format:
    2026-02-28 03:20:32 BotA vs BotB
    Round #1, BotA (0), BotB (0)
    BotA posts blind: 10
    BotA received [8c 2h]
    BotA calls / raises to X / folds / checks / bets X
    Flop [Jd 8d 9c], BotA (60), BotB (60)
    BotA bids 1419
    BotA won the auction and was revealed [9s]
    BotA awarded X
    BotA shows [cards]
    """
    lines = content.strip().split('\n')
    if not lines:
        return [], None, None, None

    # First line: "2026-02-28 03:20:32 BotA vs BotB"
    my_name = opp_name = None
    header_m = re.match(r'.+\s+(\S+)\s+vs\s+(\S+)', lines[0])
    if header_m:
        my_name  = header_m.group(1)   # BotA = us
        opp_name = header_m.group(2)   # BotB = opponent

    if not my_name or not opp_name:
        return [], None, None, None

    hands      = []
    current    = None
    street     = 'pre-flop'

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # New round
        round_m = re.match(r'Round\s+#(\d+),\s+(\S+)\s+\((-?\d+)\),\s+(\S+)\s+\((-?\d+)\)', line)
        if round_m:
            if current:
                hands.append(current)
            rn = int(round_m.group(1))
            # Figure out chip counts from cumulative scores
            # scores are cumulative, actual chips = STARTING_STACK + score
            score_a = int(round_m.group(3))
            score_b = int(round_m.group(5))
            my_chips  = STARTING_STACK + score_a
            opp_chips = STARTING_STACK + score_b
            street = 'pre-flop'
            current = {
                'round':       rn,
                'opp_name':    opp_name,
                'my_name':     my_name,
                'my_hand':     [],
                'opp_hand':    [],
                'board':       [],
                'my_bid':      0,
                'opp_bid':     0,
                'won_auction': False,
                'pot':         BIG_BLIND + BIG_BLIND//2,
                'payoff':      0,
                'my_chips':    my_chips,
                'opp_chips':   opp_chips,
                'my_wager':    0,
                'opp_wager':   0,
                'street':      'pre-flop',
                'is_bb':       False,
                'actions':     [],
                'opp_vpip':    False,
                'opp_pfr':     False,
                'opp_folded_pf': False,
                'opp_bets':    0,
                'opp_checks':  0,
                'opp_actions_count': 0,
            }
            continue

        if current is None:
            continue

        # Blinds — figure out who is BB (BB posts 20)
        blind_m = re.match(r'(\S+)\s+posts blind:\s*(\d+)', line)
        if blind_m:
            poster = blind_m.group(1)
            amount = int(blind_m.group(2))
            if amount == BIG_BLIND:
                current['is_bb'] = (poster == my_name)
            continue

        # Cards dealt
        cards_m = re.match(r'(\S+)\s+received\s+\[([^\]]+)\]', line)
        if cards_m:
            player = cards_m.group(1)
            cards  = cards_m.group(2).split()
            if player == my_name:
                current['my_hand'] = cards
            else:
                current['opp_hand'] = cards
            continue

        # Street + board
        street_m = re.match(r'(Flop|Turn|River)\s+\[([^\]]+)\]', line)
        if street_m:
            street_name = street_m.group(1).lower()
            board_cards = street_m.group(2).split()
            current['board'] = board_cards
            current['street'] = street_name
            street = street_name
            # Update pot from line if available
            pot_m = re.search(r'\)\s*$', line)
            continue

        # Bids
        bid_m = re.match(r'(\S+)\s+bids\s+(\d+)', line)
        if bid_m:
            player = bid_m.group(1)
            amount = int(bid_m.group(2))
            if player == my_name:
                current['my_bid'] = amount
            else:
                current['opp_bid'] = amount
                current['opp_actions_count'] += 1
            continue

        # Auction result
        auction_m = re.match(r'(\S+)\s+won the auction', line)
        if auction_m:
            winner = auction_m.group(1)
            current['won_auction'] = (winner == my_name)
            continue

        # Actions
        # raises to X
        raise_m = re.match(r'(\S+)\s+raises to\s+(\d+)', line)
        if raise_m:
            player = raise_m.group(1)
            amount = int(raise_m.group(2))
            if player == opp_name:
                current['opp_bets'] += 1
                current['opp_vpip'] = True
                current['opp_actions_count'] += 1
                if street == 'pre-flop':
                    current['opp_pfr'] = True
                current['actions'].append({
                    'player': 'opp', 'street': street,
                    'action': 'raise', 'amount': amount,
                })
            continue

        # bets X
        bet_m = re.match(r'(\S+)\s+bets\s+(\d+)', line)
        if bet_m:
            player = bet_m.group(1)
            amount = int(bet_m.group(2))
            if player == opp_name:
                current['opp_bets'] += 1
                current['opp_vpip'] = True
                current['opp_actions_count'] += 1
                current['actions'].append({
                    'player': 'opp', 'street': street,
                    'action': 'bet', 'amount': amount,
                })
            continue

        # calls
        call_m = re.match(r'(\S+)\s+calls', line)
        if call_m:
            player = call_m.group(1)
            if player == opp_name:
                current['opp_vpip'] = True
                current['opp_actions_count'] += 1
                current['actions'].append({
                    'player': 'opp', 'street': street,
                    'action': 'call', 'amount': 0,
                })
            continue

        # checks
        check_m = re.match(r'(\S+)\s+checks', line)
        if check_m:
            player = check_m.group(1)
            if player == opp_name:
                current['opp_checks'] += 1
                current['opp_actions_count'] += 1
                current['actions'].append({
                    'player': 'opp', 'street': street,
                    'action': 'check', 'amount': 0,
                })
            continue

        # folds
        fold_m = re.match(r'(\S+)\s+folds', line)
        if fold_m:
            player = fold_m.group(1)
            if player == opp_name:
                current['opp_actions_count'] += 1
                if street == 'pre-flop':
                    current['opp_folded_pf'] = True
                current['actions'].append({
                    'player': 'opp', 'street': street,
                    'action': 'fold', 'amount': 0,
                })
            continue

        # Showdown
        show_m = re.match(r'(\S+)\s+shows\s+\[([^\]]+)\]', line)
        if show_m:
            player = show_m.group(1)
            cards  = show_m.group(2).split()
            if player == opp_name:
                current['opp_hand'] = cards
            continue

        # Payoff
        award_m = re.match(r'(\S+)\s+awarded\s+(-?\d+)', line)
        if award_m:
            player = award_m.group(1)
            amount = int(award_m.group(2))
            if player == my_name:
                current['payoff'] = amount
            continue

    if current:
        hands.append(current)

    return hands, my_name, opp_name


# ══ MULTI-STRATEGY CLUSTERING ══════════════════════════════════

def extract_match_profile(hands):
    if len(hands) < 10: return None
    vpip    = sum(1 for h in hands if h.get('opp_vpip'))    / len(hands)
    pfr     = sum(1 for h in hands if h.get('opp_pfr'))     / len(hands)
    fold_pf = sum(1 for h in hands if h.get('opp_folded_pf'))/ len(hands)
    bids    = [h['opp_bid'] for h in hands if h.get('opp_bid',0) > 0]
    avg_bid = float(np.mean(bids)) if bids else 0.0
    total_acts = max(sum(h.get('opp_actions_count',1) for h in hands), 1)
    total_bets = sum(h.get('opp_bets',0) for h in hands)
    agg = total_bets / total_acts
    return {
        'vpip': vpip, 'pfr': pfr, 'agg': agg,
        'avg_bid': avg_bid, 'fold_rate': fold_pf,
        'n_hands': len(hands),
        'fingerprint': np.array([vpip, pfr, agg, min(avg_bid/200.,1.), fold_pf]),
        'bid_samples': bids[:50],
    }

def cluster_strategies(match_profiles, threshold=0.15):
    if not match_profiles: return []
    clusters = []
    for profile in match_profiles:
        fp = profile['fingerprint']
        best_i, best_d = None, threshold
        for i, cl in enumerate(clusters):
            d = float(np.linalg.norm(fp - cl['centroid']))
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            clusters[best_i]['profiles'].append(profile)
            all_fps = np.array([p['fingerprint'] for p in clusters[best_i]['profiles']])
            clusters[best_i]['centroid'] = all_fps.mean(axis=0)
        else:
            clusters.append({'profiles': [profile], 'centroid': fp.copy()})
    result = []
    for cl in clusters:
        ps = cl['profiles']
        total = sum(p['n_hands'] for p in ps)
        all_bids = []
        for p in ps: all_bids.extend(p.get('bid_samples',[]))
        result.append({
            'vpip_rate':  float(np.mean([p['vpip']     for p in ps])),
            'pfr_rate':   float(np.mean([p['pfr']      for p in ps])),
            'agg':        float(np.mean([p['agg']      for p in ps])),
            'avg_bid':    float(np.mean([p['avg_bid']  for p in ps])),
            'fold_rate':  float(np.mean([p['fold_rate'] for p in ps])),
            'hands':      total,
            'n_matches':  len(ps),
            'centroid':   cl['centroid'].tolist(),
            'bid_samples': all_bids[:50],
        })
    return result


# ══ SAMPLE EXTRACTION ══════════════════════════════════════════

ACTION_MAP = {'fold':0,'check':1,'call':2,'bet':3,'raise':4}

def extract_samples(all_hands):
    auction_samples = []
    range_samples   = []
    for hand in all_hands:
        if not hand.get('my_hand') or len(hand['my_hand']) < 2:
            continue
        my_hand  = hand['my_hand']
        board    = hand.get('board', [])
        pot      = max(hand.get('pot', 30), 30)
        my_chips = max(hand.get('my_chips', STARTING_STACK), 100)
        opp_chips= max(hand.get('opp_chips', STARTING_STACK), 100)
        round_n  = hand.get('round', 1)
        is_bb    = hand.get('is_bb', False)

        # Auction sample
        my_bid = hand.get('my_bid', 0)
        if my_bid > 0:
            sv = build_state_vector(
                my_hand, board[:3] if board else [], pot,
                my_chips, opp_chips, pot//2, pot//2,
                'auction', is_bb, False, round_n
            )
            auction_samples.append({
                'state':   sv,
                'my_bid':  my_bid,
                'won':     1.0 if hand.get('won_auction') else 0.0,
                'outcome': min(max(hand.get('payoff',0)/200., -1.), 1.),
            })

        # Range samples from opp actions at showdown hands
        opp_hand = hand.get('opp_hand', [])
        if len(opp_hand) >= 2:
            opp_eq = PREFLOP_EQ.get(hand_key(opp_hand), 0.45)
            for act_dict in hand.get('actions', []):
                if act_dict.get('player') != 'opp': continue
                act_street = act_dict.get('street', 'flop')
                action     = act_dict.get('action', 'check')
                amount     = act_dict.get('amount', 0)
                bet_ratio  = min(amount/max(pot,1), 3.0)/3.0
                board_for  = {'pre-flop':[],'flop':board[:3],
                               'turn':board[:4],'river':board[:5]}.get(act_street, board[:3])
                sv = build_state_vector(
                    my_hand, board_for, pot, my_chips, opp_chips,
                    pot//2, pot//2, act_street, is_bb, False, round_n
                )
                range_samples.append({
                    'state':        sv,
                    'action':       ACTION_MAP.get(action, 1),
                    'bet_ratio':    float(bet_ratio),
                    'opp_strength': float(opp_eq),
                })
    return auction_samples, range_samples


# ══ MAIN ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--out',     default='./data/dataset.pkl')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    log_files = []
    for root, dirs, files in os.walk(args.log_dir):
        for f in files:
            if f.endswith('.glog') or f.endswith('.glog.gz') or f.endswith('.log'):
                log_files.append(os.path.join(root, f))

    print(f'Found {len(log_files)} log files')

    opp_match_hands = defaultdict(list)
    all_hands = []
    errors = 0

    for i, log_path in enumerate(sorted(log_files)):
        content = read_log(log_path)
        if not content:
            errors += 1
            continue
        try:
            hands, my_name, opp_name = parse_log(content)
            if not hands:
                errors += 1
                continue
            all_hands.extend(hands)
            if opp_name:
                opp_match_hands[opp_name].append(hands)
            if (i+1) % 10 == 0:
                print(f'  Parsed {i+1}/{len(log_files)}, {len(all_hands)} hands...')
        except Exception as e:
            errors += 1
            if errors <= 3: print(f"  Parse error: {e}")

    print(f'Parsed {len(all_hands)} hands total ({errors} errors)')

    print('Extracting training samples...')
    auction_samples, range_samples = extract_samples(all_hands)
    print(f'  Auction samples: {len(auction_samples):,}')
    print(f'  Range samples:   {len(range_samples):,}')

    print('\nBuilding opponent profiles...')
    opp_profiles = {}
    for opp_name, match_list in opp_match_hands.items():
        match_profiles = [extract_match_profile(m) for m in match_list]
        match_profiles = [mp for mp in match_profiles if mp]
        if not match_profiles: continue
        strategies = cluster_strategies(match_profiles)
        latest = strategies[-1] if strategies else None
        opp_profiles[opp_name] = {
            'strategies':  strategies,
            'latest':      latest,
            'n_matches':   len(match_list),
            'vpip_rate':   latest['vpip_rate']  if latest else 0.55,
            'pfr_rate':    latest['pfr_rate']   if latest else 0.28,
            'avg_bid':     latest['avg_bid']    if latest else 15.0,
            'fold_rate':   latest['fold_rate']  if latest else 0.40,
            'hands':       latest['hands']      if latest else 0,
            'bid_samples': latest['bid_samples'] if latest else [],
        }
        n = len(strategies)
        print(f'  {opp_name}: {len(match_list)} matches, {n} strateg{"y" if n==1 else "ies"}, '
              f'avg_bid={opp_profiles[opp_name]["avg_bid"]:.0f}, '
              f'vpip={opp_profiles[opp_name]["vpip_rate"]:.2f}')

    dataset = {
        'auction_samples': auction_samples,
        'range_samples':   range_samples,
        'opp_profiles':    opp_profiles,
    }
    with open(args.out, 'wb') as f:
        pickle.dump(dataset, f)

    mb = os.path.getsize(args.out)/1024/1024
    print(f'\nSaved to {args.out} ({mb:.1f} MB)')
    print('Next: python train/train_networks.py --data ./data/dataset.pkl --out ./models/')

if __name__ == '__main__':
    main()