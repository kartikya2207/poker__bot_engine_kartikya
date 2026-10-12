# IIT Pokerbots 2026 — Sneak Peek Hold'em Bot

My bot for IIT Pokerbots 2026 (organised by Jane Street across the IITs): a heads-up No-Limit Hold'em bot built around Monte Carlo equity, plus an auction for peeking at the opponent's cards.

## What it is

The competition variant, **Sneak Peek Hold'em**, is heads-up No-Limit Texas Hold'em with one twist: after the flop both players submit a sealed bid in a second-price auction. The winner pays the loser's bid into the pot and gets to see one of the opponent's hole cards. The full rules are in [`IIT_Pokerbots_PS.pdf`](IIT_Pokerbots_PS.pdf).

Bots talk to a local game engine (`engine.py`) through the `pkbot/` framework and implement three hooks: `on_hand_start`, `on_hand_end` and `get_move`. The local config plays matches of 1000 hands, 5000-chip stacks, 10/20 blinds, with a 30-second total time bank per bot.

## What I did

**The submitted bot (`bot_final.py`)** is a fast, equity-driven heuristic bot:
- **Preflop:** a lookup table of win probabilities for all 169 starting-hand classes (`AKs`, `T9o`, ...). Thresholds decide raise / call / fold, with 3-bets or shoves for premium hands facing a raise, and pot-odds checks for marginal calls.
- **Postflop equity:** Monte Carlo rollouts (60 samples) with my own 5-card hand evaluator (`score5` / `best_hand`). If I won the auction and know one opponent card, `exact_equity` fixes that card in the opponent's hand; otherwise `no_info_equity` samples both. Results are cached per street, and if a calculation takes over 0.8 s the bot falls back to preflop equity so it never burns the time bank.
- **Auction:** bids scale with hand strength and a running estimate of the opponent's average bid (inferred from whether I won or lost previous auctions), capped at 2x the pot.
- **Postflop betting:** "raise or fold" when facing a bet. A note in the code records why: in the logged games I looked at, passive postflop calls won only 35% across 695 pots. When not facing a bet it value-bets 50–80% pot with good equity and checks otherwise.

**The ML pipeline I also built (`train/`, `make_*.py`)** — this was the ambitious part:
- `train/parse_logs.py` turns engine `.glog` files into a training dataset (`data/dataset.pkl`).
- `train/train_networks.py` trains two small PyTorch MLPs: an `AuctionNet` (state -> bid) and a `RangeNet` (state + opponent action -> opponent hand strength).
- `train/train_cfr.py` implements external-sampling Monte Carlo CFR with card abstraction (169 preflop buckets, 50 postflop buckets) and a 7-action abstraction.
- `make_cfr_bot.py` / `make_final.py` embed a CFR table or the model weights into a single uploadable bot file (zlib + base85), since the competition wanted one file.
- The trained artifacts are in `models/`. The checked-in `cfr_table.pkl` has 715 information states after 2,000,000 iterations.

**Being honest:** `bot_final.py` as it stands doesn't load any of the models — the heuristic bot is what I ended up submitting. We didn't take home the top prize, but it was a great way to learn game theory and decision-making under uncertainty.

**A later bug-fixing pass:** game logs weren't being written (the write happened after a `return`), `ActionBid` stored `-1` instead of the bid amount in client state, the shove helper fell back to folding instead of calling when already all-in, and the log parser wasn't reading pot sizes properly.

## What I learnt

- **Simple and fast beat clever and unfinished.** I put a lot of effort into CFR and neural nets, but the bot I actually submitted is a ~300-line equity bot with sensible thresholds — easy to debug and with a built-in guard against timeouts. I also never recorded a proper head-to-head comparison between the two approaches. Next time I'd get a solid baseline first, set up a benchmark, and only swap in a learned component once it measurably wins.
- **Your data is only as good as your logging.** Two of the bugs above (logs not saved, pots not parsed) quietly corrupted the dataset that the networks were trained on. I should have sanity-checked the parsed data before training anything.
- **Imperfect information changes everything.** Equity alone isn't a strategy: the auction made me think about the *value of information* — how much a card is worth depends on how it changes my later decisions, not just my hand strength.
- **Opponent modelling needs follow-through.** `OppStats` tracks VPIP and aggression and labels opponents as fish / nit / aggressive / passive, but those labels never feed into decisions. Wiring them in is the obvious next step.
- **Things I'd fix next:** the auction bids on *preflop* equity even though it happens after the flop (flop equity would be better), and in the "raise or fold" branch a strong hand facing an all-in (where raising isn't legal) folds instead of calling.
- 60 Monte Carlo samples is noisy; more samples or a precomputed lookup would make close decisions more stable within the same time budget.

## Repo structure

```
poker__bot_engine_kartikya/
├── bot_final.py         # the submitted bot
├── example_bot.py       # template check/call bot from the organisers
├── engine.py            # local game engine (round-robin between bots in config.py)
├── config.py            # bots to play + match settings
├── pkbot/               # actions, game states, bot base class, socket runner
├── train/               # parse_logs.py, train_networks.py, train_cfr.py
├── make_cfr_bot.py      # embeds models/cfr_table.pkl into a single-file bot
├── make_final.py        # embeds the neural-net models into a single-file bot
├── models/              # auction_net.pt, range_net.pt, cfr_table.pkl, opp_profiles.pkl
├── data/dataset.pkl     # parsed training data
├── BOT_GUIDE.md         # organisers' API guide (GameInfo, PokerState, actions)
└── IIT_Pokerbots_PS.pdf # official problem statement
```

## How to run

```bash
pip install -r requirements.txt
```

The engine plays every pair of bots listed in `config.py`, so you need at least two entries. For example, to play my bot against the template bot:

```python
BOTS = [
    ("Example", "./example_bot.py"),
    ("Final",   "./bot_final.py"),
]
```

```bash
python engine.py      # game logs are written to ./logs
```

The training scripts additionally need `numpy` and `torch`:

```bash
python train/parse_logs.py --log_dir ./logs --out ./data/dataset.pkl
python train/train_networks.py --data ./data/dataset.pkl --out ./models/
python train/train_cfr.py --models ./models/ --checkpoint_dir ./cfr_checkpoints/
python make_cfr_bot.py   # writes a CFR-based bot_final.py (overwrites the current one)
```

See [`BOT_GUIDE.md`](BOT_GUIDE.md) for the bot API.
