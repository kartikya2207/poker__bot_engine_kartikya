# IIT Pokerbots 2026 - Jane Street Competition

This repository contains my submission for the **IIT Pokerbots 2026** competition, organized by **Jane Street** across all IITs. 

## 🃏 Competition Overview: Sneak Peek Hold'em

The competition featured a unique poker variant called **"Sneak Peek Hold'em"**, based on standard No-Limit Texas Hold'em with a key modification:
- **The Auction:** After the flop, players participate in a sealed-bid second-price auction to view one of their opponent's hole cards.
- **Information Advantage:** The higher bidder pays the lower bid into the pot and gets to see one random hole card of the opponent.

## 🤖 My Bot Strategy: `bot_final.py`

My final bot implementation utilizes a combination of mathematical equity calculations and heuristic-based decision making:

- **Pre-flop:** Uses a lookup table (`PF`) of winning probabilities for all 169 possible starting hands to decide whether to fold, call, or raise.
- **Equity Engine:** Implements a Monte Carlo simulation (`exact_equity` and `no_info_equity`) to estimate the hand's strength against a range of possible opponent hands, accounting for the information gained during the auction.
- **Auction Logic:** A strategic bidding system that values information based on the current hand strength and pot size.
- **Hand Evaluation:** Optimized `score5` and `best_hand` functions for fast ranking of poker hands using bitmask-like scoring.

## 📁 Project Structure

- `bot_final.py`: The main competition bot.
- `engine.py`: The local game engine for testing.
- `IIT_Pokerbots_PS.pdf`: The official problem statement by Jane Street.
- `pkbot/`: Core library containing actions, states, and runner logic.
- `models/` & `train/`: Infrastructure for CFR (Counterfactual Regret Minimization) training.

## 🎓 Reflection

Participating in this competition across all IITs was an incredible learning experience in game theory, probability, and real-time decision-making algorithms under uncertainty. Although we didn't take home the top prize, building a bot capable of competing at this level was a rewarding technical challenge.

---
*Developed as part of the IIT Pokerbots 2026 Competition.*
