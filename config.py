PYTHON_CMD = "python"

# Define ALL bots here
BOTS = [
    ("BotA", "./bot_2.py"),
    ("BotB", "./bot_final.py"),
    ("BotC", "./bot_1.py"),
    ("BotD", "./bot_moghe2.py"),
]

# Tournament settings
ROUND_ROBIN_REPEATS = 1 # play full RR 10 times

# Game settings
NUM_ROUNDS = 1000
STARTING_STACK = 5000
BIG_BLIND = 20
SMALL_BLIND = 10

GAME_LOG_FOLDER = "./logs"