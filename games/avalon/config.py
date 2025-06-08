import random

RETRIES = 3
NAMES = [
    "Derek",
    "Scott",
    "Jacob",
    "Isaac",
    "Hayley",
    "David",
    "Tyler",
    "Ginger",
    "Jackson",
    "Mason",
    "Dan",
    "Bert",
    "Will",
    "Sam",
    "Paul",
    "Leah",
    "Harold",
]
MAX_DEBATE_TURNS = 8

NUM_PLAYERS = 5
_THREADS = 1

TEAM_SIZE = [2, 2, 3, 4, 5]
FAILURE_VOTE = [1, 1, 1, 2, 1]


def get_player_names():
    return random.sample(NAMES, NUM_PLAYERS)
