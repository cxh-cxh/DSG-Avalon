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
NUM_GOOD = 3
NUM_EVIL = 2
_THREADS = 1

TEAM_SIZE = [2, 3, 3, 4, 4]
FAILURE_VOTE = [1, 1, 1, 2, 1]

HAS_MORDRED = False
HAS_PERCIVAL_AND_MORGANA = False


def get_player_names():
    return random.sample(NAMES, NUM_PLAYERS)
