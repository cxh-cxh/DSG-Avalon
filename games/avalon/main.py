import sys

sys.path.append("C:/Users/cxh17/Desktop/DSGBench")

from games.avalon.game import GameMaster
from games.avalon.model import (
    Merlin,
    Assassin,
    Servant,
    Minion,
    Mordred,
    Percival,
    Morgana,
    State,
)

merlin = Merlin("Alice", model="deepseek-chat")
assassin = Assassin("Bob", model="deepseek-chat")
servant1 = Servant("Carol", model="deepseek-chat")
servant2 = Servant("Dan", model="deepseek-chat")
minion = Minion("Emily", model="deepseek-chat")
mordred = Mordred("F", model="deepseek-chat")
percival = Percival("G", model="deepseek-chat")
morgana = Morgana("H", model="deepseek-chat")


players = [
    merlin.name,
    assassin.name,
    servant1.name,
    servant2.name,
    minion.name,
    mordred.name,
    percival.name,
    morgana.name,
]

merlin.initialize_game_view(
    current_players=players,
    round_number=0,
    other_good=[servant1.name, servant2.name, percival.name, mordred.name],
    other_evil=[minion.name, assassin.name, morgana.name],
)

assassin.initialize_game_view(
    current_players=players,
    round_number=0,
    other_good=[merlin.name, servant1.name, servant2.name, percival.name],
    other_evil=[minion.name, morgana.name, mordred.name],
)

servant1.initialize_game_view(
    current_players=players,
    round_number=0,
    other_good=[merlin.name, servant2.name, percival.name],
    other_evil=[],
)

servant2.initialize_game_view(
    current_players=players,
    round_number=0,
    other_good=[merlin.name, servant1.name, percival.name],
    other_evil=[minion.name, assassin.name],
)

minion.initialize_game_view(
    current_players=players,
    round_number=0,
    other_good=[merlin.name, servant1.name, servant2.name],
    other_evil=[assassin.name],
)

state = State("111", merlin, assassin, [servant1, servant2], [minion])
gm = GameMaster(state=state)
gm.run_game()
