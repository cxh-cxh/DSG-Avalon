# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from games.avalon.lm import LmLog, generate
from agent_manager.prompts.avalon_prompt import ACTION_PROMPTS_AND_SCHEMAS
from games.avalon.utils import Deserializable
from games.avalon.config import (
    MAX_DEBATE_TURNS,
    NUM_PLAYERS,
    NUM_GOOD,
    NUM_EVIL,
    TEAM_SIZE,
)

# Role names
MERLIN = "Merlin"
SERVANT = "Servant"
ASSASSIN = "Assassin"
MINION = "Minion"
# SEER = "Seer"
# DOCTOR = "Doctor"


def group_and_format_observations(observations):
    """Groups observations by round and formats them for output.

    Args:
        observations: A list of strings, where each string starts with "Round X:".

    Returns:
        A list of strings, where each string represents the formatted observations
        for a round.
    """

    grouped = {}
    for obs in observations:
        round_num = int(obs.split(":", 1)[0].split()[1])
        obs_text = obs.split(":", 1)[1].strip().replace('"', "")
        grouped.setdefault(round_num, []).append(obs_text)

    formatted_obs = []
    for round_num, round_obs in sorted(grouped.items()):
        formatted_round = f"Round {round_num}:\n"
        formatted_round += "\n".join(f"   - {obs}" for obs in round_obs)
        formatted_obs.append(formatted_round)

    return formatted_obs


# JSON serializer that works for nested classes
class JsonEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, enum.Enum):
            return o.value
        if isinstance(o, set):
            return list(o)
        return o.__dict__


def to_dict(o: Any) -> Union[Dict[str, Any], List[Any], Any]:
    return json.loads(JsonEncoder().encode(o))


class GameView:
    """Represents the state of the game for each player."""

    def __init__(
        self,
        round_number: int,
        current_players: List[str],
        other_good: Optional[str] = None,
        other_evil: Optional[str] = None,
    ):
        self.round_number: int = round_number
        self.current_players: List[str] = current_players
        self.message: List[tuple[str, str]] = []
        self.team_message: List[str] = []
        self.other_good: Optional[str] = other_good
        self.other_evil: Optional[str] = other_evil
        self.current_team = []
        self.current_leader = None
        self.success = False

    def update_message(self, author: str, dialogue: str):
        """Adds a new dialogue entry to the message."""
        self.message.append((author, dialogue))

    def update_team_message(self, info: str):
        """Adds a new team info entry to the message."""
        self.team_message.append(info)

    def clear_message(self):
        """Clears all entries from the message."""
        self.message.clear()
        self.team_message.clear()

    def to_dict(self) -> Any:
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        return cls(**data)


class Player(Deserializable):
    """Represents a player in the game."""

    def __init__(
        self,
        name: str,
        role: str,
        model: Optional[str] = None,
        personality: Optional[str] = "",
    ):
        self.name = name
        self.role = role
        self.personality = personality
        self.model = model
        self.observations: List[str] = []
        self.gamestate: Optional[GameView] = None
        self.is_leader = False

    def initialize_game_view(
        self, round_number, current_players, other_good=None, other_evil=None
    ) -> None:
        self.gamestate = GameView(round_number, current_players, other_good, other_evil)

    def _add_observation(self, observation: str):
        """Adds an observation for the given round."""
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )

        self.observations.append(f"Round {self.gamestate.round_number}: {observation}")

    def add_announcement(self, announcement: str):
        """Adds the current game announcement to the player's observations."""
        self._add_observation(f"Moderator Announcement: {announcement}")

    def _get_game_state(self) -> Dict[str, Any]:
        """Gets the current game state from the player's perspective."""
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )

        formatted_message = [
            (
                f"{author} (You): {dialogue}"
                if author == self.name
                else f"{author}: {dialogue}"
            )
            for author, dialogue in self.gamestate.message
        ]

        formatted_observations = group_and_format_observations(self.observations)

        return {
            "name": self.name,
            "role": self.role,
            "round": self.gamestate.round_number,
            "observations": formatted_observations,
            "team": ", ".join(self.gamestate.current_team),
            "leader": self.gamestate.current_leader,
            "team_history": self.gamestate.team_message,
            "message": formatted_message,
            "personality": self.personality,
            "num_players": NUM_PLAYERS,
            "num_good": NUM_GOOD,
            "num_evil": NUM_EVIL,
            "team_size": TEAM_SIZE[self.gamestate.round_number],
            "success": self.gamestate.success,
        }

    def _generate_action(
        self,
        action: str,
        options: Optional[List[str]] = None,
    ) -> tuple[Any, LmLog]:
        """Helper function to generate player actions."""
        game_state = self._get_game_state()
        if options:
            game_state["options"] = (", ").join(options)
        prompt_template, response_schema = ACTION_PROMPTS_AND_SCHEMAS[action]

        result_key, allowed_values = (
            (action, options)
            if action in ["vote", "approve", "assassinate"]
            else (None, None)
        )

        # Set temperature based on allowed_values
        temperature = 0.5 if allowed_values else 1.0

        return generate(
            prompt_template,
            response_schema,
            game_state,
            model=self.model,
            temperature=temperature,
            allowed_values=allowed_values,
            result_key=result_key,
        )

    def team(self) -> tuple[str, str, LmLog]:
        """
        Choose a team of players.
        return: (team_message, team, log)
        """
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )
        options = [
            player for player in self.gamestate.current_players if player != self.name
        ]
        random.shuffle(options)
        result, log = self._generate_action("team", options)
        team = result.get("team", None).replace(" ", "").split(",")
        team = team[: TEAM_SIZE[self.gamestate.round_number] - 1]
        team.append(self.name)
        team_message = result.get("say", None)
        if team is not None:
            self._add_observation(f"I choose {', '.join(team)} to join my team.")
        return team_message, team, log

    def approve(self) -> tuple[bool, LmLog]:
        """
        Whether to approve to join the team.
        return: (approve_message, approve, log)
        """
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )
        approve, log = self._generate_action("approve", options=["yes", "no"])
        ret = None
        if approve is not None:
            if approve == "yes":
                ret = True
                self._add_observation(f"I agreed to join the team.")
            else:
                ret = False
                self._add_observation(f"I rejected to join the team.")

        return ret, log

    def vote(self) -> tuple[str, LmLog]:
        """Vote for a mission."""
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )
        vote, log = self._generate_action("vote", options=["success", "failure"])
        ret = None
        if vote is not None:
            if vote == "success":
                ret = True
            else:
                ret = False
            self._add_observation(f"I voted {vote} for the mission.")
        return ret, log

    def message(self) -> tuple[str, LmLog]:
        """Discuss about the current team."""
        result, log = self._generate_action("message", [])
        if result is not None:
            say = result.get("say", None)
            return say, log
        return result, log

    def post_mission(self) -> tuple[str, LmLog]:
        """Summarize the round."""
        result, log = self._generate_action("post_mission", [])
        if result is not None:
            say = result.get("say", None)
            return say, log
        return result, log

    def summarize(self) -> tuple[str, LmLog]:
        """Summarize the round."""
        result, log = self._generate_action("summarize", [])
        if result is not None:
            summary = result.get("summary", None)
            if summary is not None:
                summary = summary.strip('"')
                self._add_observation(f"Summary: {summary}")
            return summary, log
        return result, log

    def assassinate(self) -> tuple[str, LmLog]:
        """
        Assassinate a player.
        """
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )
        options = [
            player for player in self.gamestate.other_good if player != self.name
        ]
        random.shuffle(options)
        assassinate, log = self._generate_action("assassinate", options)
        return assassinate, log

    def to_dict(self) -> Any:
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        name = data["name"]
        role = data["role"]
        model = data.get("model", None)
        o = cls(name=name, role=role, model=model)
        o.gamestate = data.get("gamestate", None)
        o.bidding_rationale = data.get("bidding_rationale", "")
        o.observations = data.get("observations", [])
        return o


class Servant(Player):
    """Represents a Servant in the game."""

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        personality: Optional[str] = None,
    ):
        super().__init__(name=name, role=SERVANT, model=model, personality=personality)

    @classmethod
    def from_json(cls, data: dict[Any, Any]):
        name = data["name"]
        model = data.get("model", None)
        o = cls(name=name, model=model)
        o.gamestate = data.get("gamestate", None)
        # o.bidding_rationale = data.get("bidding_rationale", "")
        o.observations = data.get("observations", [])
        return o


class Merlin(Player):
    """Represents a Servant in the game."""

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        personality: Optional[str] = None,
    ):
        super().__init__(name=name, role=MERLIN, model=model, personality=personality)

    def _get_game_state(self, **kwargs) -> Dict[str, Any]:
        """Gets the current game state, including evil-specific context."""
        state = super()._get_game_state(**kwargs)
        state["player_context"] = self._get_context()
        return state

    def _get_context(self):
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )

        context = f"\n- The Minions are {', '.join(self.gamestate.other_evil)}."
        context += f"\n- The Servants are {', '.join(self.gamestate.other_good)}."

        return context

    @classmethod
    def from_json(cls, data: dict[Any, Any]):
        name = data["name"]
        model = data.get("model", None)
        o = cls(name=name, model=model)
        o.gamestate = data.get("gamestate", None)
        # o.bidding_rationale = data.get("bidding_rationale", "")
        o.observations = data.get("observations", [])
        return o


class Minion(Player):
    """Represents a Minion in the game."""

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        personality: Optional[str] = None,
    ):
        super().__init__(name=name, role=MINION, model=model, personality=personality)

    def _get_game_state(self, **kwargs) -> Dict[str, Any]:
        """Gets the current game state, including evil-specific context."""
        state = super()._get_game_state(**kwargs)
        state["player_context"] = self._get_evil_context()
        return state

    # def eliminate(self) -> tuple[str, "LmLog"]:
    #     """Choose a player to eliminate."""
    #     if not self.gamestate:
    #         raise ValueError(
    #             "GameView not initialized. Call initialize_game_view() first."
    #         )

    #     options = [
    #         player
    #         for player in self.gamestate.current_players
    #         if player != self.name and player != self.gamestate.other_wolf
    #     ]
    #     random.shuffle(options)
    #     eliminate, log = self._generate_action("remove", options)
    #     return eliminate, log

    def _get_evil_context(self):
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )

        context = f"\n- The other Minions are {', '.join(self.gamestate.other_evil)}."

        return context

    @classmethod
    def from_json(cls, data: dict[Any, Any]):
        name = data["name"]
        model = data.get("model", None)
        o = cls(name=name, model=model)
        o.gamestate = data.get("gamestate", None)
        # o.bidding_rationale = data.get("bidding_rationale", "")
        o.observations = data.get("observations", [])
        return o


class Assassin(Player):
    """Represents an Assassin in the game."""

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        personality: Optional[str] = None,
    ):
        super().__init__(name=name, role=ASSASSIN, model=model, personality=personality)

    def _get_game_state(self, **kwargs) -> Dict[str, Any]:
        """Gets the current game state, including evil-specific context."""
        state = super()._get_game_state(**kwargs)
        state["player_context"] = self._get_evil_context()
        return state

    def assassinate(self) -> tuple[str, "LmLog"]:
        """Assassinate a player (Merlin)."""
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )

        options = [player for player in self.gamestate.other_good]
        random.shuffle(options)
        assassinate, log = self._generate_action("assassinate", options)
        return assassinate, log

    def _get_evil_context(self):
        if not self.gamestate:
            raise ValueError(
                "GameView not initialized. Call initialize_game_view() first."
            )

        context = f"\n- The other Minions are {', '.join(self.gamestate.other_evil)}."

        return context

    @classmethod
    def from_json(cls, data: dict[Any, Any]):
        name = data["name"]
        model = data.get("model", None)
        o = cls(name=name, model=model)
        o.gamestate = data.get("gamestate", None)
        # o.bidding_rationale = data.get("bidding_rationale", "")
        o.observations = data.get("observations", [])
        return o


class Round(Deserializable):
    """Represents a round of gameplay in Avalon.

    Attributes:
      players: List of player names in this round.
      exiled: Who the players decided to exile after the debate.
      team: List of players in the current team.
      approve: Dictionary of players in the current team and whether they agreed to join the team.
      votes: Dictionary of players in the current team and whether they voted success or failure for the current mission.
      message: List of message tuples of player name and what they said during the
        message.
      success (bool): Indicates whether the round was completed successfully.

    Methods:
      to_dict: Returns a dictionary representation of the round.
    """

    def __init__(self):
        self.players: List[str] = []
        self.team: List[List[str]] = []
        self.approve: List[Dict[str, bool]] = []
        self.votes: Dict[str, bool] = {}
        self.message: List[Tuple[str, str]] = []
        self.team_message: List[Tuple[str, str]] = []
        self.approve_message: List[Tuple[str, str]] = []
        self.success: bool = False

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        o = cls()
        o.players = data["players"]
        o.team = data.get("team", [])
        o.approve = data.get("approve", {})
        o.votes = data.get("votes", {})
        o.message = data.get("message", [])
        o.team_message = data.get("team_message", [])
        o.approve_message = data.get("approve_message", [])
        o.success = data.get("success", False)
        return o


class State(Deserializable):
    """Represents a game session.

    Attributes:
      session_id: Unique identifier for the game session.
      players: Dict of players in the game.
      seer: The player with the seer role.
      doctor: The player with the doctor role.
      villagers: List of players with the villager role.
      werewolves: List of players with the werewolf role.
      rounds: List of Rounds in the game.
      error_message: Contains an error message if the game failed during
        execution.
      winner: Villager or Werewolf

    Methods:
      to_dict: Returns a dictionary representation of the game.
    """

    def __init__(
        self,
        session_id: str,
        merlin: Merlin,
        assassin: Assassin,
        servants: List[Servant],
        minions: List[Minion],
    ):
        self.session_id: str = session_id
        self.merlin: Merlin = merlin
        self.assassin: Assassin = assassin
        self.servants: List[Servant] = servants
        self.minions: List[Minion] = minions
        self.leader = 0
        self.players: Dict[str, Player] = {
            player.name: player
            for player in self.servants + self.minions + [self.merlin, self.assassin]
        }
        self.player_names = [
            player.name
            for player in self.servants + self.minions + [self.merlin, self.assassin]
        ]
        random.shuffle(self.player_names)
        # for i in range(len(self.player_names)):
        #     if self.player_names[i] == merlin.name:
        #         self.leader = i
        self.rounds: List[Round] = []
        self.error_message: str = ""
        self.winner: str = ""
        self.success_cnt = 0
        self.failure_cnt = 0

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        servants = []
        for s in data.get("servants", []):
            servants.append(servants.from_json(s))

        minions = []
        for m in data.get("minions", []):
            minions.append(minions.from_json(m))

        merlin = Merlin.from_json(data.get("merlin"))
        assassin = Assassin.from_json(data.get("assassin"))

        players = {}
        for p in servants + minions + [merlin, assassin]:
            players[p.name] = p

        o = cls(
            data.get("session_id", ""),
            merlin,
            assassin,
            servants,
            minions,
        )
        rounds = []
        for r in data.get("rounds", []):
            rounds.append(Round.from_json(r))

        o.rounds = rounds
        o.error_message = data.get("error_message", "")
        o.winner = data.get("winner", "")
        return o


class TeamLog(Deserializable):

    def __init__(self, player: str, message: str, teammates: List[str], log: LmLog):
        self.player = player
        self.message = message
        self.teammates = teammates
        self.log = log

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        player = data.get("player", None)
        message = data.get("message", None)
        teammates = data.get("teammates", None)
        log = LmLog.from_json(data.get("log", None))
        return cls(player, message, teammates, log)


class ApproveLog(Deserializable):

    def __init__(self, player: str, approve: bool, log: LmLog):
        self.player = player
        self.approve = approve
        self.log = log

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        player = data.get("player", None)
        approve = data.get("approve", None)
        log = LmLog.from_json(data.get("log", None))
        return cls(player, approve, log)


class VoteLog(Deserializable):

    def __init__(self, player: str, vote: bool, log: LmLog):
        self.player = player
        self.vote = vote
        self.log = log

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        player = data.get("player", None)
        vote = data.get("vote", None)
        log = LmLog.from_json(data.get("log", None))
        return cls(player, vote, log)


class RoundLog(Deserializable):
    """Represents the logs of a round of gameplay in Avalon.

    Attributes:
      teams: List of all teams in the round.
      approve: List of all agreements in the round.
      messages: Logs of the messages. Each round has multiple message turns, so it's a
        list. Each element is a tuple - the 1st element is the name of the player
        who spoke at this turn, and the 2nd element is the message.
      summaries: Logs from the summarize step. Every player summarizes their
        observations at the end of a round before they vote. Each element is a
        tuple where the 1st element is the name of the player, and the 2nd element
        is the log
    """

    def __init__(self):
        self.votes: List[VoteLog] = []
        self.teams: List[TeamLog] = []
        self.approve: List[List[ApproveLog]] = []
        self.messages: List[Tuple[str, LmLog]] = []
        self.summaries: List[Tuple[str, LmLog]] = []

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        o = cls()

        for votes in data.get("votes", []):
            v_logs = []
            o.votes.append(v_logs)
            for v in votes:
                v_logs.append(VoteLog.from_json(v))

        # for r in data.get("bid", []):
        #     r_logs = []
        #     o.bid.append(r_logs)
        #     for player in r:
        #         r_logs.append((player[0], LmLog.from_json(player[1])))

        for player in data.get("debate", []):
            o.debate.append((player[0], LmLog.from_json(player[1])))

        for player in data.get("summaries", []):
            o.summaries.append((player[0], LmLog.from_json(player[1])))

        return o
