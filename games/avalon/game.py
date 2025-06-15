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

"""Avalon game."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import random
from typing import List
import copy
import tqdm

from games.avalon.model import Round, RoundLog, State, VoteLog, TeamLog, ApproveLog
from games.avalon.config import NUM_PLAYERS, TEAM_SIZE, FAILURE_VOTE


# def get_max_bids(d):
#     """Gets all the keys with the highest value in the dictionary."""
#     max_value = max(d.values())
#     max_keys = [key for key, value in d.items() if value == max_value]
#     return max_keys


class GameMaster:

    def __init__(
        self,
        state: State,
        num_threads: int = 1,
    ) -> None:
        """Initialize the Avalon game.

        Args:
        """
        self.state = state
        self.current_round_num = len(self.state.rounds) if self.state.rounds else 0
        self.num_threads = num_threads
        self.logs: List[RoundLog] = []

    @property
    def this_round(self) -> Round:
        return self.state.rounds[self.current_round_num]

    @property
    def this_round_log(self) -> RoundLog:
        return self.logs[self.current_round_num]

    def choose_team(self):
        """Run the day phase which consists of the debate and voting."""

        for team_idx in range(5):  # at most 5 chances to team up
            leader = self.state.player_names[self.state.leader]
            leader = self.state.players[leader]
            leader.is_leader = True

            team_message, team, log = leader.team()
            self.this_round_log.teams.append(
                TeamLog(leader.name, team_message, team, log)
            )
            self.this_round.team_message.append([leader.name, team_message])
            self.this_round.team.append(team)

            for name in self.this_round.players:
                player = self.state.players[name]
                if player.gamestate:
                    player.gamestate.update_message(leader.name, team_message)
                    player.gamestate.update_team_message(
                        f"Team {team_idx+1} is {', '.join(team)}.\n"
                    )
                    player.gamestate.current_team = copy.copy(team)
                    player.gamestate.current_leader = leader.name
                else:
                    raise ValueError(f"{name}.gamestate needs to be initialized.")

            tqdm.tqdm.write(f"{leader.name} ({leader.role}): {team_message}")

            for idx in range(NUM_PLAYERS - 1):
                next_speaker = self.state.player_names[
                    (self.state.leader + idx + 1) % NUM_PLAYERS
                ]

                if not next_speaker:
                    raise ValueError("get_next_speaker did not return a valid player.")

                player = self.state.players[next_speaker]
                dialogue, log = player.message()
                if dialogue is None:
                    raise ValueError(
                        f"{next_speaker} did not return a valid dialouge from debate()."
                    )

                self.this_round_log.messages.append((next_speaker, log))
                self.this_round.message.append([next_speaker, dialogue])
                tqdm.tqdm.write(f"{next_speaker} ({player.role}): {dialogue}")

                for name in self.this_round.players:
                    player = self.state.players[name]
                    if player.gamestate:
                        player.gamestate.update_message(next_speaker, dialogue)
                    else:
                        raise ValueError(f"{name}.gamestate needs to be initialized.")

            if team_idx < 4:
                approves, approve_logs = self.run_approve()
                self.this_round.approve.append(approves)
                self.this_round_log.approve.append(approve_logs)

                agree = 1  # leader automatically agree
                for player, approve in self.this_round.approve[-1].items():
                    for name in self.this_round.players:
                        p = self.state.players[name]
                        if p.gamestate:
                            if approve:
                                p.gamestate.update_team_message(
                                    f"{player} agreed to join team {team_idx+1}.\n"
                                )
                            else:
                                p.gamestate.update_team_message(
                                    f"{player} rejected to join team {team_idx+1}.\n"
                                )
                        else:
                            raise ValueError(
                                f"{name}.gamestate needs to be initialized."
                            )
                    if approve:
                        agree += 1
                        tqdm.tqdm.write(f"{player} agreed to join the team {team_idx}")
                    else:
                        tqdm.tqdm.write(
                            f"{player} rejected to join the team {team_idx}"
                        )

                if agree > TEAM_SIZE[self.current_round_num] // 2:
                    announcement = (
                        f"Team {team_idx + 1} was approved and went on the mission."
                    )
                    tqdm.tqdm.write(f"The team is ready")
                    break
                else:
                    announcement = f"Team {team_idx + 1} was rejected."
                    leader.is_leader = False
                    self.state.leader = (self.state.leader + 1) % NUM_PLAYERS
            else:
                announcement = f"Team {team_idx + 1} went on the mission."
                tqdm.tqdm.write(f"The team is ready")
                break

            for name in self.this_round.players:
                player = self.state.players[name]
                if player.gamestate:
                    player.gamestate.update_team_message(announcement)
                else:
                    raise ValueError(f"{name}.gamestate needs to be initialized.")

        for name in self.this_round.players:
            player = self.state.players[name]
            if player.gamestate:
                player.gamestate.update_team_message(announcement)
            else:
                raise ValueError(f"{name}.gamestate needs to be initialized.")

    def post_mission_discussion(self):
        for idx in range(NUM_PLAYERS):
            next_speaker = self.state.player_names[
                (self.state.leader + idx) % NUM_PLAYERS
            ]

            if not next_speaker:
                raise ValueError("get_next_speaker did not return a valid player.")

            player = self.state.players[next_speaker]
            dialogue, log = player.post_mission()
            if dialogue is None:
                raise ValueError(
                    f"{next_speaker} did not return a valid dialouge from."
                )

            self.this_round_log.messages.append((next_speaker, log))
            self.this_round.message.append([next_speaker, dialogue])
            tqdm.tqdm.write(f"{next_speaker} ({player.role}): {dialogue}")

            for name in self.this_round.players:
                player = self.state.players[name]
                if player.gamestate:
                    player.gamestate.update_message(next_speaker, dialogue)
                else:
                    raise ValueError(f"{name}.gamestate needs to be initialized.")

    def summary(self):
        for idx in range(NUM_PLAYERS):
            next_speaker = self.state.player_names[
                (self.state.leader + idx) % NUM_PLAYERS
            ]

            if not next_speaker:
                raise ValueError("get_next_speaker did not return a valid player.")

            player = self.state.players[next_speaker]
            dialogue, log = player.post_mission()
            if dialogue is None:
                raise ValueError(f"{next_speaker} did not return a valid summary from.")

            self.this_round_log.messages.append((next_speaker, log))
            self.this_round.message.append([next_speaker, dialogue])
            tqdm.tqdm.write(f"{next_speaker} ({player.role}) summary: {dialogue}")

    def run_mission(self):
        votes, vote_log = self.run_voting()
        self.this_round.votes = votes
        self.this_round_log.votes = vote_log

        failure = 0
        for player, vote in self.this_round.votes.items():
            if not vote:
                failure += 1
                tqdm.tqdm.write(f"{player} voted failure for the mission")
            else:
                tqdm.tqdm.write(f"{player} voted success for the mission")

        success = None
        if failure >= FAILURE_VOTE[self.current_round_num]:
            self.state.failure_cnt += 1
            success = False
            announcement = f"Round {self.current_round_num + 1}: The team of {', '.join(self.this_round.team[-1])} lead by {self.state.player_names[self.state.leader]} failed the mission."
            tqdm.tqdm.write(f"The mission failed")
        else:
            self.state.success_cnt += 1
            success = True
            announcement = f"Round {self.current_round_num + 1}: The team of {', '.join(self.this_round.team[-1])} lead by {self.state.player_names[self.state.leader]} succeeded the mission."
            tqdm.tqdm.write(f"The mission succeeded")
        for name in self.this_round.players:
            player = self.state.players[name]
            player.gamestate.success = success
            player.add_history(announcement)

        self.post_mission_discussion()
        self.summary()

        leader = self.state.player_names[self.state.leader]
        leader = self.state.players[leader]
        leader.is_leader = False
        self.state.leader = (self.state.leader + 1) % NUM_PLAYERS

    def run_approve(self):
        """Conduct a vote among players for the team."""
        approves = {}
        approve_logs = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            player_votes = {
                name: executor.submit(self.state.players[name].approve)
                for name in self.this_round.team[-1][:-1]
            }

            for player_name, approve_task in player_votes.items():
                approve, log = approve_task.result()
                approve_logs.append(ApproveLog(player_name, approve, log))

                if approve is not None:
                    approves[player_name] = approve
                else:
                    self.this_round.approve.append(approves)
                    self.this_round_log.approve.append(approve_logs)
                    raise ValueError(
                        f"{player_name} vote did not return a valid player."
                    )

        return approves, approve_logs

    def run_voting(self):
        """Conduct a vote among players for the mission."""
        vote_log = []
        votes = {}

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            player_votes = {
                name: executor.submit(self.state.players[name].vote)
                for name in self.this_round.team[-1]
            }

            for player_name, vote_task in player_votes.items():
                vote, log = vote_task.result()
                vote_log.append(VoteLog(player_name, vote, log))

                if vote is not None:
                    votes[player_name] = vote
                else:
                    self.this_round.votes = votes
                    self.this_round_log.votes = vote_log
                    raise ValueError(
                        f"{player_name} vote did not return a valid player."
                    )

        return votes, vote_log

    def assassinate(self):
        """The assassin trys to assassinate Merlin."""
        target, log = self.state.assassin.assassinate()

        if target == self.state.merlin.name:
            announcement = "The assassin assassinated Merlin."
            self.state.winner = "Evil"
        else:
            announcement = "The assassin failed to assassinate Merlin."
            self.state.winner = "Good"

        for name in self.this_round.players:
            player = self.state.players[name]
            player.add_announcement(announcement)

        tqdm.tqdm.write(announcement)

    def run_round(self):
        """Run a single round of the game."""
        self.state.rounds.append(Round())
        self.logs.append(RoundLog())

        self.this_round.players = list(self.state.players.keys())

        for action, message in [
            (
                self.choose_team,
                "The team leader is choosing the teamates",
            ),
            (self.run_mission, "The selected team is voting for a mission"),
            (self.check_for_winner, "Checking if good side wins"),
        ]:
            tqdm.tqdm.write(message)
            action()

            if self.state.winner:
                tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
                self.this_round.success = True
                return

        tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
        self.this_round.success = True

    def get_winner(self) -> str:
        """Determine the winner of the game."""
        if self.state.success_cnt >= 3:
            return "Good"
        elif self.state.failure_cnt >= 3:
            return "Evil"
        else:
            return ""

    def check_for_winner(self):
        """Check if there is a winner and update the state accordingly."""
        self.state.winner = self.get_winner()
        if self.state.winner == "Good":
            self.assassinate()
        if self.state.winner:
            tqdm.tqdm.write(f"The winner is {self.state.winner}!")

    def run_game(self) -> str:
        """Run the entire Avalon game and return the winner."""
        while not self.state.winner:
            tqdm.tqdm.write(f"STARTING ROUND: {self.current_round_num}")
            self.run_round()
            for name in self.this_round.players:
                if self.state.players[name].gamestate:
                    self.state.players[name].gamestate.round_number = (
                        self.current_round_num + 1
                    )
                    self.state.players[name].gamestate.clear_message()
            self.current_round_num += 1

        tqdm.tqdm.write("Game is complete!")
        return self.state.winner
