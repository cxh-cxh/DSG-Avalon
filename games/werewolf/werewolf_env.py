import json
import random
from typing import List, Tuple
import enum
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import wandb
from tqdm import tqdm
import gym
from games.werewolf.runner import initialize_players
from games.werewolf.model import State
from games.werewolf.game import GameMaster
from games.werewolf.config import _THREADS
from games.werewolf.model import Round, RoundLog, State, VoteLog

class WereWolfEnv(gym.Env):
    """ WereWolfEnv

        The goal of hotter colder is to *****

        After each step the agent receives an observation of:
        *****

        The rewards is calculated as:
        *****
        """
    def __init__(self, args):
        # 初始化
        self.args = args
        self.logger = self.args.logger
        self.game=None
        self.state=None
        self.round_debate_players = []
        self.round_vote_players = []
        self.round_votes = {}
        self.round_votes_log = []
        self.round_summ_players = []
        self.action_seq=['new_round','eliminate','protect','unmask','resolve_night_phase','check_for_winner','run_day_phase','vote','exile','check_for_winner','run_summaries']
        self.action_seq_pointer=-1
        self.wolf_alive=2
        self.vote_wolf=0
        self.vote_exile_wolf=0
        self.valid_roundvote_times=0
        self.good_vote_times=0
        self.valid_vote_times=0
        self.all_vote_times=0
        self.key_role_alive=['seer','doctor']
        self.cur_alive_wolf_name = []
        self.cur_alive_good_name=[]

        self.logger.info("=" * 5 + f"WereWolfEnv Init Successfully!: " + "=" * 5)

    def _init_players(self):
        seer, doctor, villagers, werewolves = initialize_players()
        session_id = "10"  # You might want to make this unique per game
        self.state = State(
            villagers=villagers,
            werewolves=werewolves,
            seer=seer,
            doctor=doctor,
            session_id=session_id,
        )

    def reset(self):
        if self.game is not None:
            self.state=None
            self.game=None
            self.action_seq_pointer = -1
            self.round_debate_players = []
            self.round_vote_players = []
            self.round_votes = {}
            self.round_votes_log = []
            self.round_summ_players = []
        self._init_players()
        self.game = GameMaster(self.state, num_threads=_THREADS)
        self.logger.info("=" * 5 + f"WereWolfEnv Reset successfully!: " + "=" * 5)


    def reset_new_round(self):
        self.game.state.rounds.append(Round())
        self.game.logs.append(RoundLog())
        self.game.this_round.players = (
            list(self.game.state.players.keys())
            if self.game.current_round_num == 0
            else self.game.state.rounds[self.game.current_round_num - 1].players.copy()
        )
        self.logger.info(f"STARTING ROUND: {self.game.current_round_num}")
        self.action_seq_pointer += 1
        cur_action = self.action_seq[self.action_seq_pointer]
        # Execute Night Phase
        if cur_action == "eliminate":
            message = "The Werewolves are picking someone to remove from the game."
            self.logger.info(message)
            return self.game.eliminate_pre()

    def _wandb_log(self):
        # IRP(Identity Inference Accuracy) (correct_identifications / total_identification_attempts) * 100%
        irp=self.vote_wolf/self.good_vote_times if self.good_vote_times!=0 else 0

        # IUR(Information utilization) (effective_use_of_information / total_available_information) * 100%
        wandb.log({"irp":irp},step=self.game.current_round_num)

    def to_dict(self,o: Any) -> Union[Dict[str, Any], List[Any], Any]:
        return json.loads(JsonEncoder().encode(o))
    def save_game(self,state: State, logs: List[RoundLog], directory: str):
        """Save the current game state to a specified file.

        This function serializes the game state to JSON and writes it to the
        specified file. If an error message is provided, it adds the error
        message to the current round of the game state before saving.

        Args:
          state: Instance of the `State` class.
          logs: Logs of the  game.
          directory: where to save the game.
        """
        os.makedirs(directory, exist_ok=True)

        partial_game_state_file = f"{directory}/game_partial.json"
        if state.error_message:
            game_file = partial_game_state_file
        else:
            game_file = f"{directory}/game_complete.json"
            # Remove the partial game file if it exists
            if os.path.exists(partial_game_state_file):
                os.remove(partial_game_state_file)

        log_file = f"{directory}/game_logs.json"

        with open(game_file, "w") as file:
            json.dump(state.to_dict(), file, indent=4)

        with open(log_file, "w") as file:
            json.dump(self.to_dict(logs), file, indent=4)
    def _wandb_ksr(self,key_role_alive):
        # KSR(Key role survival rate) (key_role_survived / total_key_role_games) * 100%
        ksr=1
        if isinstance(key_role_alive,int):
            ksr=key_role_alive/2
        # VSS(Key voting success rate) (successful_votes / total_critical_votes) * 100%
        vss=self.vote_exile_wolf/self.valid_roundvote_times if self.valid_roundvote_times!=0 else 0
        wandb.log({"ksr":ksr,"vss":vss},step=self.game.current_round_num)



    def step(self,action,pre_observations=None):
        self.action_seq_pointer += 1
        cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "new_round":
            observations=self.reset_new_round()
            if observations is not None:
                return observations,None

        # 1. input None and return the observed value
        # 2. input action and return the next observed value
        if cur_action=="protect":
            eliminated,log=action
            self.game.eliminate_post(pre_observations,eliminated,log)
            if self.game.state.doctor.name not in self.game.this_round.players:
                # 执行下一个unmask
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                options=list(self.game.state.doctor.gamestate.current_players)
                random.shuffle(options)
                observations = {
                    "player_name": self.game.state.doctor.name,
                    "game_state": self.game.state.doctor._get_game_state(),
                    "action": "protect",
                    "options": options
                }
                return observations,None
        if cur_action=="unmask":
            protect, log = action
            self.game.protect_post(protect, log)
            if self.game.state.seer.name not in self.game.this_round.players:
                # Execute the next resolve_night_phase
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                options = [
                    player
                    for player in self.game.state.seer.gamestate.current_players
                    if player != self.game.state.seer.name and player not in self.game.state.seer.previously_unmasked.keys()
                ]
                random.shuffle(options)
                observations = {
                    "player_name": self.game.state.seer.name,
                    "game_state": self.game.state.seer._get_game_state(),
                    "action": "investigate",
                    "options": options
                }
                return observations,None
        if cur_action=="resolve_night_phase":
            unmask, log = action
            self.game.this_round_log.investigate = log

            if unmask is not None:
                self.game.this_round.unmasked = unmask
                self.game.state.seer.reveal_and_update(unmask, self.game.state.players[unmask].role)
            else:
                raise ValueError("Unmask function did not return a valid player.")
            self.game.resolve_night_phase()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="check_for_winner":
            self.game.check_for_winner()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="run_day_phase":
            # 1. Process the previous debate results first
            if pre_observations is not None and pre_observations['action']=="debate":
                result, log=action
                pre_speaker=pre_observations['player_name']
                if result is not None:
                    dialogue = result.get("say", None)
                    self.game.this_round_log.debate.append((pre_speaker, log))
                    self.game.this_round.debate.append([pre_speaker, dialogue])
                    tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    # update all player gamestate
                    for name in self.game.this_round.players:
                        player = self.game.state.players[name]
                        if player.gamestate:
                            player.gamestate.update_debate(pre_speaker, dialogue)
                        else:
                            raise ValueError(f"{name}.gamestate needs to be initialized.")
                else:
                    raise ValueError(
                        f"{pre_speaker} did not return a valid dialouge from debate()."
                    )

            # 2. Select the player without debate
            next_speaker=None
            for speaker in self.game.this_round.players:
                # speaker = self.game.state.players[name]
                if speaker not in self.round_debate_players:
                    next_speaker=speaker
                    self.round_debate_players.append(speaker)
                    break
            if next_speaker is not None:
                player = self.state.players[next_speaker]
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "debate",
                    "options": []
                }
                self.action_seq_pointer -= 1
                return observations,None
            else:
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
                self.round_debate_players=[]

        if cur_action=="vote":
            # 1. Process the vote return value first
            if pre_observations is not None and pre_observations['action'] == "vote":
                vote, log=action
                pre_player_name=pre_observations['player_name']
                pre_player=self.game.state.players[pre_player_name]
                if vote is not None:
                    pre_player._add_observation(
                        f"After the debate, I voted to remove {vote} from the game."
                    )
                self.round_votes_log.append(VoteLog(pre_player_name, vote, log))

                if vote is not None:
                    self.round_votes[pre_player_name]=vote
                else:
                    self.game.this_round.votes.append(self.round_votes)
                    self.game.this_round_log.votes.append(self.round_votes_log)
                    self.round_votes = {}
                    self.round_votes_log = []
                    raise ValueError(f"{pre_player_name} vote did not return a valid player.")
            
            # If every player has spoken, then vote
            # Loop through the votes
            next_voter=None
            for voter in self.game.this_round.players:
                # voter = self.game.state.players[name]
                if voter not in self.round_vote_players:
                    next_voter = voter
                    self.round_vote_players.append(voter)
                    break
            if next_voter is not None:
                player = self.state.players[next_voter]
                options=[ player_name for player_name in player.gamestate.current_players
                        if player_name != player.name       ]
                random.shuffle(options)
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "vote",
                    "options": options
                }
                self.action_seq_pointer -= 1
                return observations,None
            else:

                self.round_vote_players=[]
                self.game.this_round.votes.append(self.round_votes)
                self.game.this_round_log.votes.append(self.round_votes_log)
                # ====record vote for wandb====
                cur_alive_wolf=[w for w in self.game.state.werewolves if w.name in self.game.this_round.players]
                self.cur_alive_wolf_name=[w.name for w in cur_alive_wolf]
                self.cur_alive_good_name=[w for w in self.game.this_round.players if w not in self.cur_alive_wolf_name]
                for k,v in self.round_votes.items():
                    if k in self.cur_alive_good_name:
                        self.good_vote_times+=1
                        if v in self.cur_alive_wolf_name:
                            self.vote_wolf+=1
                self.all_vote_times+=1
                self._wandb_log()
                # ====record vote for wandb====
                for player, vote in self.game.this_round.votes[-1].items():
                    tqdm.write(f"{player} voted to remove {vote}")
                self.round_votes_log=[]
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="exile":
            self.game.exile()
            # record

            if self.game.this_round.exiled :
                self.valid_roundvote_times+=1
                if self.game.this_round.exiled in self.cur_alive_wolf_name:
                    self.vote_exile_wolf+=1
            key_role_alive=0
            if self.game.state.seer.name in self.game.this_round.players:
                key_role_alive+=1
            if self.game.state.doctor.name in self.game.this_round.players:
                key_role_alive+=1
            self._wandb_ksr(key_role_alive)

            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "check_for_winner":
            self.game.check_for_winner()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "run_summaries":
            # Process the returned results first
            if pre_observations is not None and pre_observations['action'] == "summarize":
                result, log=action
                pre_player_name = pre_observations['player_name']
                pre_player = self.game.state.players[pre_player_name]
                if result is not None:
                    summary = result.get("summary", None)
                    if summary is not None:
                        summary = summary.strip('"')
                        pre_player._add_observation(f"Summary: {summary}")
                    tqdm.write(f"{pre_player_name} summary: {summary}")
                    self.game.this_round_log.summaries.append((pre_player_name, log))
            
            # To summarize
            next_summ_player = None
            for summ_player in self.game.this_round.players:
                if summ_player not in self.round_summ_players:
                    next_summ_player = summ_player
                    self.round_summ_players.append(summ_player)
                    break
            if next_summ_player is not None:
                player = self.state.players[next_summ_player]
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "summarize",
                    "options": []
                }
                self.action_seq_pointer -= 1
                return observations, None
            else:
                self.round_summ_players=[]
                if self.game.state.winner:
                    tqdm.write(f"Round {self.game.current_round_num} is complete.")
                    self.game.this_round.success = True
                    return None, self.game.state.winner
                else:
                    for name in self.game.this_round.players:
                        if self.game.state.players[name].gamestate:
                            self.game.state.players[name].gamestate.round_number = (
                                    self.game.current_round_num + 1
                            )
                            self.game.state.players[name].gamestate.clear_debate()
                    self.game.current_round_num += 1
                    self.action_seq_pointer=0
                    observations = self.reset_new_round()
                    if observations is not None:
                        return observations, None


    def log_directory(self) -> str:
        import datetime
        pacific_timezone = datetime.timezone(datetime.timedelta(hours=-8))
        timestamp = datetime.datetime.now(pacific_timezone).strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        directory = f"{os.getcwd()}/games/werewolf/visual/logs/{session_id}"
        return directory

    def render(self):
        self.logger.warnning("WereWolfEnv has no render!!!")
        return None
class JsonEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, enum.Enum):
      return o.value
    if isinstance(o, set):
      return list(o)
    return o.__dict__