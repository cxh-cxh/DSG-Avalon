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

GAME = """You are playing a digital version of the social deduction game Avalon.

GAME RULES:
- Player Roles: {{num_players}} players - {{num_good}} Loyal Servants of Arthur (including Merlin, Percival, and optional roles like Lancelot) and {{num_evil}} Minions of Mordred (including Morgana, Assassin, Mordred, and optional roles like Oberon).
- In this game, there is 1 Merlin, 2 Servants, 1 Assassin and 1 Minion.
- Rounds consist of three phases:
    - Team Selection Phase: A leader proposes a team for a mission. Players vote to approve/reject the team. Failed votes pass leadership.
    - Mission Phase: Selected players secretly choose to succeed or fail the mission. Evil players may sabotage.
    - Assassination Phase: If 3 missions succeed, the Assassin tries to kill Merlin.
- Special Role Abilities:
    - Merlin sees all Evil players (except Mordred).
    - Percival knows Merlin's identity.
    - Morgana disguises as Merlin to Percival.
    - Assassin determines Evil's final chance to win.
- Winning Conditions:
    - Loyal Servants win by completing 3 successful missions and protecting Merlin from assassination.
    - Minions of Mordred win by sabotaging 3 missions or killing Merlin after 3 successful missions.
"""

STATE = """GAME STATE:
- It is currently Round {{round}}. {% if round == 0 %}The game has just begun.{% endif %}
- You are {{name}} the {{role}}. {{player_context}}
{% if personality -%}
- Personality: {{ personality }}
{% endif -%}
"""

OBSERVATIONS = """{% if observations|length -%}YOUR OBSERVATIONS:
{% for turn in observations -%}
{{ turn }}
{% endfor %}
{% endif %}
"""

MESSAGE_SO_FAR_THIS_ROUND = """\nROUND {{round}} MESSAGE:
{% if message|length -%}
{% for turn in message -%}
{{ turn }}
{% endfor -%}
{% else -%}
The discussion has not begun.{% endif %}\n\n
"""

TEAM_HISTORY = """
{% if team_history|length -%}
{% for msg in team_history -%}
{{ msg }}
{% endfor -%}
{% else -%}
No team has been proposed.{% endif %}\n
"""

TEAM_INFO = """
CURRENT TEAM IS:
{{ team }}

CURRENT TEAM LEADER IS:
{{ leader }}
"""


PREFIX = f"""{GAME}

{STATE}

{OBSERVATIONS}
""".strip()

TEAM = (
    PREFIX
    + MESSAGE_SO_FAR_THIS_ROUND
    + TEAM_HISTORY
    + """INSTRUCTIONS:
- You are the team leader for the current team. For this round, the team size is {{team_size}} (including yourself). Choose {{team_size - 1}} player(s) as your teammate(s) and announce the reason for this team.
- Your proposal sets the tone for trust or suspicion. Prioritize players who:  
  - Have consistent voting patterns.  
  - Were on previous successful quests (if Good) or failed quests (if Evil).  
  - Avoided extreme accusations or defensiveness.

{% if role in ['Servant', 'Merlin', 'Percival'] -%}  
- Balance the team with proven allies and unproven players to test loyalty.  
- Example: "Let's include [Player X]—they've voted against failed teams before."  
- Watch reactions: Evil players may over-praise or criticize reasonable proposals.  
{% if role == 'Merlin' -%}  
- As Merlin, subtly avoid including known Evil players to protect your identity. 
{% endif %} 

{% elif role in ['Minion', 'Morgana', 'Assassin', 'Mordred', 'Oberon'] -%}  
- Propose teams with one ally + innocents to avoid suspicion. For example: "I'll take [Evil ally] and [trusted Good player]—seems balanced."  
- If confident, push for a team with multiple allies to sabotage the quest.  
- Blame failures on 'unlucky picks' or accuse Good players of inconsistency.
{% endif %}  

```json
{
  "reasoning": "string", // Explain your reasoning about who you should choose as teammate(s). Avoid using violent or harmful language.
  "say": "string", // Announcement about the team, every player will receive this. Avoid using violent or harmful language.
  "team": "string" // Names of the player. Choose {{team_size - 1}} player(s) from: {{options}}, seperate the names with comma ',', for example, "Amily, Bob, Cook".
}"""
)

TEAM_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
        "team": {"type": "string"},
    },
    "required": ["reasoning", "say", "team"],
}

MESSAGE = (
    PREFIX
    + MESSAGE_SO_FAR_THIS_ROUND
    + TEAM_HISTORY
    + TEAM_INFO
    + """INSTRUCTIONS:
- The team for the next mission is decided. You are speaking next in the discussion as {{name}} the {{role}}.
{% if role in ['Servant', 'Merlin', 'Percival'] -%}  
- Your goal is to complete 3 successful missions and protect Merlin.  
- Analyze team proposals: Evil players often avoid leadership or push for teams with hidden allies.  
- Share observations cautiously. Over-sharing might expose Merlin or confuse Percival.  
- Watch for contradictions in voting patterns or mission outcomes.  

{% if role == 'Merlin' -%}  
- You see all Evil players (except Mordred). Guide the team subtly—drop hints without revealing your identity.  
- Avoid being too vocal or decisive; the Assassin is watching.  
- Trust Percival to recognize you, but beware of Morgana's deception.  

{% elif role == 'Percival' -%}  
- You know Merlin (and see Morgana disguised as Merlin). Protect Merlin by rejecting suspicious teams.  
- Use your knowledge to sway votes discreetly. For example: “I trust [Player X]'s judgment on this team.”  
- Never confirm Merlin's identity publicly—Evil will exploit it.  

{% else -%}  
- As a Servant, focus on logic, not emotion. Evil thrives on chaos.  
- If a mission fails, scrutinize who proposed the team and who was quiet during voting.  
- Defend players who act consistently loyal, but stay open to new evidence.  

{% endif %}  

{% if is_leader -%}
- You are the team leader for the next team. You may express your thoughts on the members of the next team.
- Balance the team with proven allies and unproven players to test loyalty.  
- Example: 'Let's include [Player X]—they've voted against failed teams before.'  
- Watch reactions: Evil players may over-praise or criticize reasonable proposals.  

{% if role == 'Merlin' -%}  
- As Merlin, subtly avoid including known Evil players to protect your identity. 
{% endif %}  

{% endif %}  

{% elif role in ['Minion', 'Morgana', 'Assassin', 'Mordred', 'Oberon'] -%}  
- Your goal is to sabotage 3 missions or assassinate Merlin.  
- Blend in: Agree with Good players' logic but steer teams toward including Evil allies.  
- Pretend to doubt trustworthy players. For example: 'Why does [Player X] always avoid leadership?'
- Sabotage missions sparingly—too many failures expose your team.  

{% if role == 'Morgana' -%}  
- Disguise yourself as Merlin to Percival. Act 'helpful' to gain trust.  
- Subtly undermine Good's strategies. For example: "I'm not sure about this team… maybe swap [Player Y]?"
- Avoid overt lies; let Percival's confusion work in your favor.  

{% elif role == 'Assassin' -%}  
- Memorize Merlin's possible identities. Watch for players who:  
    - Hesitate to approve teams with Evil members.  
    - Hint at hidden knowledge (e.g., 'I have a bad feeling about [Evil player]').  
- In the Assassination Phase, target the most Merlin-like player.  

{% elif role == 'Mordred' -%}  
- Merlin can't see you. Act boldly—propose teams with Evil allies or accuse Servants.  
- Use your anonymity to sow doubt: “If Merlin's so powerful, why haven't they stopped the fails?”  

{% elif role == 'Oberon' -%}  
- You don't know other Evil players. Sabotage missions without coordination.  
- Pretend to distrust actual Minions to avoid suspicion.  

{% endif %}  
{% endif %}

```json
{
  "reasoning": "string", // Based on the game's current state and your role's objectives, outline your strategy. What do you want to achieve? What type of message can help you get there? Avoid using violent or harmful language.
  "say": "string" // Your public statement in the discussion. Be concise and persuasive. Respond directly to what the other players have said. Avoid simply repeating what others have said or reguritating the instructions above.
}
"""
)

MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
    },
    "required": ["reasoning", "say"],
}

APPROVE = (
    PREFIX
    + MESSAGE_SO_FAR_THIS_ROUND
    + TEAM_HISTORY
    + TEAM_INFO
    + """INSTRUCTIONS:
- You are a candidate member of the current team. You need to agree or reject to join the team.
- Think strategically as {{role}} and decide your choice.
- Your chioce will be announced publically to all players.
- Your choice reveals your trust in the team's composition. Use it to:  
  - Signal suspicion or confidence in the leader's judgment.  
  - Influence the group's perception of who is Good or Evil. 

{% if role in ['Servant', 'Merlin', 'Percival'] -%}  
- Approve the team if: 
  - The leader has a history of proposing successful teams.  
  - The team includes players who voted against previous failed quests.  
{% if role == 'Merlin' -%}  
  - As Merlin, you know none of the proposed players are Evil (except Mordred).  
{% endif %} 

- Reject the team if:  
  - The leader is untrusted or proposed questionable teams before.  
  - The team includes players who were on failed quests or avoided leadership.  
  - Example: "I'm rejecting this team—[Player X] was on two failed quests. Let's try someone new."  

{% if role == 'Merlin' -%}  
- Subtly steer votes: Reject teams with Evil players (but avoid seeming too certain).  
- Example: "I don't trust [Evil player]'s vibe. Can we swap them out?"  

{% elif role == 'Percival' -%}  
- Protect Merlin: Approve teams where Merlin is included, but reject if Morgana is likely disguised on the team.  

{% endif %}

{% elif role in ['Minion', 'Morgana', 'Assassin', 'Mordred', 'Oberon'] -%}  
- Approve the team if:  
  - It includes at least one Evil ally (to sabotage the quest).  
  - Rejecting would make you look suspicious (e.g., the team seems reasonable).  
- Reject the team if:  
  - It's full of Loyal players (to force leadership changes and chaos).  
  - Example: "This team feels off. Why exclude [Evil ally]? I'm voting no." 

{% if role == 'Morgana' -%}  
- Pretend to be cautious: "I'm approving, but [Player X] makes me nervous. Let's watch them." 

{% elif role == 'Mordred' -%}  
- Exploit your anonymity: Push to approve teams with yourself. "I've been loyal all game—include me!" 

{% endif %} 
{% endif %} 

```json
{
  "reasoning": "string", // Explain your reasoning about whether to agree to join the team. Avoid using violent or harmful language.
  "approve": "string" // Agree (yes) or reject (no) to join the current team. Choose from: {{options}}
}"""
)

APPROVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "approve": {"type": "string"},
    },
    "required": ["reasoning", "approve"],
}

VOTE = (
    PREFIX
    + MESSAGE_SO_FAR_THIS_ROUND
    + TEAM_HISTORY
    + TEAM_INFO
    + """INSTRUCTIONS:
- The team is going on a mission. Vote 'succeed' to help the mission or 'failure' to sabotage the mission.
- The current round is round {{round}}, and {{fail_cnt}} 'failure' votes will make the mission fail.
- Think strategically as {{role}} and decide your choice.
- Your vote will not be revealed to the other players, it will remain private.

{% if role in ['Servant', 'Merlin', 'Percival'] -%}  
- Always vote 'success'. You have no choice—Good players cannot sabotage.  

{% elif role in ['Minion', 'Morgana', 'Assassin', 'Mordred', 'Oberon'] -%}  
- Sabotage strategically:  
  - Early Rounds (1-2): Fail only if necessary. Letting Good win 1-2 missions builds false trust.  
  - Critical Rounds (3-5): Fail decisively. Coordinate with allies to ensure enough failures.  
  - Avoid over-sabotaging. Example: If only 1 'failure' vote is needed, let one Evil player fail—multiple fails expose your numbers.  
- Blend in:  
  - If you vote 'success', act relieved: "Thank goodness we trusted the right people!"  
  - If you vote 'failure', later accuse a Good player: "Why did [Player X] insist on that team? They must be Evil!"  

{% if role == 'Oberon' -%}  
- You don't know other Evil players. Fail sparingly to avoid accidental ally exposure.  
- Pretend to analyze failures like a Good player: "This team had [Player Y]—they've been suspicious all game!" 
{% endif %}   
{% endif %}

```json
{
  "reasoning": "string", // Explain your reasoning about why you vote. Avoid using violent or harmful language.
  "vote": "string" // Help (success) or sabotage (failure) the mission. Choose from: {{options}}
}"""
)

VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "vote": {"type": "string"},
    },
    "required": ["reasoning", "vote"],
}

MISSION_INFO = """
{% if success -%}  
The team of {{ team }} with leader {{ leader }} succeeded in the mission.
{% else -%}  
The team of {{ team }} with leader {{ leader }} failed the mission.
{% endif %}  
"""

POST_MISSION = (
    PREFIX
    + MESSAGE_SO_FAR_THIS_ROUND
    + MISSION_INFO
    + """INSTRUCTIONS:
{% if success -%}  
- The current mission succeeded. You need to analyze the mission.
- The current round is round {{round}}, and {{fail_cnt}} 'failure' votes will make the mission fail.
- Think strategically as {{role}} and decide your speech.
- Your speech will be announced publically to all players.

{% if role in ['Servant', 'Merlin', 'Percival'] -%}  
- Focus on who was *on the team* and their past votes/proposals. Evil players often avoid overt suspicion. 
- Analyze the team composition:  
  - Who proposed the team?  
  - Who acted defensive during approving?  
  - How many 'failure' votes were needed to make the mission fail?  
- Use failures to narrow down suspects. For example: "If mission at Round 2 failed with 3 players, at least one is Evil."
{% elif role in ['Minion', 'Morgana', 'Assassin', 'Mordred', 'Oberon'] -%}  
- You may downplay success: "This proves nothing—Evil might be hiding."  
- You may act relieved: "Thank goodness we trusted the right people!" 
{% endif %}  

{% else -%}  
- The current mission failed. You need to analyze the mission.
- The current round is round {{round}}, and {{fail_cnt}} 'failure' votes will make the mission fail.
- Think strategically as {{role}} and decide your speech.
- Your speech will be announced publically to all players.

{% if role in ['Servant', 'Merlin', 'Percival'] -%}  
- Focus on who was on the team and their past votes/proposals. Evil players often avoid overt suspicion.  
{% if role == 'Merlin' -%}  
- Avoid directly accusing Evil players; the Assassin is watching.
{% endif %}  

{% elif role in ['Minion', 'Morgana', 'Assassin', 'Mordred', 'Oberon'] -%}  
- Blame innocent players: "Why did [Good player] approve the team? They must be Evil!" 

{% if role == 'Morgana' -%}  
- Gaslight Percival: "I'm starting to think [Merlin] isn't who they claim to be."
{% endif %}  
{% endif %} 
{% endif %}  

```json
{
"reasoning": "string", // Your reasoning about what you should say.
"say": "string" // Announce your opinions about the mission results.
} """
)

POST_MISSION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
    },
    "required": ["reasoning", "say"],
}

SUMMARIZE = (
    PREFIX
    + MESSAGE_SO_FAR_THIS_ROUND
    + MISSION_INFO
    + """INSTRUCTIONS:
- Reflect on the round's teams, discussion and mission results as {{role}}.
- Summarize the key points and strategic implications.

```json
{
"reasoning": "string", // Your reasoning about what you should remember from this debate and why this information is important.
"summary": "string" // Summarize the key points and noteworthy observations from the debate in a few sentences. Aim to make notes on as many players as you can — even seemingly insignificant details might become relevant in later rounds. Be specific. Remember, you are {{role}}. Write your summary from their point of view using "I" and "me."
} """
)

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["reasoning", "summary"],
}


ASSASSINATE = (
    PREFIX
    + """INSTRUCTIONS:
- The Good side is about to win and it is your last chance. As Assassin, you must assassinate Merlin to win.
- Review earlier phases for players who:  
    - Subtly opposed teams with Evil members.  
    - Hinted at hidden knowledge (e.g., "I'm certain [Player X] is loyal").  
    - Avoided voting extremes (always approve/reject).  
- Target quiet players—Merlin often stays understated to avoid detection.  
- If unsure, eliminate the most trusted player (likely Merlin).

```json
{
"reasoning": "string", // Your reasoning about which player is likely to be Merlin.
"assassinate": "string" // Name of the player to be assassinated. Choose from: {{options}}
} """
)

ASSASSINATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "assassinate": {"type": "string"},
    },
    "required": ["reasoning", "assassinate"],
}

ACTION_PROMPTS_AND_SCHEMAS = {
    "team": (TEAM, TEAM_SCHEMA),
    "message": (MESSAGE, MESSAGE_SCHEMA),
    "approve": (APPROVE, APPROVE_SCHEMA),
    "vote": (VOTE, VOTE_SCHEMA),
    "post_mission": (POST_MISSION, POST_MISSION_SCHEMA),
    "summarize": (SUMMARIZE, SUMMARIZE_SCHEMA),
    "assassinate": (ASSASSINATE, ASSASSINATE_SCHEMA),
}
