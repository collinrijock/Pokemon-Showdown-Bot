import numpy as np
import tensorflow as tf

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):

        # -1 indicates that the move does not have a base power
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        moves_accuracy = np.ones(4)

        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (move.base_power / 100)
            moves_accuracy[i] = (move.accuracy)
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        )

        remaining_mon_team = (len([mon for mon in battle.team.values() if mon.fainted]) / 6)
        remaining_mon_opponent = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)

        # Final vector with 14 components
        return np.concatenate ([moves_base_power,moves_accuracy, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent],])

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper (
                battle, fainted_value=2, hp_value=1, victory_value=30
        )

