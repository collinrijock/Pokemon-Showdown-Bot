import numpy as np
import tensorflow as tf
import pdb
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

    def embed_battle(self, battle):
        # Data
        ally_team = self.get_ally_team_data(battle.team,battle)
        enemy_team = self.get_enemy_team_data(battle.opponent_team,battle)
        fixed_input_data = np.concatenate([ally_team, enemy_team])
        breakpoint()
        return fixed_input_data

    def get_ally_team_data(self,team,battle):
        pokemon_array = [-1.0] * 6

        for i,pokemon in enumerate(team.values()):

            #Initialization of arrays on each iteration
            moves_base_power = [-1.0] * 4
            moves_dmg_multiplier =  [-1.0] * 4
            moves_accuracy = [-1.0] * 4
            moves_healing = [-1.0] * 4
            moves_hits = [-1.0] * 4
            status = 0

            if pokemon.status:
                status = pokemon.status.value / 10
            for k,move in enumerate(pokemon.moves.values()):
                moves_base_power[k] = move.base_power / 100
                moves_accuracy[k] = move.accuracy
                if move.type:
                    moves_dmg_multiplier[k] = (move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                    ))
                moves_healing[k] = move.heal
                moves_hits[k] = max(move.n_hit)

            pokemon_array[i] = [
                pokemon.current_hp_fraction,
                int(pokemon.active),
                status,
                moves_base_power,
                moves_dmg_multiplier,
                moves_accuracy,
                moves_healing,
                moves_hits
            ]
        return pokemon_array

    def get_enemy_team_data(self,team,battle):
        pokemon_array = [-1.0] * 6
        for i in range(6):

            #Initialization of arrays on each iteration
            moves_base_power = [-1.0] * 4
            moves_dmg_multiplier =  [-1.0] * 4
            moves_accuracy = [-1.0] * 4
            moves_healing = [-1.0] * 4
            moves_hits = [-1.0] * 4
            status = 0
            if i < len(team):
                pokemon = list(team.values())[i]
                if pokemon.status:
                    status = pokemon.status.value / 10
                for k,move in enumerate(pokemon.moves.values()):
                    moves_base_power[k] = move.base_power / 100
                    moves_accuracy[k] = move.accuracy
                    if move.type:
                        moves_dmg_multiplier[k] = (move.type.damage_multiplier(
                            battle.opponent_active_pokemon.type_1,
                            battle.opponent_active_pokemon.type_2,
                        ))
                    moves_healing[k] = move.heal
                    moves_hits[k] = max(move.n_hit)

            pokemon_array[i] = [
                pokemon.current_hp_fraction,
                int(pokemon.active),
                status,
                moves_base_power,
                moves_dmg_multiplier,
                moves_accuracy,
                moves_healing,
                moves_hits
            ]
        return pokemon_array

