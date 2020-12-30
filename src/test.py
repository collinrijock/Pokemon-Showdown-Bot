import asyncio
import time
import pdb

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        print(battle.active_pokemon.status)
        breakpoint()
        return self.choose_random_move(battle)


async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(battle_format="gen8randombattle")
    max_damage_player = MaxDamagePlayer(battle_format="gen8randombattle")

    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=1)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (max_damage_player.n_won_battles, time.time() - start)
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
