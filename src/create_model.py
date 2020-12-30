# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from dqn_agent import SimpleRLPlayer
from max_damage import MaxDamagePlayer
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    n_action = len(env_player.action_space)
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 12)))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))
    print("Model complete")
    model.save("models/v2")
