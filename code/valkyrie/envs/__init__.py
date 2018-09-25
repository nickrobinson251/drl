import os

from .balancebot_env import BalancebotEnv
from .pendulum_env import InvertedPendulumEnv
from .two_room_grid_env import TwoRoomGridEnv
from .valkyrie_env import ValkyrieBalanceEnv

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
os.sys.path.insert(0, PARENT_DIR)
