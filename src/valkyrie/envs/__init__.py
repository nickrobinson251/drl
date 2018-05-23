import os

from valkyrie.envs.valkyrie_env import ValkyrieEnv

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
os.sys.path.insert(0, PARENT_DIR)
