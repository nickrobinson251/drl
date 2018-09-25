import pickle
from pprint import pprint


def record_configuration(config, dir):
    """Write configuration to dir/'configuration.txt'."""
    txt_file = open(dir + "/configuration.txt", "w")
    pprint(config, stream=txt_file)
    txt_file.close()


def save_configuration(config, dir):
    """Write cnfiguration dict to dir/'configuration.pickle'."""
    output = open(dir + "/configuration.pickle", "wb")
    pickle.dump(config, output)
    output.close()


def update_configuration(config, dir):
    """Update curretn config with dict from file dir/'configuration.pickle'."""
    pkl_file = open(dir + "/configuration.pickle", "rb")
    conf_temp = pickle.load(pkl_file)
    config.update(conf_temp)
    pkl_file.close()


def load_configuration(dir):
    """Load configuration dict from file dir/'configuration.pickle'."""
    pkl_file = open(dir + "/configuration.pickle", "rb")
    config = pickle.load(pkl_file)
    pkl_file.close()
    return config
