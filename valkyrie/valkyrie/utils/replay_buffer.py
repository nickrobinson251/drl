from collections import deque, namedtuple
import os
import pickle
import random


Experience = namedtuple(
    "Experience",
    ["observation", "action", "reward", "next_observation", "done"])


class ReplayBuffer(object):
    """A replay buffer which stores experiences and can return random samples.

    Parameters
    ----------
    size : int
        maximum size for buffer. One the size limit is reached, when a new
        experiece is added to the buffer, the oldest experience is discarded.
    seed : int (default=None)
        number used to seed the random number generator used for sampling

    Attributes
    ----------
    contents : deque

    Methods
    -------
    add(observation, action, reward, next_observation, done)
    random_batch(batch_size)
    clear()
    load(direcotry, verbose=False)
    save(direcotry, verbose=False)
    """

    def __init__(self, size, seed=None):
        self.seed = seed
        if seed:
            random.seed(seed)
        self.size = int(size)
        self.contents = deque(maxlen=self.size)

    def add_sample(self, observation, action, reward, next_observation, done):
        """Add an experience to the replay buffer.

        Parameters
        ----------
        observation
        action
        reward : float
        next_observation
        done : bool
        """
        self.contents.append(
            Experience(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done))

    def random_batch(self, batch_size):
        """Randomly sample a list of batch_size experiences."""
        return random.sample(list(self.contents), k=batch_size)

    def clear(self):
        """Clear the replay contents."""
        self.contents.clear()

    def load(self, directory, verbose=False):
        filename = os.path.join(directory, "memories.pickle")
        with open(filename, "rb") as f:
            memories = pickle.load(f)
            self.contents += memories
        if verbose:
            print("Loaded {} memories from '{}'. Buffer length now {:,}".format(
                len(memories), filename, len(self)))

    def save(self, directory, verbose=False):
        filename = os.path.join(directory, "memories.pickle")
        with open(filename, "wb") as f:
            pickle.dump(self.contents, f)
        if verbose:
            print("Saved buffer of size {} to '{}'".format(len(self), filename))

    def __len__(self):
        return len(self.contents)

    def __repr__(self):
        return "ReplayBuffer(**{})".format(dict(
            size=self.size,
            seed=self.seed))
