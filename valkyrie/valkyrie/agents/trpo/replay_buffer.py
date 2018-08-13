from collections import deque
import pickle
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_path = 0
        self.buffer = deque()

    def get_path(self, size):
        # Randomly sample batch_size examples
        if self.num_path < size:
            return random.sample(self.buffer, self.num_path)
        else:
            return random.sample(self.buffer, size)

    def size(self):
        return self.buffer_size

    def add_single_path(self, path):
        if self.num_path < self.buffer_size:
            self.buffer.append(path)
            self.num_path += 1
        else:
            self.buffer.popleft()
            self.buffer.append(path)

    def add_paths(self, paths):
        self.buffer.extend(paths)
        self.num_path += len(paths)
        while(1):
            if self.num_path< self.buffer_size:
                break
            else:
                self.buffer.popleft()
                self.num_path -= 1

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_path = 0

    def save_menory(self, filename):
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump(self.buffer, fp)
            print("Buffer length saved: " + str(self.num_experiences))

    def load_memory(self, filename):
        #b=[]
        with open(filename, "rb") as fp:  # Unpickling
            self.b = pickle.load(fp)
            #print(len(self.b))
            self.buffer = self.buffer + self.b
            self.num_experiences = len(self.buffer)
            print("Buffer length loaded: "+ str(self.num_experiences))