import numpy as np 
import random
from collections import deque, namedtuple
import torch


class ReplayMemory:
    '''ReplayMemory with uniformly distributed probability on samples.'''

    def __init__(self, batch_size, buffer_size, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple('experience',
            field_names=['state','action','reward','next_state','done'])
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        ''' Add new experience in memory.

        Params
        ======
            state(list),
            action(int),
            reward(float),
            next_state(list),
            done(boolean)
        '''
        new_e = self.experience(state, action, reward, next_state, done)
        self.memory.append(new_e)

    def sample(self, device):
        '''Sample a batch of experiences from memory.
        
        Params
        ======
            device(string): either 'cpu' or 'cuda:0'
        '''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
            ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''Return the length of current memory.'''
        return len(self.memory)

class PrioritizedReplayMemory(object):
    '''ReplayMemory with proportional prioritized probability on samples.'''

    def __init__(self, batch_size, buffer_size, seed):
        self.seed = random.seed(seed)
        self.epsilon = 0.01 # small amount to avoid zero priority
        self.alpha = 0.6 # [0~1] convert the importance of TD error to priority,
                         # it is a trade-off between using priority and totally uniformly randomness
        self.absolute_error_upper = 1.0 # clipped abs error (abs error is the absolute value of TD error)
        self.beta = 0.4 # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.batch_size = batch_size
        self.replay_buffer = SumTree(buffer_size)
        self.experience_counter = 0 # track the number of experience being added in 

    def add(self, state, action, reward, next_state, done):
        '''Add new experience meanwhile assign it the maximum priority from the current group.'''

        self.experience_counter += 1
        # assign maximal priority from current group for new experience
        max_priority = np.max(self.replay_buffer.tree[-self.replay_buffer.buffer_size:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.replay_buffer.store(max_priority, state, action, reward, next_state, done) # set the max_priority for new experience
    
    def sample(self, device):
        '''Sample a batch of examples. 
        First, divide the range (0, total_priority) into batch_size segments. Then uniformly sample one value from each segment. 
        Use the value in get_leaf() to search through the tree and retrieve the closest associated leaf_idx.
        In addition, compute the relevant ISWeights (Importance-Sampling Weights).

        Params
        ======
            device(string): either 'cpu' or 'cuda:0'

        Return
        ======
            batch_idx (array): leaf index for SumTree
            (states, actions, rewards, next_states, dones) (tensor tuple): sampled experiences in tuple
            batch_ISWeights (tensor array): importance-sampling weights for experiences
        '''

        batch_experiences = [None] * self.batch_size
        batch_idx, batch_ISWeights = np.empty((self.batch_size,), dtype=np.int32), np.empty((self.batch_size,1), dtype=np.float32)
        priority_segment = self.replay_buffer.total_priority/self.batch_size
        
        # increase beta each time when sampling a new minibatch
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # max = 1

        # compute the max ISweight
        min_prob = np.max(self.replay_buffer.tree[-self.replay_buffer.buffer_size:]) / self.replay_buffer.total_priority
        max_ISWeight = np.power(min_prob*self.replay_buffer.buffer_size, -self.beta) 

        for i in range(self.batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a,b)
            leaf_idx, priority, experience_tuple = self.replay_buffer.get_leaf(value)
            prob = priority/self.replay_buffer.total_priority # convert from priority to probability
            batch_ISWeights[i, 0] = np.power(prob*self.replay_buffer.buffer_size, -self.beta)/max_ISWeight # normalize ISWeights
            batch_idx[i] = leaf_idx
            batch_experiences[i] = experience_tuple
        
        states = torch.from_numpy(np.vstack([e.state for e in batch_experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch_experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch_experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch_experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in batch_experiences if e is not None]).astype(np.uint8)
            ).float().to(device)        

        batch_ISWeights = torch.from_numpy(batch_ISWeights).float().to(device)

        return batch_idx, (states, actions, rewards, next_states, dones), batch_ISWeights
    
    def batch_update(self, leaf_indexes, abs_errors):
        '''Update the sample batch's priorities and their parent node's priorities.
        Notice that the absolute TD error is clipped to fall within 1 for stability reason.

        Params
        ======
            leaf_indexes (numpy array): leaf indexes for SumTree
            abs_errors (numpy array): the absolute value of TD error

        '''

        abs_errors += self.epsilon # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper) #  the element-wise minima
        priorities = np.power(clipped_errors, self.alpha)
        for leaf_i, p in zip(leaf_indexes, priorities):
            self.replay_buffer.update(leaf_i, p)
    
    def __len__(self):
        '''Track how many experiences being added in the memory'''
        return self.experience_counter



class SumTree(object):
    '''This code is from:
    1. https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    2. https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb

    Store experience in the memory and its priority in the tree.
    '''

    def __init__(self, buffer_size):
        self.memory_idx = 0
        self.buffer_size = buffer_size
        self.tree = np.zeros(2*self.buffer_size - 1)
        self.experience = namedtuple('experience',
            ['state','action','reward','next_state','done']) # initialize namedtuple class
        self.memory = [None] * buffer_size
    
    def store(self, priority, state, action, reward, next_state, done):
        '''Store new experience in memory and update the relevant priorities in tree. 
        The new experience will overwrite the old experience from the beginning once the memory is full.

        Params
        ======
            priority (float)
            state (array)
            action (int)
            reward (float)
            next_state (array)
            done (boolean)
        '''

        leaf_idx  = self.memory_idx + self.buffer_size - 1
        new_e = self.experience(state, action, reward, next_state, done)        
        self.memory[self.memory_idx] = new_e # update experience
        self.update(leaf_idx, priority) # update priorities in tree

        self.memory_idx += 1
        if self.memory_idx >= self.buffer_size: # replace the old experience when exceeding buffer_size
            self.memory_idx = 0
    
    def update(self, leaf_idx, priority):
        '''Update the priority of the leaf_idx and also propagate the priority-change through tree.
        
        Params
        ======
            leaf_idx (int)
            priority (float)
        '''

        priority_change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority

        # then propagate the priority change through tree
        tree_idx = leaf_idx
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += priority_change

    def get_leaf(self, value):
        '''Use the value in get_leaf() to search through the tree and 
        retrieve the closest associated leaf_idx and its memory.
        
        Params
        ======
            value (float): used to search through the tree for closest leaf_idx

        Return
        ======
            leaf_idx (int)
            priority (float)
            experience (namedtuple)

        '''

        parent_idx = 0
        while True: # this node's left and right kids
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else: # downward search, always search for a higher priority node
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx                    
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        memory_idx = leaf_idx - self.buffer_size + 1
        return leaf_idx, self.tree[leaf_idx], self.memory[memory_idx]

    @property
    def total_priority(self):
        return self.tree[0] # the total priorities stored in the root node






    



    

    
    

