
import numpy as np 
import random
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from replaymemory import PrioritizedReplayMemory, ReplayMemory

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
LR = 6.25e-5            # learning rate 
LEARN_EVERY_STEP = 4    # how often to activate the learning process
UPDATE_EVERY_STEP = 10000 # how often to update the target network parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class PrioritizedAgent:
    '''Interact with and learn from the environment.'''

    def __init__(self, state_size, action_size, seed, is_prioritized_sample=False):
        '''Initialize an Agent.

        Params
        ======
            state_size (int): the dimension of the state
            action_size (int): the number of actions
            seed (int): random seed
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0 # Initialize time step (for tracking LEARN_EVERY_STEP and UPDATE_EVERY_STEP)

        self.is_prioritized_sample = is_prioritized_sample

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetowrk_target = QNetwork(self.state_size, self.action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        if self.is_prioritized_sample == False:
            self.replay_memory = ReplayMemory(BATCH_SIZE, BUFFER_SIZE, seed)
        else:
            self.replay_memory = PrioritizedReplayMemory(BATCH_SIZE, BUFFER_SIZE, seed)
            
    
    def act(self, state, epsilon=0.):
        '''Returns actions for given state as per current policy.
        
        Params
        ======
            state (array-like): current state
            epsilon (float): for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # shape of state (1, state)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        if random.random() <= epsilon: # random action
            action = random.choice(np.arange(self.action_size))
        else: # greedy action
            action = np.argmax(action_values.cpu().data.numpy()) # pull action values from gpu to local cpu
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        # add new experience in memory
        self.replay_memory.add(state, action, reward, next_state, done)

        # activate learning every few steps
        self.t_step = self.t_step + 1
        if self.t_step % LEARN_EVERY_STEP == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_memory) >= BUFFER_SIZE and self.is_prioritized_sample==False:
                experiences = self.replay_memory.sample(device)
                self.learn(experiences, GAMMA)
            elif len(self.replay_memory) >= BUFFER_SIZE and self.is_prioritized_sample==True:
                batch_idx, experiences, batch_ISWeights = self.replay_memory.sample(device)
                self.learn(experiences, GAMMA, ISWeights=batch_ISWeights, leaf_idxes=batch_idx)

    
    def learn(self, experiences, gamma, ISWeights=None, leaf_idxes=None):
        """Update value parameters using given batch of experience tuples.

        If is_prioritized_sample, then weights update is adjusted by ISWeights. 
        In addition, Double DQN is optional.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            ISWeights (tensor array): importance-sampling weights for prioritized experience replay
            leaf_idxes (numpy array): indexes for update priorities in SumTree

        """

        # compute and minimize the loss
        if self.is_prioritized_sample == False:
            states, actions, rewards, next_states, dones = experiences

            q_local_chosen_action_values = self.qnetwork_local.forward(states).gather(1, actions)
            q_target_action_values = self.qnetowrk_target.forward(next_states).detach() # # detach from graph, don't backpropagate
            q_target_best_action_values = q_target_action_values.max(1)[0].unsqueeze(1) # shape (batch_size, 1)
            q_target_values = rewards + gamma * q_target_best_action_values * (1 - dones) # zero value for terminal state 
        
            loss = F.mse_loss(q_local_chosen_action_values, q_target_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            states, actions, rewards, next_states, dones = experiences
            
            q_local_chosen_action_values = self.qnetwork_local.forward(states).gather(1, actions)
            #q_local_next_actions = self.qnetwork_local.forward(next_states).detach().max(1)[1].unsqueeze(1) # shape (batch_size, 1)
            q_target_action_values = self.qnetowrk_target.forward(next_states).detach()
            q_target_best_action_values = q_target_action_values.max(1)[0].unsqueeze(1) # shape (batch_size, 1)
            #q_target_best_action_values = q_target_action_values.gather(1, q_local_next_actions) # Double DQN
            q_target_values = rewards + gamma * q_target_best_action_values * (1 - dones) # zero value for terminal state

            abs_errors = torch.abs(q_target_values - q_local_chosen_action_values).cpu().data.numpy() # pull back to cpu
            self.replay_memory.batch_update(leaf_idxes, abs_errors) # update priorities in SumTree

            loss = F.mse_loss(q_local_chosen_action_values, q_target_best_action_values, reduce=False)
            loss = (ISWeights * loss).mean() # adjust TD loss by Importance-Sampling Weights
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step % UPDATE_EVERY_STEP == 0:
            self.update(self.qnetwork_local, self.qnetowrk_target)
    
    def update(self, local_netowrk, target_network):
        """Hard update model parameters, as indicated in original paper.
        
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to

        """
        for local_param, target_param in zip(local_netowrk.parameters(), target_network.parameters()):
            target_param.data.copy_(local_param.data)
        



        







        
        
         

