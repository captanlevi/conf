import numpy as np
import torch
from .core import MemoryElement, SumTree
from .datasets import MemoryDataset
from typing import List
import random



class PMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.7  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.35  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = None
    
    absolute_error_upper = 1.  # clipped abs error



    def getMemoryItem(self, index : int):
        memory = self.memories[index]
        state_length,next_state_length = memory.state.length, memory.next_state.length
        return dict(
            state = memory.state.timeseries[:state_length,:],
            next_state = memory.next_state.timeseries[:next_state_length,:], 
            is_terminal = memory.next_state.isTerminal(),
            reward = memory.reward,
            action = memory.action,
            state_length = state_length,
            label = memory.state.label,
            index = index
        )

    def __init__(self, memories : List[MemoryElement], capacity,batch_size):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.

        Memories will be shuffeled exactly once inplace !!!!
        
        """
        assert capacity <= len(memories)
        self.tree = SumTree(capacity = capacity)
        self.batch_size = batch_size
        self.memories = memories
        self.memory_index_to_priority = self._getMemoryIndexToPriorityMap()
        random.shuffle(memories)

        self.PER_b_increment_per_sampling = 0#(1 - self.PER_b)*self.batch_size/len(self.memories)/5
        self.step = 0
        for i in range(capacity):
            self.store(memory_index= i)
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def _getMemoryIndexToPriorityMap(self):
        mp = dict()

        for i in range(len(self.memories)):
            mp[i] = self._getNewPriority()
        return mp


    def _getNewPriority(self):
        return np.power(self.absolute_error_upper, self.PER_a)
    

    def store(self, memory_index):
        self.tree.add(priority= self.memory_index_to_priority[memory_index], data=  memory_index)  # set the max p for new p

    

    def replace(self):
        indices = np.random.randint(0,len(self.memories),self.batch_size).tolist()
        for index in indices:
            self.store(memory_index= index)


        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self):
        # Create a sample array that will contains the minibatch
        memory_b = [] # this memory_b is a list of indices
        
        b_leaf_indices, b_ISWeights = np.empty((self.batch_size,), dtype=np.int32), np.empty((self.batch_size,), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / self.batch_size       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * self.batch_size) ** (-self.PER_b)

        
        
        for i in range(self.batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            leaf_index, priority, memory_index = self.tree.get_leaf(value)
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            
            b_ISWeights[i] = np.power(self.batch_size * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_leaf_indices[i] = leaf_index
            
            memory_b.append(memory_index)
        
        memory_b = list(map(lambda x : self.getMemoryItem(x), memory_b))
        memory_b = MemoryDataset.collateFn(memory_b)
        memory_b["tree_indices"] = torch.tensor(b_leaf_indices)
        memory_b["IS_weights"] = torch.tensor(b_ISWeights)
        return memory_b
    
    """
    Update the priorities on the tree and replace if needed 
    """
    def batch_update(self, tree_idx, abs_errors):
        self.step += 1
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            data_index = self.tree.getDataIndex(tree_index= ti)
            memory_index = self.tree.data[data_index]
            self.memory_index_to_priority[memory_index] = p
            self.tree.update(ti, p)
        

        self.replace()