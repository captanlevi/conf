from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .core import MemoryElement
from typing import List
import torch
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
from ...utils.commons import augmentData
import numpy as np

class MemoryDataset(Dataset):
    """
    This dataset is used for internal training for the DQN 

    num_class are classes excluding the extra wait 
    """
    def __init__(self, memories : List[MemoryElement],num_classes : int,min_length : int,max_length):
        super().__init__()
        self.memories = memories
        self.num_classes = num_classes
        self.min_length = min_length
        self.max_length = max_length

    def __getitem__(self, index : int):

    
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
        

    def __getActions(self):
        return list(map(lambda x : x.action,self.memories))

        
    def __len__(self):
        return len(self.memories)
    
    @staticmethod
    def collateFn(batch):   
        state = list(map(lambda x : torch.tensor(x["state"]).float(),batch ))
        next_state = list(map(lambda x : torch.tensor(x["next_state"]).float(), batch))

        state = pack_sequence(state,enforce_sorted= False)
        next_state = pack_sequence(next_state,enforce_sorted= False)


        is_terminal = torch.tensor(list(map(lambda x : x["is_terminal"],batch))).bool()
        reward = torch.tensor(list(map(lambda x : x["reward"],batch))).float()
        action = torch.tensor(list(map(lambda x : x["action"],batch))).long()
        label = torch.tensor(list(map(lambda x : x["label"], batch))).long()
        index = torch.tensor(list(map(lambda x : x["index"], batch))).long()
        state_length = torch.tensor(list(map(lambda x :x["state_length"], batch))).int()
        return dict(
            state = state, next_state = next_state, action = action, reward = reward,
            is_terminal = is_terminal,state_length = state_length,label = label, index = index
        )
    


    




    