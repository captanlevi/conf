from torch.utils.data import Dataset,DataLoader
import numpy as np

class SineWaveDataset(Dataset):
    def __init__(self,freq_range,phase_range,num_timesteps,ts_dim,dataset_length,is_sin = True) -> None:
        super().__init__()
        self.freq_range = freq_range
        self.phase_range = phase_range
        self.timestamps = np.arange(num_timesteps)
        self.ts_dim = ts_dim
        self.data_length = dataset_length
        self.is_sin = is_sin


    def __sampleFromRange(self,range):
        rand = np.random.random()
        return range[0] + rand*(range[1] - range[0])

    def __generateSingleRandomSin(self):
        sampled_freq = self.__sampleFromRange(self.freq_range)
        sampled_phase = self.__sampleFromRange(self.phase_range)
        if self.is_sin == True:
            return np.abs(np.sin(self.timestamps*sampled_freq + sampled_phase))
        else:
            return np.abs(.8*np.sin(self.timestamps*sampled_freq + sampled_phase))
    

    def generateData(self):

        sines = []
        for _ in range(self.ts_dim):
            sines.append(self.__generateSingleRandomSin())
        
        sines = np.stack(sines,axis= 0)
        return sines.T
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        return dict(data = self.generateData(), label = -1)


    def __len__(self):
        return self.data_length


class FreqWaveDataset(Dataset):
    def __init__(self,ts_dim,dataset_length,freq_ranges = ((.1,.5),(1,1.4)),phase_range = (.1,.11),num_timesteps= 25) -> None:
        super().__init__()
        self.freq_ranges = freq_ranges
        self.phase_range = phase_range
        self.timestamps = np.arange(num_timesteps)
        self.ts_dim = ts_dim
        self.data_length = dataset_length
        self.label_to_index = dict()
        for i in range(len(freq_ranges)):
            self.label_to_index[i] = i 


    def __sampleFromRange(self,range):
        rand = np.random.random()
        return range[0] + rand*(range[1] - range[0])

    def __generateSingleRandomSin(self,freq_range):

        sampled_freq = self.__sampleFromRange(freq_range)
        sampled_phase = self.__sampleFromRange(self.phase_range)
        return np.abs(np.sin(self.timestamps*sampled_freq + sampled_phase))
    

    def generateData(self):

        waves = []
        range_index = np.random.randint(low = 0, high= len(self.freq_ranges))
        
        for _ in range(self.ts_dim):
                waves.append(self.__generateSingleRandomSin(freq_range= self.freq_ranges[range_index]))
        waves = np.stack(waves,axis= 0)
        return dict(data = waves.T, label = self.label_to_index[range_index])
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        return self.generateData()


    def __len__(self):
        return self.data_length
