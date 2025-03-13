from ..utils.commons import downSampleArray
import numpy as np
from .flowConfig import FlowConfig

class FlowRepresentation:
    really_small_number = 1e-6
    bytes_division_factor = 8

    def __init__(self,up_bytes, down_bytes,up_packets,down_packets,class_type,flow_config : FlowConfig,**kwargs):
        """
        all arrays are 2D of shape (numbands,timestamps)
        """
        self.up_bytes = up_bytes
        self.down_bytes = down_bytes
        self.up_packets = up_packets
        self.down_packets = down_packets
        self.class_type = class_type
        self.flow_config = flow_config

        assert self.up_packets.shape[1] == self.down_packets.shape[1] == self.down_bytes.shape[1] == self.up_bytes.shape[1], "length of arrays must match"
        # adding packet length array
        self._addPacketLengths()

    
        for key, value in kwargs.items():
            setattr(self, key, value)


    def _addPacketLengths(self):
        self.up_packets_length = self.up_bytes/(self.up_packets + FlowRepresentation.really_small_number)
        self.down_packets_length = self.down_bytes/(self.down_packets + FlowRepresentation.really_small_number)



    def _downSampleFrequency(self,down_sample_factor):
        self.up_bytes = downSampleArray(self.up_bytes,factor= down_sample_factor)
        self.down_bytes = downSampleArray(self.down_bytes,factor= down_sample_factor)
        self.up_packets = downSampleArray(self.up_packets,factor= down_sample_factor)
        self.down_packets = downSampleArray(self.down_packets,factor= down_sample_factor)

    
    def _cutArrays(self,new_length):
        self.up_bytes = self.up_bytes[:, :new_length]
        self.down_bytes = self.down_bytes[:,:new_length]
        self.up_packets = self.up_packets[:,:new_length]
        self.down_packets = self.down_packets[:,:new_length]


    def ser(self):
        return dict(
            up_bytes = self.up_bytes.tolist(),
            down_bytes = self.down_bytes.tolist(),
            up_packets = self.up_packets.tolist(),
            down_packets = self.down_packets.tolist(),
            class_type = self.class_type,
            flow_config = self.flow_config.__dict__
        )
    
    @staticmethod
    def deSer(data):
        return FlowRepresentation(up_bytes= np.array(data["up_bytes"]), down_bytes= np.array(data["down_bytes"]), up_packets= np.array(data["up_packets"])
                                  , down_packets= np.array(data["down_packets"]), class_type= data["class_type"] if "class_type" in data else "__unknown",
                                  flow_config= FlowConfig(**data["flow_config"])
                                  )
    


    def isZeroFlow(self):
        if (self.up_bytes + self.down_bytes).sum() == 0:
            return True
        return False
    

    def matchConfig(self,other_config : FlowConfig):
        """
        TODO check and implement band difference as well
        """
        if other_config == self.flow_config:
            return

        if self.flow_config.band_thresholds != other_config.band_thresholds:
            assert False, "not implemented yet"
        
        if self.flow_config.grain != other_config.grain:
            # grains are different after upsampling also check for length mishmatch and packet length remake
            current_grain = self.flow_config.grain
            required_grain = other_config.grain

            assert required_grain > current_grain, "can only downsample ;( cause interpolation is not included in this package"
            assert (required_grain/current_grain)%1 == 0, "need the factor to be a whole number"
            factor = int(required_grain/current_grain)
    
            self._downSampleFrequency(down_sample_factor= factor)
            # now remaking packet lengths
            self._addPacketLengths() 

        self.flow_config = other_config

    def __len__(self):
        """
        Returns the length of the flow regardless of the grain
        """
        return self.up_packets.shape[1]
    




    def __getitem__(self, idx):
        return self.up_bytes[:,idx],self.down_bytes[:,idx],self.up_packets[:,idx], self.down_packets[:,idx]
    
    def getSubFlow(self,start_index,length):
        end_index = start_index + length
        up_bytes,down_bytes,up_packets,down_packets = self[start_index:end_index]
        return FlowRepresentation(up_bytes= up_bytes.copy(), down_bytes= down_bytes.copy(), up_packets= up_packets.copy(),
                                  down_packets= down_packets.copy(),class_type= self.class_type,flow_config= self.flow_config
                                  )
    





class PacketFlowRepressentation:
    def __init__(self,lengths,directions,inter_arrival_times,class_type,provider_type = None) -> None:
        assert len(lengths) == len(directions) == len(inter_arrival_times)
        self.lengths = lengths
        self.directions = directions
        self.inter_arrival_times = inter_arrival_times
        self.class_type = class_type
        self.provider_type = provider_type
    def __getitem__(self, idx):
        return self.lengths[idx],self.directions[idx],self.inter_arrival_times[idx]
    
    def getSubFlow(self,start_index,length):
        end_index = start_index + length
        lengths,directions,interarrival_times = self[start_index:end_index]
        return PacketFlowRepressentation(lengths= lengths.copy(),directions= directions.copy(),inter_arrival_times= interarrival_times.copy(), class_type= self.class_type)
    


    def ser(self):
        return dict(
            lengths = self.lengths,
            directions = self.directions,
            inter_arrival_times = self.inter_arrival_times,
            class_type = self.class_type,
            provider = self.provider_type
        )
    
    @staticmethod
    def deSer(dct):
        return PacketFlowRepressentation(lengths = dct["lengths"],
        directions = dct["directions"],
        inter_arrival_times = dct["inter_arrival_times"],
        class_type = dct["class_type"], provider_type= None if "provider_type" not in dct else dct["provider_type"]
        )

    
    def __len__(self):
        return len(self.lengths)
    


class TimeslotRepresentation:
    really_small_number = 1e-6

    def __init__(self,up_bytes, down_bytes,up_packets,down_packets,timeslots_since_last,class_type,flow_config : FlowConfig,**kwargs):
        """
        all arrays are 2D of shape (numbands,timestamps)

        timeslot_since_last is 1D array of shape (timestamps)
        """
        self.up_bytes = up_bytes
        self.down_bytes = down_bytes
        self.up_packets = up_packets
        self.down_packets = down_packets
        self.timeslots_since_last = timeslots_since_last
        self.class_type = class_type
        self.flow_config = flow_config

        assert self.up_packets.shape[1] == self.down_packets.shape[1] == self.down_bytes.shape[1] == self.up_bytes.shape[1], "length of arrays must match"
        assert self.up_packets.shape[1] == len(self.timeslots_since_last)
        # adding packet length array
        self._addPacketLengths()

    def _addPacketLengths(self):
        self.up_packets_length = self.up_bytes/(self.up_packets + FlowRepresentation.really_small_number)
        self.down_packets_length = self.down_bytes/(self.down_packets + FlowRepresentation.really_small_number)

    def __getitem__(self, idx):
        return self.up_bytes[:,idx],self.down_bytes[:,idx],self.up_packets[:,idx], self.down_packets[:,idx], self.timeslots_since_last[idx]
    
    def getSubFlow(self,start_index,length):
        end_index = start_index + length
        up_bytes,down_bytes,up_packets,down_packets, timeslots_since_last = self[start_index:end_index]
        return TimeslotRepresentation(up_bytes= up_bytes.copy(), down_bytes= down_bytes.copy(), up_packets= up_packets.copy(), timeslots_since_last= timeslots_since_last.copy(),
                                  down_packets= down_packets.copy(),class_type= self.class_type,flow_config= self.flow_config
                                  )
    
    def __len__(self):
        return self.up_packets.shape[1]
    

    def ser(self):
        return dict(
            up_bytes = self.up_bytes.tolist(),
            down_bytes = self.down_bytes.tolist(),
            up_packets = self.up_packets.tolist(),
            down_packets = self.down_packets.tolist(),
            timeslots_since_last = self.timeslots_since_last.tolist(),
            class_type = self.class_type,
            flow_config = self.flow_config.__dict__
        )
    
    @staticmethod
    def deSer(data):
        return TimeslotRepresentation(up_bytes= np.array(data["up_bytes"]), down_bytes= np.array(data["down_bytes"]), up_packets= np.array(data["up_packets"])
                                  , down_packets= np.array(data["down_packets"]), class_type= data["class_type"] if "class_type" in data else "__unknown",
                                  flow_config= FlowConfig(**data["flow_config"]), timeslots_since_last= np.array(data["timeslots_since_last"])
                                  )
