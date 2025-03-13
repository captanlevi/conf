from ..core.flowRepresentation import FlowRepresentation,PacketFlowRepressentation,TimeslotRepresentation
from ..core.flowConfig import FlowConfig
from ..dataAnalysis.vNATDataFrameProcessor import VNATDataFrameProcessor
import numpy as np
import datetime
from .commons import getTimeStampsFromIAT




def convertPacketRepToFlowRep(packet_flow_rep : PacketFlowRepressentation,grain = .001,band_thresholds = [1250]):
    inter_arrival_times = packet_flow_rep.inter_arrival_times
    timestamps = getTimeStampsFromIAT(inter_arrival_times= inter_arrival_times)
    lengths = [1500*x for x in packet_flow_rep.lengths]
    aggregated =  VNATDataFrameProcessor.aggregateByTimeBins(grain= datetime.timedelta(seconds= grain),normalized_timestamps= timestamps,direction= packet_flow_rep.directions,
                                               packet_sizes= lengths,band_thresholds= band_thresholds)

    keys = list(aggregated.keys())
    keys.sort()
    sorted_values = [aggregated[i] for i in keys]
    up_bytes,down_bytes,up_packets,down_packets = [],[],[],[]

    for values in sorted_values:
        up_packets.append(values["up_packets"])
        down_packets.append(values["down_packets"])
        up_bytes.append(values["up_bytes"])
        down_bytes.append(values["down_bytes"])


    up_bytes,down_bytes,up_packets,down_packets = np.array(up_bytes).T,np.array(down_bytes).T,np.array(up_packets).T,np.array(down_packets).T

    # multiply with 1500 as we divided for packet flow rep
    return FlowRepresentation(up_bytes= up_bytes, down_bytes= down_bytes, up_packets= up_packets,down_packets= down_packets,
                              class_type= packet_flow_rep.class_type,flow_config= FlowConfig(grain=grain, band_thresholds= band_thresholds)
                              )



def convertFlowRepToTimeSlotRep(flow : FlowRepresentation):
    def isZeroTimeSlot(flow, index):
        return np.array(flow[index]).sum() == 0

    data_indices = []
    for index in range(len(flow)):
        if isZeroTimeSlot(flow= flow, index= index) == False:
            data_indices.append(index)
    
    up_bytes, down_bytes, up_packets, down_packets, timeslots_since_last = [],[],[],[],[]

    for i,data_index in enumerate(data_indices):
        if i == 0:
            timeslots_since_last.append(1)
        else:
            timeslots_since_last.append(data_indices[i] - data_indices[i-1])
        
        data_up_bytes, data_down_bytes, data_up_packets, data_down_packets = flow[data_index]
        up_bytes.append(data_up_bytes)
        down_bytes.append(data_down_bytes)
        up_packets.append(data_up_packets)
        down_packets.append(data_down_packets)
    
    return TimeslotRepresentation(up_bytes= np.array(up_bytes).T, down_bytes= np.array(down_bytes).T,
                                up_packets= np.array(up_packets).T, down_packets= np.array(down_packets).T,
                                timeslots_since_last= np.array(timeslots_since_last), class_type= flow.class_type,
                                flow_config= flow.flow_config
                                )



def convertPacketRepToTimeslotRepEffecient(packet_flow_rep : PacketFlowRepressentation,grain = .001,band_thresholds = [1250]):
    def aggSlotPackets(packets : PacketFlowRepressentation):
        def getEmptyBand():
            return [0]*(len(band_thresholds) + 1)
        def getBandIndex(size):
            if len(band_thresholds) == 0:
                return 0

            for i in range(len(band_thresholds)):
                if size <= band_thresholds[i]:
                    return i
            return len(band_thresholds)
        
        up_bytes,down_bytes,up_packets,down_packets = [getEmptyBand() for _ in range(4)]

        for i in range(len(packets)):
            direction = packets.directions[i]
            size = packets.lengths[i]*1500
            band_index = getBandIndex(size= size)
            if direction == 0:
                up_bytes[band_index] += packets.lengths[i]*1500
                up_packets[band_index] += 1
            else:
                down_bytes[band_index] += packets.lengths[i]*1500
                down_packets[band_index] += 1
        
        return up_bytes,down_bytes,up_packets,down_packets


    inter_arrival_times = packet_flow_rep.inter_arrival_times
    timestamps = getTimeStampsFromIAT(inter_arrival_times= inter_arrival_times)
    start_timestamp = timestamps[0]
    timestamps = list(map(lambda x : (x - start_timestamp).total_seconds(), timestamps))


   
    current_slot = (0,grain)
    last_slot = 1
    current_iterator = (0,0,last_slot) # start,end,timeslots_since_last
    iterators = []
    for i in range(1, len(timestamps)):
        if timestamps[i] < current_slot[1]:
            current_iterator = (current_iterator[0], i, current_iterator[2])
        else:
            iterators.append(current_iterator)
            last_slot = 1
            while not(current_slot[0] <= timestamps[i] <= current_slot[1]):
                last_slot += 1
                current_slot = (current_slot[0] + grain, current_slot[1] + grain)
            current_iterator = (i,i,last_slot)
    iterators.append(current_iterator)
    
    up_bytes,down_bytes,up_packets,down_packets,timeslots_since_last = [],[],[],[],[]
    for iterator in iterators:
        start_,end_,timeslots_since_last_ = iterator
        u_b,d_b,u_p,d_p = aggSlotPackets(packets= packet_flow_rep.getSubFlow(start_index= start_, length= end_ - start_ + 1))
        up_bytes.append(u_b)
        down_bytes.append(d_b)
        up_packets.append(u_p)
        down_packets.append(d_p)
        timeslots_since_last.append(timeslots_since_last_)


    
    return TimeslotRepresentation(up_bytes= np.array(up_bytes).T, down_bytes= np.array(down_bytes).T, up_packets= np.array(up_packets).T,
                                  down_packets= np.array(down_packets).T, timeslots_since_last= np.array(timeslots_since_last), class_type= packet_flow_rep.class_type,
                                  flow_config= FlowConfig(grain= grain, band_thresholds= band_thresholds)
                                  )


def convertPacketRepToTimeslotRep(packet_flow_rep : PacketFlowRepressentation,grain = .001,band_thresholds = [1250] ):
    flow_rep = convertPacketRepToFlowRep(packet_flow_rep= packet_flow_rep, grain= grain, band_thresholds= band_thresholds)
    return convertFlowRepToTimeSlotRep(flow= flow_rep)





