from ..core.flowRepresentation import FlowRepresentation,PacketFlowRepressentation,TimeslotRepresentation
from typing import List
import json
import numpy as np
import datetime

def saveFlows(path,flows : List[FlowRepresentation]):
    with open(path,"w") as f:
        serilized_data = list(map(lambda x : json.dumps(x.ser()),flows))
        json.dump(serilized_data,f)

def loadFlows(path,cls = FlowRepresentation) -> List[FlowRepresentation]:
    with open(path,"r") as f:
        data = json.load(f)
        flows = list(map(lambda x : cls.deSer(json.loads(x)), data))
        
    return flows



def prefixConvolution(array, window_size):
    """Perform a convolution where the result is aligned with the first element of the window."""
    # Manually pad the end of the array to ensure the convolution result aligns as desired
    window = np.ones(int(window_size))
    pad_width = len(window) - 1
    padded_array = np.pad(array, (0, pad_width), mode='constant', constant_values=0)
    # Perform the convolution
    result = np.convolve(padded_array, window, 'valid')
    return result



def getValidInvalidStartingPointsForSubFlowStart(flow : FlowRepresentation,required_length,min_activity : int):
    """
    Valid points must have atleast some activity hapenning in the required length region.
    Activity is the number of packets in either direction
    """
    sum_array = (flow.up_packets +  flow.down_packets).sum(axis = 0)
    data_point_length = sum_array.shape[0]
    max_start_index = data_point_length - required_length
    convoluted = prefixConvolution(sum_array,required_length)
    valid_points = []
    invalid_points = []
    for i in range(max_start_index):
        if convoluted[i] >= min_activity:
            valid_points.append(i)
        else:
            invalid_points.append(i)
    
    return valid_points,invalid_points





def getActivityArrayFromFlow(flow : FlowRepresentation):
    """
    Takes a flow and returns a global normalized array
    Dividing the packetlength by the band_thresholds 

    returns array of shape ((num_thresholds)*2, seq_lens)
    """
    band_thresholds = flow.flow_config.band_thresholds[:]
    band_thresholds.append(1500)
    band_thresholds = np.array(band_thresholds).reshape(-1,1)

    ratio_array_byte = flow.down_bytes.sum(axis = 0,keepdims = True)/(flow.up_bytes + flow.down_bytes + 1e-8).sum(axis = 0,keepdims = True)
    ratio_array_packet = flow.down_packets.sum(axis = 0,keepdims = True)/(flow.up_packets + flow.down_packets + 1e-8).sum(axis = 0,keepdims = True)
    activity_array = np.concatenate([flow.up_packets_length/band_thresholds,flow.down_packets_length/band_thresholds,ratio_array_byte,ratio_array_packet])
    activity_array = np.clip(activity_array,a_max= 1, a_min= 0)
    # must transpose the array as flow has (numbands, timesteps)
    return activity_array.T



def getActivityArrayFromTimeslotRep(flow : TimeslotRepresentation):
    """
    Takes a flow and returns a global normalized array
    Dividing the packetlength by the band_thresholds 

    returns array of shape ((num_thresholds)*2, seq_lens)
    """
    band_thresholds = flow.flow_config.band_thresholds[:]
    band_thresholds.append(1500)
    band_thresholds = np.array(band_thresholds).reshape(-1,1)

    ratio_array_byte = flow.down_bytes.sum(axis = 0,keepdims = True)/(flow.up_bytes + flow.down_bytes + 1e-8).sum(axis = 0,keepdims = True)
    ratio_array_packet = flow.down_packets.sum(axis = 0,keepdims = True)/(flow.up_packets + flow.down_packets + 1e-8).sum(axis = 0,keepdims = True)
    timeslots_since_last = np.log(flow.timeslots_since_last + 1e-8).reshape(1,-1)/np.log(1000)

    activity_array = np.concatenate([flow.up_packets_length/band_thresholds,flow.down_packets_length/band_thresholds,ratio_array_byte,ratio_array_packet, timeslots_since_last])
    activity_array = np.clip(activity_array,a_max= 2, a_min= 0)
    # must transpose the array as flow has (numbands, timesteps)
    return activity_array.T






def maxNormalizeFlow(flow : FlowRepresentation):

        def normalizeArrayByBand(array : np.ndarray):
            # array is of shape (bands,timesteps), we divide timestamps by the max in each slot
            maxes = array.max(axis= -1,keepdims= True) + 1e-6
            array = array/maxes
            return array
    
        feature_array = np.concatenate((flow.up_packets,flow.down_packets,flow.up_bytes,flow.down_bytes),axis= 0)
        feature_array = normalizeArrayByBand(feature_array).T
        normalized_packetlengths = getActivityArrayFromFlow(flow= flow)
        feature_array = np.concatenate((feature_array,normalized_packetlengths), axis= 1)
        return feature_array


def minimizeOverlaps(starting_points,requested_interval,required_number_of_points):
    """
    Runs a binary search over max interval up untill requested interval
    so that we can get required_number_of_points while being as non overlapping as possible.
    """

    def pointsAfterNonOverlap(starting_points : List[int], interval_length) -> int:
        # starting points are sorted
        ans = 0
        last = starting_points[0] + interval_length
        included_starting_points = [starting_points[0]]
        for i in range(1,len(starting_points)):
            if starting_points[i] < last:
                ans += 1
            else:
                included_starting_points.append(starting_points[i])
                last = starting_points[i] + interval_length
        return included_starting_points
    
    if len(starting_points) == 0:
        return starting_points
    if len(starting_points) < required_number_of_points:
        # nonsense
        return starting_points
    
    
    starting_points.sort()
    mn_interval = 1
    mx_interval = requested_interval
    optimized_interval = None
    remaining_points_answer = []
    while mn_interval <= mx_interval:

        mid_interval = (mn_interval + mx_interval)//2
        remaining_points = pointsAfterNonOverlap(starting_points= starting_points,interval_length= mid_interval)

        if len(remaining_points) < required_number_of_points:
            mx_interval = mid_interval -1
        else:
            optimized_interval = mid_interval
            remaining_points_answer = remaining_points
            mn_interval = mid_interval + 1
    
    return remaining_points_answer




def getIATFromTimeStamps(timestamps):
    """
    timestamps is an array of datetime.datetime objects in sorted order
    or the same order but in np.timestamp
    """
    inter_arrival_times = []
    for i in range(len(timestamps)):
        if i == 0:
            inter_arrival_times.append(0)
        else:
            time_diff = (timestamps[i] - timestamps[i-1])
            if isinstance(time_diff, datetime.timedelta):
                inter_arrival_times.append(time_diff.total_seconds()*1e6)
            elif isinstance(time_diff, np.timedelta64):
                inter_arrival_times.append((time_diff/np.timedelta64(1,"s"))*1e6)
            assert inter_arrival_times[-1] >= 0, "{}".format(inter_arrival_times[-1])
            inter_arrival_times[-1] = np.log(inter_arrival_times[-1] + 1)/np.log(900000)
    return inter_arrival_times


def getTimeStampsFromIAT(inter_arrival_times):
    timestamps = [0]
    C = np.log(900000)
    base_time = datetime.datetime(year= 2023, month= 7, day= 31)
    for i in range(1,len(inter_arrival_times)):
        this_timestamp = timestamps[i-1] + np.exp(inter_arrival_times[i]*C) - 1
        timestamps.append(this_timestamp)

    for i in range(len(timestamps)):
        timestamps[i] = base_time + datetime.timedelta(microseconds= float(np.round(timestamps[i])))
    return timestamps




def normalizePacketRep(lengths,timestamps,directions):
    assert len(lengths) == len(timestamps) == len(directions)
    inter_arrival_times = getIATFromTimeStamps(timestamps= timestamps)

    for i in range(len(timestamps)):
        lengths[i] = lengths[i]/1500
        assert directions[i] in [0,1]

    return lengths,inter_arrival_times,directions



def dropPacketFromPacketRep(flow_rep : PacketFlowRepressentation,max_drop_rate,min_length):
    """
    Bhai copy karega data aug karne pehle to zindagi mai kush rahega

    """ 
    if len(flow_rep) <= min_length:
        return flow_rep


    drop_rate = min(max_drop_rate, (1 - min_length/len(flow_rep)))    
    drop_rate = np.random.random()*drop_rate



    lengths = np.array(flow_rep.lengths.copy())
    inter_arrival_times = np.array(flow_rep.inter_arrival_times.copy())
    directions = np.array(flow_rep.directions.copy())

    timestamps = np.array(getTimeStampsFromIAT(inter_arrival_times= inter_arrival_times))
    num_drop = int(drop_rate* len(flow_rep))
    keep_indices = np.random.choice(a= np.arange(len(flow_rep)),size= len(flow_rep) - num_drop, replace= False)

    # sorting the keep indices is very important
    keep_indices.sort()

    lengths = lengths[keep_indices].tolist()
    timestamps = timestamps[keep_indices].tolist()
    directions = directions[keep_indices].tolist()
    inter_arrival_times = getIATFromTimeStamps(timestamps)
    aug_rep = PacketFlowRepressentation(lengths= lengths, directions= directions, inter_arrival_times= inter_arrival_times,class_type= flow_rep.class_type)
    return aug_rep


def ReTransmissionBasedDisorder(flow_rep : PacketFlowRepressentation, max_rate, max_re_appear_gap = 5):
    """
    Here we simulate a TCP re transmission based packet disorder
    Random packets are dropped and later reappear after max_re_appear_gap

    TODO
    Have to handle the timestamps carefully, must increment the timestamp of all the packets 
    after one retransmitted arrives
    """
    re_order_rate = np.random.random()*max_rate
    num_re_order = int(np.ceil(len(flow_rep)*re_order_rate).item())
    if num_re_order == 0:
        return flow_rep
    re_order_indices = np.random.randint(0,len(flow_rep), num_re_order).tolist()
    re_order_counter_map = {index: 1 + int(max_re_appear_gap*np.random.random()) for index in re_order_indices}



    lengths = np.array(flow_rep.lengths.copy())
    inter_arrival_times = np.array(flow_rep.inter_arrival_times.copy())
    directions = np.array(flow_rep.directions.copy())
    timestamps = np.array(getTimeStampsFromIAT(inter_arrival_times= inter_arrival_times))

    new_lengths, new_timestamps , new_directions = [],[],[]


    for i in range(len(lengths)):
        if i in re_order_indices:
            continue
        else:
            new_lengths.append(lengths[i])
            new_directions.append(directions[i])
            new_timestamps.append(timestamps[i])

            for re_order_index in list(re_order_counter_map.keys()):
                counter = re_order_counter_map[re_order_index]
                if counter == 0:
                    new_lengths.append(lengths[re_order_index])
                    new_directions.append(directions[re_order_index])
                    new_timestamps.append(None)

                    re_order_counter_map.pop(re_order_index)
                else:
                    re_order_counter_map[re_order_index] -= 1
    
    if len(re_order_counter_map) != 0:
        left_over_indices = sorted(re_order_counter_map, key= re_order_counter_map.get)
        
        for index in left_over_indices:
            new_lengths.append(lengths[index])
            new_directions.append(directions[index])

    
    return PacketFlowRepressentation(lengths= new_lengths, directions= new_directions, inter_arrival_times= inter_arrival_times, class_type= flow_rep.class_type)



























    
