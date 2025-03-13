from ..core.flowRepresentation import PacketFlowRepressentation, TimeslotRepresentation
from ..core.flowConfig import FlowConfig
from .commons import dropPacketFromPacketRep,ReTransmissionBasedDisorder
from .flowDatasets import PacketFlowDataset,DDQNActivityDataset
from .conversions import convertPacketRepToTimeslotRepEffecient



def getPacketDroppedPacketDataset(packet_dataset : PacketFlowDataset, max_drop_rate = .05):
    if packet_dataset == None:
        return None
    flows = packet_dataset.flows
    aug_flows = []
    for flow in flows:
        #aug_flows.append(dropPacketFromPacketRep(flow_rep= flow, max_drop_rate= max_drop_rate, min_length= 8))
        aug_flows.append(ReTransmissionBasedDisorder(flow_rep= flow,max_rate= max_drop_rate))
        
    aug_dataset = PacketFlowDataset(flows= aug_flows, label_to_index= packet_dataset.label_to_index, aug= packet_dataset.aug,
                                    fixed_length= packet_dataset.fixed_length,truncate_length= packet_dataset.truncate_length                                    
                                    )
    return aug_dataset


def getPacketDroppedTimeslotDataset(packet_dataset : PacketFlowDataset,label_to_index, flow_config : FlowConfig ,aug,max_drop_rate = .05):
    if packet_dataset == None:
        return None
    aug_dataset = getPacketDroppedPacketDataset(packet_dataset= packet_dataset, max_drop_rate= max_drop_rate)
    packet_flows = aug_dataset.flows

    timeslot_flows = []

    for packet_flow in packet_flows:
        timeslot_flow : TimeslotRepresentation = convertPacketRepToTimeslotRepEffecient(packet_flow_rep= packet_flow, grain= flow_config.grain, band_thresholds= flow_config.band_thresholds)
        if len(timeslot_flow) > 15:
            # this is needed in timeslot not packet rep
            timeslot_flow.getSubFlow(start_index= 0, length= 15)
        timeslot_flows.append(timeslot_flow)
    
    
    return DDQNActivityDataset(flows= timeslot_flows, label_to_index= label_to_index, aug= aug)