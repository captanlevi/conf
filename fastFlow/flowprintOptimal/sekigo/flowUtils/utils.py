from .commons import loadFlows
import os
from ..core.flowRepresentation import PacketFlowRepressentation, TimeslotRepresentation

def readTrainTestOODFlows(base_path, dataset_name):
    dir_path = os.path.join(base_path,dataset_name)
    print(dir_path)
    train_packet_flows = loadFlows(path= os.path.join(dir_path,"train_packet_flows.pkl"), cls= PacketFlowRepressentation)
    test_packet_flows = loadFlows(path= os.path.join(dir_path,"test_packet_flows.pkl"), cls= PacketFlowRepressentation)
    train_timeslot_flows = loadFlows(path= os.path.join(dir_path,"train_timeslot_flows.pkl"), cls= TimeslotRepresentation)
    test_timeslot_flows = loadFlows(path= os.path.join(dir_path,"test_timeslot_flows.pkl"), cls= TimeslotRepresentation)

    ood_packet_flows = []
    ood_timeslot_flows = []
    if os.path.exists(os.path.join(dir_path,"ood_packet_flows.pkl")):
        ood_packet_flows = loadFlows(path= os.path.join(dir_path,"ood_packet_flows.pkl"), cls= PacketFlowRepressentation)
        ood_timeslot_flows = loadFlows(path= os.path.join(dir_path,"ood_timeslot_flows.pkl"), cls= TimeslotRepresentation)

    return train_packet_flows,test_packet_flows,ood_packet_flows, train_timeslot_flows,test_timeslot_flows,ood_timeslot_flows