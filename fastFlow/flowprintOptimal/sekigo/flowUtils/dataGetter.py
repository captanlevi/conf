import pandas as pd
from .commons import loadFlows
from ..core.flowRepresentation import PacketFlowRepressentation, TimeslotRepresentation
from ..dataAnalysis.vNATDataFrameProcessor import VNATDataFrameProcessor
from sklearn.model_selection import train_test_split
from typing import List, Dict
import random
import numpy as np
import datetime
from .commons import getTimeStampsFromIAT
from ..flowUtils.conversions import convertPacketRepToTimeslotRepEffecient
from joblib import Parallel, delayed



def getFlowLength(data_point):
        time_stamps = getTimeStampsFromIAT(data_point.inter_arrival_times)
        flow_length = (time_stamps[-1] - time_stamps[0]).total_seconds()
        return flow_length




def balanceFlows(flow_reps, return_indices = False):
    count_dict =  dict(pd.Series(map(lambda x : x.class_type, flow_reps)).value_counts())
    counts = sorted(list(count_dict.values()))
    alpha = counts[0]/counts[-1]
    keep_number = int(counts[-1]*alpha + counts[0]*(1- alpha))
    print("keep_number = {}".format(keep_number))

    balanced_flow_reps = []
    chosen_indices = []

    for i,flow_rep in enumerate(flow_reps):
        class_count = count_dict[flow_rep.class_type]

        if class_count <= keep_number:
            balanced_flow_reps.append(flow_rep)
            chosen_indices.append(i)
        else:
            drop_chance = (class_count - keep_number)/class_count
            if random.random() > drop_chance:
                balanced_flow_reps.append(flow_rep)
                chosen_indices.append(i)

    if return_indices == False:
        return balanced_flow_reps
    else:
        return balanced_flow_reps,chosen_indices


def samplePoints(flow_reps : List[PacketFlowRepressentation| TimeslotRepresentation],min_gap,max_gap,length):
    sampled_flow_reps = []
    for packet_flow_rep in flow_reps:
        start_index = 0

        while start_index + length < len(packet_flow_rep.lengths) + 1:
            sampled_flow_reps.append(packet_flow_rep.getSubFlow(start_index= start_index,length= length))
            start_index = start_index + length + min_gap + int((max_gap - min_gap)*random.random())

    return sampled_flow_reps





def assignUNibsClass(class_type):
    class_type = class_type.lower().strip()
    if class_type in ["amule","transmission", "bittorrent.exe"]:
        return "P2P"
    elif class_type in ["mail","thunderbird.exe"]:
        return "MAIL"
    elif class_type in ["skype", "skype"]:
        return "Skype"
    elif class_type in ["safari", "firefox-bin", "opera","safari webpage p", "safari webpage"]:
        return "BROWSERS"
    else:
        return "OTHER"
    







def getTrainTestOOD(dataset_name,min_timesteps, max_timesteps,test_size,data_type = "packet_representation",ood_classes = None,subsampleConfig : Dict = None, do_balance = False,max_flow_length = None):
    """
    subSampleConfig if provided has 
    {
    "min_gap" , "max_gap"
    }
    """
    
    if dataset_name == "unibs":
        if data_type == "packet_representation":
            flow_reps = loadFlows(path= "data/unibs/unibsPacketRep.json", cls= PacketFlowRepressentation)
        else:
            flow_reps = loadFlows(path= "data/unibs/unibsTimeslotRep.json", cls= TimeslotRepresentation)
        for flow_rep in flow_reps:
            flow_rep.class_type = assignUNibsClass(class_type= flow_rep.class_type)
        
    elif dataset_name == "VNAT":
        if data_type == "packet_representation":
            flow_reps = loadFlows(path= "data/VNAT/flowStore/vnatPacketRep.json", cls= PacketFlowRepressentation)#VNATDataFrameProcessor.getPacketFlows()
        else:
            flow_reps = loadFlows(path= "data/VNAT/flowStore/vnatTimeslotRep.json", cls= TimeslotRepresentation)
        flow_reps = VNATDataFrameProcessor.convertLabelsToTopLevel(flows= flow_reps)
    elif dataset_name == "UTMobileNet2021":
        if data_type == "packet_representation":
            flows = loadFlows(path= "data/UTMobileNet2021/mobilenetPacketRep.json", cls= PacketFlowRepressentation)
        else:
            flows = loadFlows(path= "data/UTMobileNet2021/mobilenetTimeslotRep.json", cls= TimeslotRepresentation)
        keep_class = set(["facebook","gmail", "google-drive", "google-maps","hangout","instagram","messenger","netflix", "pinterest", "reddit", "spotify","twitter", "youtube"])
        flow_reps = flows
        flow_reps = list(filter(lambda x : x.class_type in keep_class, flow_reps))
    elif dataset_name == "mirage":
        if data_type == "packet_representation":
            flow_reps = loadFlows(path= "data/MIRAGE-2019/miragePacketRepApp.json", cls= PacketFlowRepressentation)
        else:
            flow_reps = loadFlows(path= "data/MIRAGE-2019/mirageTimeslotRepApp.json", cls= TimeslotRepresentation)

    else:
        assert False, "dataset name not recognized -- {}".format(dataset_name)
    


    print("full class distrubation")
    print(pd.Series(map(lambda x : x.class_type,flow_reps)).value_counts())

    # filtering flows with at least packet_limit packets in it
    flow_reps = list(filter(lambda x : len(x) >= min_timesteps, flow_reps))
    if subsampleConfig == None:
        print("using no sampling")
        flow_reps = list(map(lambda x : x.getSubFlow(0,min(len(x), max_timesteps)), flow_reps))

    else:
        flow_reps = list(filter(lambda x : len(x) >= max_timesteps, flow_reps))
        print("using subsampling with {}".format(subsampleConfig))
        flow_reps = samplePoints(flow_reps= flow_reps, length= max_timesteps, min_gap= subsampleConfig["min_gap"], max_gap= subsampleConfig["max_gap"])
     

    print("hererererer")
    print(flow_reps[0].uid)


    if max_flow_length != None and data_type == "packet_representation":
        print("filtering max_flow_length = {}".format(max_flow_length))
        flow_reps = list(filter(lambda x : getFlowLength(x) <= max_flow_length, flow_reps))


    print("hererererer")
    print(flow_reps[0].uid)
    if do_balance == True:
        print("balancing")
        flow_reps = balanceFlows(flow_reps= flow_reps)
    print("post num packet filter class distrubation")
    print(pd.Series(map(lambda x : x.class_type,flow_reps)).value_counts())


    if ood_classes != None:
        id_flow_reps = list(filter(lambda x : x.class_type not in  ood_classes, flow_reps))
        ood_flow_reps = list(filter(lambda x : x.class_type in ood_classes, flow_reps))
    else:
        id_flow_reps = flow_reps
        ood_flow_reps = None
    
    labels = list(map(lambda x : x.class_type,id_flow_reps))
    train_flows,test_flows,train_labels,test_labels = train_test_split(id_flow_reps,labels,test_size= test_size)
    print("---"*10)
    print("train class distrubation")
    print(pd.Series(train_labels).value_counts())
    print("test class distrubation")
    print(pd.Series(test_labels).value_counts())

    return train_flows,test_flows,ood_flow_reps
