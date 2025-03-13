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
from .dataGetter import assignUNibsClass, getFlowLength
import os
from .commons import saveFlows,loadFlows


def truncateLargeFlow(packet_rep : PacketFlowRepressentation,threshold_in_seconds):
    time_stamps = getTimeStampsFromIAT(packet_rep.inter_arrival_times)

    for i in range(1,len(time_stamps)):
        if (time_stamps[i] - time_stamps[0]).total_seconds() > threshold_in_seconds:
            return packet_rep.getSubFlow(0,i)

    return packet_rep




def getFlowReps(dataset_name, max_flow_length):
    """
    max flow length is in seconds
    """
    
    if dataset_name == "dummy":
        packet_flow_reps = loadFlows(path= "data/dummy/dummy.json", cls= PacketFlowRepressentation)
        keep_class = set(["Microsoft Teams","GoogleMeet", "WhatsAppVoice"])
        packet_flow_reps = list(filter(lambda x : x.class_type in keep_class, packet_flow_reps))


    else:
        assert False, "dataset name not recognized -- {}".format(dataset_name)


    print(len(packet_flow_reps))
    packet_flow_reps = list(map(lambda x : truncateLargeFlow(x,max_flow_length), packet_flow_reps))
    print(len(packet_flow_reps))
    
    timeslot_flow_reps  = Parallel(n_jobs=18)(delayed(convertPacketRepToTimeslotRepEffecient)(flow, .001, []) for flow in packet_flow_reps)

    return packet_flow_reps, timeslot_flow_reps



def filterBasedOnLength(reps,min_length):
    indices = []

    for i,rep in enumerate(reps):
        if len(rep) >= min_length:
            indices.append(i)
    
    return indices

def balanceFlows(flow_reps):
    count_dict =  dict(pd.Series(map(lambda x : x.class_type, flow_reps)).value_counts())
    print(count_dict)
    counts = sorted(list(count_dict.values()))
    #alpha = counts[0]/counts[-1]
    #keep_number = int(counts[-1]*alpha + counts[0]*(1- alpha))
    keep_number = counts[min(len(counts)//2 + 3, len(counts) - 1)]
    keep_number = 800
    print("keep_number = {}".format(keep_number))

    chosen_indices = []

    for i,flow_rep in enumerate(flow_reps):
        class_count = count_dict[flow_rep.class_type]

        if class_count <= keep_number:
            chosen_indices.append(i)
        else:
            drop_chance = (class_count - keep_number)/class_count
            if random.random() > drop_chance:
                chosen_indices.append(i)



    return chosen_indices



def chooseFlowsBasedOnIndices(reps,indices):
    new_reps = []

    for index in indices:
        new_reps.append(reps[index])
    return new_reps


def getIDOODIndices(reps, ood_classes):
    ood_indices = []
    id_indices = []
    for i, rep in enumerate(reps):
        if rep.class_type in ood_classes:
            ood_indices.append(i)
        else:
            id_indices.append(i)
    
    return id_indices,ood_indices
        

def getTrainTestIndices(reps,test_size):
   
    labels = list(map(lambda x : x.class_type,reps))
    indices = np.arange(len(reps))
    (
    _,
    _,
    train_labels,
    test_labels,
    indices_train,
    indices_test,
    ) = train_test_split(reps, labels, indices, test_size=test_size)
    print("---"*10)
    print("train class distrubation")
    print(pd.Series(train_labels).value_counts())
    print("test class distrubation")
    print(pd.Series(test_labels).value_counts())
    return indices_train, indices_test




def getTrainTestOODBalanced(dataset_name, test_size,ood_classes = [], packet_max_length = 15, timeslot_max_length= 15, max_flow_length_in_seconds = 90):
    packet_flows, timeslot_flows = getFlowReps(dataset_name, max_flow_length= max_flow_length_in_seconds)

    packet_flows = list(map(lambda x : x.getSubFlow(0,min(len(x), packet_max_length)), packet_flows))
    timeslot_flows = list(map(lambda x : x.getSubFlow(0,min(len(x), timeslot_max_length)), timeslot_flows))

    indices = filterBasedOnLength(reps= timeslot_flows,min_length= 1)
    filt_packet_flows, filt_timeslot_flows = chooseFlowsBasedOnIndices(packet_flows,indices), chooseFlowsBasedOnIndices(timeslot_flows,indices)

    
    indices = balanceFlows(flow_reps= filt_packet_flows)
    filt_packet_flows, filt_timeslot_flows = chooseFlowsBasedOnIndices(filt_packet_flows,indices), chooseFlowsBasedOnIndices(filt_timeslot_flows,indices)


    id_indices, ood_indices = getIDOODIndices(filt_packet_flows,ood_classes)

    ood_packet_flows, ood_timeslot_flows = chooseFlowsBasedOnIndices(filt_packet_flows,ood_indices), chooseFlowsBasedOnIndices(filt_timeslot_flows,ood_indices)
    filt_packet_flows, filt_timeslot_flows = chooseFlowsBasedOnIndices(filt_packet_flows,id_indices), chooseFlowsBasedOnIndices(filt_timeslot_flows,id_indices)
    
    
    train_indices,test_indices = getTrainTestIndices(filt_packet_flows,test_size)
    train_packet_flows , train_timeslot_flows = chooseFlowsBasedOnIndices(filt_packet_flows,train_indices), chooseFlowsBasedOnIndices(filt_timeslot_flows,train_indices)
    test_packet_flows, test_timeslot_flows = chooseFlowsBasedOnIndices(filt_packet_flows,test_indices), chooseFlowsBasedOnIndices(filt_timeslot_flows,test_indices)


    return train_packet_flows,test_packet_flows,ood_packet_flows, train_timeslot_flows,test_timeslot_flows,ood_timeslot_flows



    

    

def saveTrainTestOODFlows(base_path,dataset_name,train_packet_flows,test_packet_flows,ood_packet_flows, train_timeslot_flows,test_timeslot_flows,ood_timeslot_flows):
    dir_path = os.path.join(base_path,dataset_name)
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

    saveFlows(path= os.path.join(dir_path, "train_packet_flows.pkl"),flows = train_packet_flows)
    saveFlows(path= os.path.join(dir_path, "test_packet_flows.pkl"),flows = test_packet_flows)
    saveFlows(path= os.path.join(dir_path, "train_timeslot_flows.pkl"),flows = train_timeslot_flows)
    saveFlows(path= os.path.join(dir_path, "test_timeslot_flows.pkl"),flows = test_timeslot_flows)

    if len(ood_packet_flows) > 0:
        saveFlows(path= os.path.join(dir_path, "ood_packet_flows.pkl"),flows = ood_packet_flows)
        saveFlows(path= os.path.join(dir_path, "ood_timeslot_flows.pkl"),flows = ood_timeslot_flows)
    
    

def readTrainTestOODFlows(base_path, dataset_name):
    dir_path = os.path.join(base_path,dataset_name)

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


