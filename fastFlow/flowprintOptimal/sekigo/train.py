from .flowUtils.utils import readTrainTestOODFlows
from .flowUtils.flowDatasets import PacketFlowDataset,DDQNActivityDataset
from .earlyClassification.DQL.core import Rewarder
from .earlyClassification.DQL.memoryFiller import MemoryFiller
from .earlyClassification.DQL.datasets import MemoryDataset
from .modeling.neuralNetworks import LSTMDuelingNetwork
from .modeling.loggers import Logger
import pandas as pd
from .earlyClassification.DQL.trainers import EarlyClassificationtrainer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import random_split

class Trainer:
    def __init__(self,configs,ood_support_percentage = None):
        # OOD support percentage in [0,1]
        self.configs = configs
        self.ddq_model = None
        self.test_dataset = None
        self.train_dataset = None
        self.ood_dataset = None
        self.ood_support_percentage = ood_support_percentage



    def getOODSupportDataset(self):
        # Define the total number of samples
        total_samples = len(self.ood_dataset)

        # Define the size of the support set
        ood_support_size = int(total_samples * self.ood_support_percentage)  # ood_support_percentage in range [0, 1]

        # Define the size of the remaining set
        remaining_size = total_samples - ood_support_size

        # Split the dataset
        ood_support_dataset, ood_remaining_dataset = random_split(self.ood_dataset, [ood_support_size, remaining_size])

        return ood_support_dataset, ood_remaining_dataset
    
    def start(self):
        print("start")
        train_dataset,test_dataset,ood_dataset = self.getDatasets(**self.getFlows())
        print("Got datasets")
        self.train(train_dataset= train_dataset, test_dataset= test_dataset, ood_dataset= ood_dataset)


    def train(self,train_dataset,test_dataset,ood_dataset):
        rewarder = Rewarder(**self.configs["rewarder_config"])

        ood_support_dataset = None
        if self.ood_support_percentage != None:
            ood_dataset,ood_support_dataset = self.getOODSupportDataset()

        memory_filler = MemoryFiller(dataset= train_dataset,rewarder= rewarder, min_length= self.configs["memory_fillter_config"]["min_length"], ood_dataset= ood_support_dataset,
                              max_length= rewarder.max_length,ood_config= self.configs["memory_fillter_config"]["ood_config"], use_balancer= self.configs["memory_fillter_config"]["use_balancer"]
                              )

        memory = memory_filler.processDataset()
        print(len(memory))
        memory_dataset = MemoryDataset(memories= memory,num_classes= len(train_dataset.label_to_index),
                               min_length= memory_filler.min_length,max_length= memory_filler.max_length)
        predictor = LSTMDuelingNetwork(**self.configs["early_model_kwargs"])
        logger = Logger(verbose= True)
        logger.default_step_size = 1000

        labels = []

        for m in memory_dataset:
            labels.append(m["label"])

        print(pd.Series(labels).value_counts())

        self.ddq_model = EarlyClassificationtrainer(predictor= predictor,train_dataset = train_dataset,test_dataset= test_dataset,memory_dataset= memory_dataset,
                                       ood_dataset= ood_dataset,use_sampler= self.configs["early_trainer_config"]["use_sampler"],
                                       logger= logger,device=device,model_replacement_steps= 500)
        
        self.ddq_model.train(epochs= 20,batch_size= 128,lr= .0003,lam= .99)  

    def getDatasets(self,train_packet_flows,test_packet_flows,ood_packet_flows, train_timeslot_flows,test_timeslot_flows,ood_timeslot_flows):
        truncate_length = self.configs["data_config"]["truncate_length"]
        if self.configs["data_config"]["data_type"] == "packet_representation":
            train_dataset = PacketFlowDataset(flows= train_packet_flows,label_to_index= None,aug= self.configs["dataset_config"]["aug"], truncate_length= truncate_length)
            test_dataset = PacketFlowDataset(flows= test_packet_flows,label_to_index= train_dataset.label_to_index, truncate_length= truncate_length)
            ood_dataset = PacketFlowDataset(flows= ood_packet_flows, label_to_index= None, truncate_length= truncate_length) if (ood_packet_flows != None and len(ood_packet_flows) != 0) else None
            
            train_fixed_length_dataset = PacketFlowDataset(flows= train_packet_flows, label_to_index= None, aug= self.configs["dataset_config"]["aug"], fixed_length= truncate_length)
            test_fixed_length_dataset = PacketFlowDataset(flows= test_packet_flows, label_to_index= train_fixed_length_dataset.label_to_index, aug= None, fixed_length= truncate_length)
        else:
            assert False, "Time interval not supported in this mode"
            train_dataset = DDQNActivityDataset(flows= train_timeslot_flows,label_to_index= None, aug= self.configs["dataset_config"]["aug"])
            test_dataset = DDQNActivityDataset(flows= test_timeslot_flows,label_to_index= train_dataset.label_to_index)
            ood_dataset = DDQNActivityDataset(flows= ood_timeslot_flows, label_to_index= None) if (ood_timeslot_flows != None and len(ood_timeslot_flows) != 0) else None

            
            train_fixed_length_dataset = DDQNActivityDataset(flows= train_timeslot_flows, label_to_index= None, aug= self.configs["dataset_config"]["aug"], fixed_length= 12)
            test_fixed_length_dataset = DDQNActivityDataset(flows= test_timeslot_flows, label_to_index= train_fixed_length_dataset.label_to_index, aug= None, fixed_length= 12)

            self.configs["early_model_kwargs"]["lstm_input_size"] = 5
            self.configs["full_model_kwargs"]["lstm_input_size"] = 5


        num_labels = len(train_dataset.label_to_index)
        self.configs["full_model_kwargs"]["output_dim"] = num_labels 
        self.configs["early_model_kwargs"]["output_dim"] = num_labels + 1
        self.configs["rewarder_config"]["num_labels"] = num_labels
        self.configs["rewarder_config"]["max_length"] = self.configs["common_config"]["max_timesteps"]


        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.ood_dataset = ood_dataset
        return train_dataset,test_dataset,ood_dataset

    

    def getFlows(self):
        if len(self.configs["data_config"]["ood_classes"]) == 0:
            base_dir_path = "data/ClassificationOnlyFlows"
        else:
            base_dir_path = "data/ClassificationOODFlows"

        train_packet_flows,test_packet_flows,ood_packet_flows, train_timeslot_flows,test_timeslot_flows,ood_timeslot_flows = \
        readTrainTestOODFlows(base_path= base_dir_path, dataset_name= self.configs["data_config"]["dataset_name"])

        return dict(train_packet_flows = train_packet_flows,
                    test_packet_flows= test_packet_flows,ood_packet_flows = ood_packet_flows,
                    train_timeslot_flows = train_timeslot_flows,test_timeslot_flows = test_timeslot_flows,
                    ood_timeslot_flows = ood_timeslot_flows)



