import os
import torch
import json
from ..modeling.neuralNetworks import CNNNetwork1D,LSTMNetwork,LSTMDuelingNetwork
from .evaluations import Evaluator,EarlyEvaluation
import pandas as pd


class Documenter:
    def __init__(self,train_dataset,test_dataset,ood_dataset,full_model,early_model,configs):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.ood_dataset = ood_dataset
        self.full_model = full_model
        self.early_model = early_model
        self.configs = configs
        

    @staticmethod
    def basePath():
        return os.path.join(os.getcwd(),"results") 

    @staticmethod
    def getPaths(name):
        dir_path = os.path.join(Documenter.basePath(),name)
        full_model_path = os.path.join(dir_path,"full_model.pth")
        early_model_path = os.path.join(dir_path, "early_model.pth")
        train_dataset_path = os.path.join(dir_path,"train_dataset.pt")
        test_dataset_path = os.path.join(dir_path, "test_dataset.pt")
        ood_dataset_path = os.path.join(dir_path, "ood_dataset.pt")
        configs_path = os.path.join(dir_path, "configs.json")


        return dict(
            dir_path = dir_path,ood_dataset_path = ood_dataset_path,
            full_model_path = full_model_path, early_model_path = early_model_path, train_dataset_path = train_dataset_path,
            test_dataset_path = test_dataset_path, configs_path = configs_path
        )

        
    def document(self,name):
        paths = Documenter.getPaths(name= name)
        dir_path = paths["dir_path"]
        if os.path.exists(dir_path) == False:
            os.mkdir(dir_path)
        

        torch.save(self.train_dataset,paths["train_dataset_path"])
        torch.save(self.test_dataset,paths["test_dataset_path"])
        if self.ood_dataset != None:
            torch.save(self.ood_dataset,paths["ood_dataset_path"])

        self.early_model.cpu()
        self.full_model.cpu()       
        torch.save(self.early_model.state_dict(),paths["early_model_path"])
        torch.save(self.full_model.state_dict(), paths["full_model_path"])


        if self.configs == None:
            return
        
        with open(paths["configs_path"], "w") as f:
            json.dump(self.configs,f)
    


    
    @staticmethod
    def load(name):
        paths = Documenter.getPaths(name= name)
        train_dataset, test_dataset = torch.load(paths["train_dataset_path"]), torch.load(paths["test_dataset_path"])
        ood_dataset = None
        if os.path.exists(paths["ood_dataset_path"]):
            ood_dataset = torch.load(paths["ood_dataset_path"])
        
        with open(paths["configs_path"], "r") as f:
            configs = json.load(f) 

        try:
            early_model = LSTMNetwork(**configs["early_model_kwargs"])
            early_model.load_state_dict(torch.load(paths["early_model_path"]))
        except Exception as e:
            early_model = LSTMDuelingNetwork(**configs["early_model_kwargs"])
            early_model.load_state_dict(torch.load(paths["early_model_path"]))
        
        try:
            full_model = LSTMNetwork(**configs["full_model_kwargs"])
            full_model.load_state_dict(torch.load(paths["full_model_path"]))
        except Exception as e:
            print("Loading dummy full model")
            full_model = LSTMDuelingNetwork(**configs["early_model_kwargs"])

        return Documenter(train_dataset= train_dataset,test_dataset= test_dataset,ood_dataset= ood_dataset,full_model= full_model, early_model= early_model, configs= configs)
        

    
    def __getDatasetIndexToLabel(self,dataset):
        index_to_label = dict()
        for label,index in dataset.label_to_index.items():
            index_to_label[index] = label
        return index_to_label

    def __getDatasetLabels(self,dataset):
        index_to_label = self.__getDatasetIndexToLabel(dataset)
        labels = list(map(lambda x : index_to_label[x["label"]], dataset))
        return labels

        
    def getScores(self,device):
        early_evaluator = EarlyEvaluation(min_steps= 5,device=device,model= self.early_model)
        full_evaluator = Evaluator(model= self.full_model, device= device)

        early_metrices = early_evaluator.getMetrices(dataset= self.test_dataset)
        full_metrices = full_evaluator.getMetrices(dataset= self.test_dataset)

        
        train_dataset_counts = pd.Series(self.__getDatasetLabels(dataset= self.train_dataset)).value_counts()
        test_dataset_counts = pd.Series(self.__getDatasetLabels(dataset= self.test_dataset)).value_counts()


        index_to_label = self.__getDatasetIndexToLabel(dataset= self.test_dataset)
        labels = list(map(lambda x : index_to_label[x], range(0,len(self.test_dataset.label_to_index))))

        


        return dict(
            early_metrices = early_metrices,
            full_metrices = full_metrices,
            train_dataset_counts = train_dataset_counts,
            test_dataset_counts = test_dataset_counts,
            labels = labels
        )
        