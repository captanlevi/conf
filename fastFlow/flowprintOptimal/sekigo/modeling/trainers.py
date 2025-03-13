import torch.nn as nn
from .loggers import Logger
import torch
from torch.utils.data import DataLoader
from ..flowUtils.flowDatasets import BaseFlowDataset
from torchsampler import ImbalancedDatasetSampler
from ..utils.evaluations import Evaluator
import copy

class NNClassificationTrainer:
    def __init__(self,classifier,device,logger : Logger):
        self.classifier = classifier.to(device)
        self.device = device
        self.logger = logger
        self.cross_entropy_loss =  nn.CrossEntropyLoss()

        self.evaluator = Evaluator(model= self.classifier,device= device)

        self.best = dict(
            model = copy.deepcopy(self.classifier),
            score = 0
        )

        self.logger.setMetricReportSteps(metric_name= "train_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_loss", step_size= 10)
        self.logger.setMetricReportSteps(metric_name= "test_accuracy", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_accuracy", step_size= 1)

    

    def trainStep(self,batch,classifier_optimizer):
        X,y = batch["data"].float().to(self.device), batch["label"].to(self.device)
        model_out = self.classifier(X)[0]
        classifier_optimizer.zero_grad()
        loss = self.cross_entropy_loss(model_out,y).mean()
        loss.backward()
        classifier_optimizer.step()
        self.logger.addMetric(metric_name= "train_loss", value= loss.cpu().item())

   

    def train(self,train_dataset,test_dataset,epochs,batch_size,lr,use_balanced_sampler = False):

        classifier_optimizer =  torch.optim.Adam(params= self.classifier.parameters(), lr= lr)
        step = 0
        if use_balanced_sampler == True:
            train_dataloader = DataLoader(
                                            train_dataset,
                                            sampler=ImbalancedDatasetSampler(train_dataset),
                                            batch_size=batch_size,
                                        )
        else:
            train_dataloader = DataLoader(dataset= train_dataset,shuffle= True,drop_last= True,batch_size= batch_size)
        for epoch in range(epochs):
            for batch in train_dataloader:
                self.trainStep(batch= batch,
                               classifier_optimizer= classifier_optimizer)
                
                if step%500 == 0:
                    metrices = self.evaluator.getMetrices(dataset= test_dataset)
                    test_f1  =  metrices["macro_f1"]

                    if test_f1 >= self.best["score"]:
                        self.best["score"] = test_f1
                        self.best["model"] = copy.deepcopy(self.classifier)


                    test_accuracy = metrices["accuracy"]
                    self.logger.addMetric("test_f1", test_f1)
                    self.logger.addMetric(metric_name= "test_accuracy", value= test_accuracy)
                if step%1000 == 0:
                    metrices = self.evaluator.getMetrices(dataset= train_dataset)
                    train_f1  =  metrices["macro_f1"]
                    train_accuracy = metrices["accuracy"]
                    self.logger.addMetric(metric_name= "train_accuracy", value= train_accuracy)
                    self.logger.addMetric("train_f1", train_f1)
                step += 1