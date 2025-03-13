import torch
import torch.nn as nn
from ...modeling.neuralNetworks import LSTMNetwork, LinearPredictor
from .datasets import MemoryDataset
from ...flowUtils.flowDatasets import BaseFlowDataset
from copy import deepcopy
from torch.utils.data import DataLoader
from ...modeling.loggers import Logger
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from ...utils.commons import augmentData
from ...utils.evaluations import EarlyEvaluation, EarlyEvaluationV2
from torch.utils.data import WeightedRandomSampler
import random
from collections import deque
from torch.nn.utils.rnn import unpack_sequence
from .pMemory import PMemory

class EarlyClassificationtrainer:
    def __init__(self,predictor : LSTMNetwork,train_dataset : BaseFlowDataset,memory_dataset : MemoryDataset,use_sampler : bool,
                 test_dataset : BaseFlowDataset,ood_dataset : BaseFlowDataset,logger : Logger,model_replacement_steps : int,device : str):
        
        self.device = device
        self.use_sampler = use_sampler
        self.predictor = predictor.to(device)
        self.lag_predictor = deepcopy(predictor).to(device)
        self.lag_predictor.eval()


        self.train_dataset = train_dataset
        self.memory_dataset = memory_dataset
        self.test_dataset = test_dataset
        self.ood_dataset = ood_dataset
        self.logger = logger

        self.best = dict(
            score = 0,
            model = deepcopy(self.predictor)
        )

        self.evaluator = EarlyEvaluation(min_steps= memory_dataset.min_length, device= device,model= self.predictor)

        self.mse_loss_function = nn.MSELoss(reduction= "none")
        self.model_replacement_steps = model_replacement_steps

        self.logger.setMetricReportSteps(metric_name= "test_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_test", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_train", step_size= 1)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction= "none")
        self.priority_sample_indices = []  # (priority,index)
        self.softmax = nn.Softmax(dim= -1)


        
        


    def getWeightedSampler(self,memory_dataset):
        """
        memory dataset has all actions at least once
        """
        actions = []

        for d in memory_dataset.memories:
            actions.append(d.action)
        actions = np.array(actions)

        unique_actions = np.unique(actions)
        weights = np.array([1/len(actions)]*len(actions))

        weights[actions == (len(unique_actions) - 1)] *= (len(unique_actions) -1)
        weights = weights/weights.sum()
        sampler = WeightedRandomSampler(weights= weights,num_samples= len(weights))
        return sampler
    
   


    def calcQLearning(self,batch,lam):
        state,next_state,action,reward,is_terminal = batch["state"].to(self.device), batch["next_state"].to(self.device),\
                                                    batch["action"].to(self.device), batch["reward"].to(self.device),batch["is_terminal"].to(self.device)
        
        #label, state_length, indices = batch["label"].to(self.device), batch["state_length"].to(self.device), batch["index"]

        predicted_values = self.predictor(state)[0]
        predicted_values_for_taken_action = torch.gather(input= predicted_values, dim= 1,index= action.unsqueeze(-1)).squeeze() # (BS)
        
        with torch.no_grad():
            next_state_max_actions_model = torch.argmax(self.predictor(next_state)[0],dim = -1,keepdim= True) # (BS,1)
            next_state_values_lag_model = self.lag_predictor(next_state)[0] # (BS,K+1)
            next_state_values_for_max_action = torch.gather(input= next_state_values_lag_model, dim= 1, index= next_state_max_actions_model) # (BS,1)
            next_state_values_for_max_action = next_state_values_for_max_action*(~(is_terminal.unsqueeze(-1)))
            target = reward + lam*(next_state_values_for_max_action.squeeze()) # (BS)

        return predicted_values_for_taken_action,target
   



    def trainStep(self,steps,batch : dict,lam : float,predictor_optimizer):
        """
        state and next state is (BS,num_classes)
        """
        #self.trainFullClassifier(batch_size= batch["action"].shape[0],predictor_optimizer= predictor_optimizer)

        predicted_values_for_taken_action,target = self.calcQLearning(batch= batch, lam= lam)


        weight = torch.ones_like(target).to(self.device)
        if "IS_weights" in batch:
            weight = batch["IS_weights"].to(self.device)
       
        q_loss = (self.mse_loss_function(target, predicted_values_for_taken_action)*weight).mean()
        loss = q_loss
        predictor_optimizer.zero_grad()
        loss.backward()
        predictor_optimizer.step()
        self.logger.addMetric(metric_name= "q_loss", value= q_loss.item())


        if steps%self.model_replacement_steps == 0:
            self.__refreshLagModel()
        
        with torch.no_grad():
            abs_errors = abs(predicted_values_for_taken_action - target).cpu().numpy()
        return abs_errors
        
    def __refreshLagModel(self):
        self.lag_predictor = deepcopy(self.predictor)
        self.lag_predictor.eval()
    


    def eval(self,dataset : BaseFlowDataset):
        metrices = self.evaluator.getMetrices(dataset= dataset,ood_dataset= None)
        return metrices["macro_f1"],metrices["time"],metrices["incorrect_ood"]
    

    def evalTrain(self):
        f1,average_time,incorrect_ood = self.eval(dataset= self.train_dataset)
        self.logger.addMetric(metric_name= "train_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "train_eval_time", value= average_time)
        self.logger.addMetric(metric_name= "incorrect_ood_train", value = incorrect_ood)

    def evalTest(self):
        f1,average_time,incorrect_ood = self.eval(dataset= self.test_dataset)

        if f1 >= self.best["score"]:
            print("updated best with f1 = {}".format(f1))
            self.best["score"] = f1
            self.best["model"] = deepcopy(self.predictor)
        
        self.logger.addMetric(metric_name= "test_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "test_eval_time", value= average_time)
        self.logger.addMetric(metric_name= "incorrect_ood_test", value= incorrect_ood)

    def evalOOD(self):
        metrices = self.evaluator.getMetrices(ood_dataset= self.ood_dataset, dataset= None)
        self.logger.addMetric(metric_name= "ood_eval", value= metrices["ood_accuracy"])
        self.logger.addMetric(metric_name= "ood_eval_time", value= metrices["ood_time"])
        

        

    def train(self,epochs : int,batch_size = 64,lr = .001,lam = .99):
        # TODO add batch_sampler
        """
        Can stress enough how important the shuffle == True is in the Dataloader
        """
        if self.use_sampler == False:
            train_dataloader = DataLoader(dataset= self.memory_dataset,collate_fn= MemoryDataset.collateFn,batch_size= batch_size,drop_last= True,shuffle= True, num_workers= 0)
        else:
            sampler = self.getWeightedSampler(memory_dataset= self.memory_dataset)
            train_dataloader = DataLoader(dataset= self.memory_dataset, collate_fn= MemoryDataset.collateFn, batch_size= batch_size,drop_last= True,sampler= sampler, num_workers= 0)
        predictor_optimizer = torch.optim.Adam(params= self.predictor.parameters(), lr = lr)
        steps = 0

        for epoch in range(epochs):
            for batch in train_dataloader:
                self.trainStep(steps = steps,batch= batch,lam= lam, predictor_optimizer= predictor_optimizer)
                steps += 1            

                if steps%1000 == 0:
                    self.evalTest()
                    if self.ood_dataset != None:
                        self.evalOOD()
                if steps%2000 == 0:
                    self.evalTrain()
                





class PEarlyClassificationTrainer(EarlyClassificationtrainer):
    def __init__(self, predictor: LSTMNetwork, train_dataset: BaseFlowDataset, p_memory: PMemory,
                  test_dataset: BaseFlowDataset, ood_dataset: BaseFlowDataset, logger: Logger, min_length : int,
                    model_replacement_steps: int, device: str):
        self.device = device
        self.predictor = predictor.to(device)
        self.lag_predictor = deepcopy(predictor).to(device)
        self.lag_predictor.eval()


        self.p_memory = p_memory
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.ood_dataset = ood_dataset
        self.logger = logger

        self.best = dict(
            score = 0,
            model = deepcopy(self.predictor)
        )

        self.evaluator = EarlyEvaluation(min_steps= min_length, device= device,model= self.predictor)

        self.mse_loss_function = nn.MSELoss(reduction= "none")
        self.model_replacement_steps = model_replacement_steps

        self.logger.setMetricReportSteps(metric_name= "test_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_test", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_train", step_size= 1)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction= "none")
        self.priority_sample_indices = []  # (priority,index)
        self.softmax = nn.Softmax(dim= -1)
    


    def train(self,iterations, lr = 3e-4, lam = .99):
        predictor_optimizer = torch.optim.Adam(params= self.predictor.parameters(), lr = lr)
        for step in range(1,iterations + 1):
            batch = self.p_memory.sample()
            abs_errors = self.trainStep(steps = step,batch= batch,lam= lam, predictor_optimizer= predictor_optimizer)
            self.p_memory.batch_update(tree_idx= batch["tree_indices"], abs_errors= abs_errors)     

            if step%1000 == 0:
                self.evalTest()
                if self.ood_dataset != None:
                    self.evalOOD()
            if step%2000 == 0:
                self.evalTrain()
                



















class EarlyClassificationTrainerV2:
    def __init__(self, decider ,predictor: LSTMNetwork, train_dataset: BaseFlowDataset, memory_dataset: MemoryDataset, hint_loss_alpha: float,
                q_loss_alpha: float, hint_loss_gap: float, test_dataset: BaseFlowDataset, ood_dataset: BaseFlowDataset,
                logger: Logger, model_replacement_steps: int, device: str):
        self.device = device
        self.predictor = predictor.to(device)
        self.decider = decider.to(self.device)
        self.lag_decider = deepcopy(decider).to(device)
        self.lag_decider.eval()

        self.hint_loss_alpha = hint_loss_alpha
        self.q_loss_alpha = q_loss_alpha
        self.hint_loss_gap = hint_loss_gap

        self.train_dataset = train_dataset
        self.memory_dataset = memory_dataset
        self.test_dataset = test_dataset
        self.ood_dataset = ood_dataset
        self.logger = logger

        self.best = dict(
            score = 0,
            predictor = deepcopy(self.predictor),
            decider = deepcopy(self.decider)
        )

        self.evaluator = EarlyEvaluationV2(min_steps= memory_dataset.min_length, device= device,model= self.predictor, decider= self.decider)
        self.mse_loss_function = nn.MSELoss(reduction= "none")
        self.model_replacement_steps = model_replacement_steps

        self.logger.setMetricReportSteps(metric_name= "test_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_f1", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "train_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "test_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "ood_eval_time", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_test", step_size= 1)
        self.logger.setMetricReportSteps(metric_name= "incorrect_ood_train", step_size= 1)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction= "none")
 

    def __refreshLagModel(self):
        self.lag_decider = deepcopy(self.decider)
        self.lag_decider.eval()
    


    def eval(self,dataset : BaseFlowDataset):
        metrices = self.evaluator.getMetrices(dataset= dataset,ood_dataset= None)
        return metrices["macro_f1"],metrices["time"],metrices["incorrect_ood"]
    

    def evalTrain(self):
        f1,average_time,incorrect_ood = self.eval(dataset= self.train_dataset)
        self.logger.addMetric(metric_name= "train_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "train_eval_time", value= average_time)
        self.logger.addMetric(metric_name= "incorrect_ood_train", value = incorrect_ood)

    def evalTest(self):
        f1,average_time,incorrect_ood = self.eval(dataset= self.test_dataset)

        if f1 >= self.best["score"]:
            self.best["score"] = f1
            self.best["model"] = deepcopy(self.predictor)
        
        self.logger.addMetric(metric_name= "test_eval_f1", value= f1)
        self.logger.addMetric(metric_name= "test_eval_time", value= average_time)
        self.logger.addMetric(metric_name= "incorrect_ood_test", value= incorrect_ood)

    def evalOOD(self):
        metrices = self.evaluator.getMetrices(ood_dataset= self.ood_dataset, dataset= None)
        self.logger.addMetric(metric_name= "ood_eval", value= metrices["ood_accuracy"])
        self.logger.addMetric(metric_name= "ood_eval_time", value= metrices["ood_time"])


    
    def trainStep(self, steps, batch: dict, lam: float, predictor_optimizer, decider_optimizer):
        state,next_state,action,reward,is_terminal = batch["state"].to(self.device), batch["next_state"].to(self.device),\
                                                    batch["action"].to(self.device), batch["reward"].to(self.device),batch["is_terminal"].to(self.device)
        
        label, state_length = batch["label"].to(self.device), batch["state_length"].to(self.device)

        predictor_state_output,predictor_state_features = self.predictor(state)
        predicted_values = self.decider(predictor_state_features)
        with torch.no_grad():
            next_state_max_actions_model = torch.argmax(self.decider(self.predictor(next_state)[1]),dim = -1,keepdim= True)
            next_state_values_lag_model = self.lag_decider(self.predictor(next_state)[1])
            next_state_values_for_max_action = torch.gather(input= next_state_values_lag_model, dim= 1, index= next_state_max_actions_model) # (BS,1)
            next_state_values_for_max_action = next_state_values_for_max_action*(~(is_terminal.unsqueeze(-1)))


            # calculate reward
            predicted_classification = torch.argmax(predictor_state_output)
            correct_classification = (predicted_classification == label)
            do_predict = (action == 1)
            reward[(correct_classification & do_predict) ] = 1
            reward[(~correct_classification & do_predict)] = -1
            reward[~do_predict] = -.1


            target = reward + lam*(next_state_values_for_max_action.squeeze()) # (BS)
        
        
        predicted_values_for_taken_action = torch.gather(input= predicted_values, dim= 1,index= action.unsqueeze(-1)).squeeze() # (BS)
        q_loss = self.mse_loss_function(target, predicted_values_for_taken_action).mean()
        cross_entropy_loss = self.cross_entropy_loss(predictor_state_output,label)

        loss = q_loss + cross_entropy_loss

        self.logger.addMetric(metric_name= "q_loss", value= q_loss.item())
        self.logger.addMetric(metric_name= "cross_entropy_loss", value= cross_entropy_loss.item())

        decider_optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        loss.backward()
        predictor_optimizer.step()
        decider_optimizer.step()


        if steps%self.model_replacement_steps == 0:
            self.hint_memory = []
            self.__refreshLagModel()
        

    def train(self,epochs : int,batch_size = 64,lr = .001,lam = .99):
        # TODO add batch_sampler
        """
        Can stress enough how important the shuffle == True is in the Dataloader
        """
       
        sampler = self.getWeightedSampler(memory_dataset= self.memory_dataset)
        train_dataloader = DataLoader(dataset= self.memory_dataset, collate_fn= self.memory_dataset.collateFn, batch_size= batch_size,drop_last= True,sampler= sampler)
        predictor_optimizer = torch.optim.Adam(params= self.predictor.parameters(), lr = lr)
        decider_optimizer = torch.optim.Adam(params= self.decider.parameters(), lr = lr)
        steps = 0

        for epoch in range(epochs):
            for batch in train_dataloader:
                self.trainStep(steps = steps,batch= batch,lam= lam, predictor_optimizer= predictor_optimizer, decider_optimizer= decider_optimizer)
                steps += 1            

                if steps%1000 == 0:
                    self.evalTest()
                    if self.ood_dataset != None:
                        self.evalOOD()
                if steps%2000 == 0:
                    self.evalTrain()





        