from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_sequence, unpack_sequence


def evaluateModelOnDataSet(dataset ,model : nn.Module,device : str,calc_f1 = True):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            last_pred_scores = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(device), batch["label"].to(device)
                model_out_probs = F.softmax(model(batch_X)[-1],dim= -1)
                batch_predicted = torch.argmax(model_out_probs,dim= -1).cpu().numpy().tolist() # (BS,seq_len)
                batch_last_pred_scores = model_out_probs[:,-1].cpu().numpy().tolist()
                predictions.extend(batch_predicted)
                last_pred_scores.extend(batch_last_pred_scores)
                labels.extend(batch_y.cpu().numpy().tolist())


        model.train()
        last_pred_scores = np.array(last_pred_scores)
        if calc_f1:
            _,_,f1,_ = precision_recall_fscore_support(labels, predictions, average= "weighted",zero_division=0)
            return f1,last_pred_scores.mean()
        else:
            return np.array(predictions),last_pred_scores.mean()


def evaluateModelOnDataSetFeature(dataset ,feature_extractor : nn.Module,classifier : nn.Module,device : str):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        feature_extractor.eval()
        classifier.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(device), batch["label"].to(device)
                batch_predicted = torch.argmax(classifier(feature_extractor(batch_X)),dim= -1).cpu().numpy().tolist() # (BS,seq_len)
                predictions.extend(batch_predicted)
                labels.extend(batch_y.cpu().numpy().tolist())


        classifier.train()
        feature_extractor.train()
        _,_,f1,_ = precision_recall_fscore_support(labels, predictions, average= "weighted",zero_division=0)
        return f1





class Evaluator:
    def __init__(self,model,device):
        self.model = model.to(device)
        self.device = device


    
    def classificationScores(self,predicted,y_true,labels):
        predicted,y_true,labels = np.array(predicted),np.array(y_true), np.array(labels)
        _,_,micro_f1,_ = precision_recall_fscore_support(y_true = y_true, y_pred= predicted, average= "micro",zero_division=0)
        _,_,macro_f1,_ = precision_recall_fscore_support(y_true= y_true, y_pred= predicted, average= "macro",zero_division=0)
        per_class_f1 = f1_score(y_true=y_true, y_pred= predicted, average= None)
        
      
        accuracy = accuracy_score(y_true= y_true, y_pred= predicted)
        cm = confusion_matrix(y_true= y_true, y_pred= predicted, labels= labels)
        return dict(micro_f1 = micro_f1,macro_f1 = macro_f1, accuracy = accuracy, cm = cm, per_class_f1 = per_class_f1)
    

    def predictOnDataset(self,dataset):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        self.model.eval()
        with torch.no_grad():

            labels = []
            predictions = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                predicted = self.model(batch_X)[0] # (BS,num_class)
                predicted = torch.argmax(predicted,dim= -1).cpu().numpy()
                predictions.extend(predicted)
                labels.extend(batch_y.cpu().numpy().tolist())


        self.model.train()
        return predictions,labels
    

    def getMetrices(self,dataset):
        predictions,y_true = self.predictOnDataset(dataset= dataset)
        labels = list(range(0,len(dataset.label_to_index)))
        metrices =  self.classificationScores(predicted= predictions,y_true= y_true,labels= labels)
        return metrices

        


class EarlyEvaluation(Evaluator):
    def __init__(self,min_steps,device,model):
        super().__init__(model= model,device= device)
        self.min_steps = min_steps

    def __processSinglePrediction(self,prediction,num_classes,num_packets, return_slots, threshold):
        """
        predictions are of shape (seq_len)

        slots are the number of timesteps 
        For packet they are equal to packets used
        For timeslot they will be different
        """
        # min_steps - 1 as if the min steps is 5 then after proccessing the 5th timestep index will be 4 !!!!
        packets_used = 0
        slots = 0
        for time in range(self.min_steps -1,len(prediction)):
            packets_used += num_packets[time]
            slots += 1

            if threshold is not None and time > threshold:
                if return_slots == False:
                    return (-1,packets_used)
                else:
                    return (-1,packets_used,slots)
            if prediction[time] < num_classes:
                if return_slots == False:
                    return (prediction[time],packets_used)
                else:
                    return (prediction[time], packets_used, slots)
        
        if return_slots == False:
            return (-1,len(prediction))
        else:
            return (-1,len(prediction), slots)

    def collateFn(self,batch):   
        data = list(map(lambda x : torch.tensor(x["data"]).float(),batch ))
        label = list(map(lambda x : torch.tensor(x["label"]).float(),batch ))
        
        num_packets = list(map(lambda x : x["num_packets"], batch))
        data = pack_sequence(data,enforce_sorted= False)

        return dict(
            data = data, label = torch.tensor(label), num_packets = num_packets
        )

    def predictOnDataset(self,dataset, return_slots = False, threshold = None):
        dataloader = DataLoader(dataset= dataset, batch_size = 64,collate_fn= self.collateFn)
        self.model.eval()                                          
        with torch.no_grad():                                    
            labels = []
            predictions_time = []
            slots = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                num_packets = batch["num_packets"] # 2D list [[1,2,1,1], [4,5,3,6]]
                predicted = self.predictStep(batch_X = batch_X)  # (list of seq_len) , (list of last_pred )
                predicted = list(map(lambda x : x.cpu().numpy().tolist(), predicted))
                processed_predictions = [self.__processSinglePrediction(x,len(dataset.label_to_index),num_p,return_slots,threshold) for x,num_p in zip(predicted,num_packets)]
                

                predictions_time.extend(processed_predictions)
                labels.extend(batch_y.cpu().numpy().tolist())


        self.model.train()
        predictions_time = np.array(predictions_time)
        predictions, time = predictions_time[:,0], predictions_time[:,1]
        
        labels = np.array(labels)
        if return_slots:
            slots = predictions_time[:,2]
            return predictions,time,labels,slots
        return predictions,time,labels
    

    def predictStep(self,batch_X):
        """
        Here the return is of shape (BS,seq_len)
        """
        self.model.eval()
        with torch.no_grad():
            model_out = self.model.earlyClassificationForward(batch_X)[0]
        self.model.train()
        # model_out is a list( (BS,seq_len,num_classes + 1))
        return map(lambda x : torch.argmax(x,dim= -1), model_out)# returning list(seq_len)
        #return torch.argmax(model_out,dim= -1)
    



    def getMetrices(self, dataset, ood_dataset = None, threshold = None):
        metrices = dict()
        if dataset != None:
            predictions,time,y_true = self.predictOnDataset(dataset= dataset, threshold= threshold)
            labels = np.array(list(range(0,len(dataset.label_to_index))))

            incorrect_ood = (predictions == -1).sum()/ len(predictions)

            included = predictions != -1
            predictions = predictions[included]
            time = time[included]
            y_true = y_true[included]
            

            metrices = self.classificationScores(predicted= predictions, labels= labels, y_true= y_true)
            metrices["time"] =  time.mean()
            metrices["time_std"] = time.std()
            metrices["incorrect_ood"] = incorrect_ood

        if ood_dataset != None:
            predictions,time,_ = self.predictOnDataset(dataset= ood_dataset,threshold= threshold)
    
            ood_accuracy = (predictions == -1).sum()/len(predictions)
            ood_time = time.mean()

            metrices["ood_accuracy"] = ood_accuracy
            metrices["ood_time"] = ood_time

        return metrices



class EarlyEvaluationV2(EarlyEvaluation):
    def __init__(self, min_steps, device, model, decider):
        super().__init__(min_steps, device, model)
        self.decider = decider.to(self.device)


    def predictStep(self, batch_X):
        self.model.eval()
        with torch.no_grad():
            model_out, model_features = self.model.earlyClassificationForward(batch_X)
            decider_out = self.decider(model_features)
        self.model.train()
        return torch.argmax(model_out,dim= -1), torch.argmax(decider_out, dim = -1)
    

    def __processSinglePrediction(self,prediction,validities):
        """
        predictions are of shape (seq_len)
        validities are of shape (seq_len)
        """
        
        for time in range(self.min_steps,len(prediction)):
            if validities[time] == 1:
                return (prediction[time],time)
        
        return (-1,len(prediction))
        
    
    def predictOnDataset(self,dataset,enforce_prediction):
        dataloader = DataLoader(dataset= dataset, batch_size = 64)
        self.model.eval()
        with torch.no_grad():

            labels = []
            predictions_time = []
            for batch in dataloader:
                batch_X,batch_y = batch["data"].float().to(self.device), batch["label"].to(self.device)
                predicted, validities = self.predictStep(batch_X = batch_X) # (BS,seq_len)
                predicted,validities = predicted.cpu().numpy(), validities.cpu().numpy()
                processed_predictions = []
                for i in range(len(predicted)):
                    processed_predictions.append(self.__processSinglePrediction(prediction= predicted[i], validities= validities[i]))

                predictions_time.extend(processed_predictions)
                labels.extend(batch_y.cpu().numpy().tolist())


        self.model.train()
        predictions_time = np.array(predictions_time)
        predictions, time = predictions_time[:,0], predictions_time[:,1]

        return predictions,time,labels