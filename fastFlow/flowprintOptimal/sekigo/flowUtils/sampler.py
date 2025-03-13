import random
import numpy as np
import copy
from typing import List
from ..core.flowRepresentation import FlowRepresentation
from .commons import getValidInvalidStartingPointsForSubFlowStart, minimizeOverlaps
from ..core.flowConfig import FlowConfig
import math
import pandas as pd
from sklearn.model_selection import train_test_split


class FixedLengthSampler:

    def __init__(self,flow_config : FlowConfig,required_length_in_seconds,ratio_of_median_to_sample,min_activity_for_start_point,sample_wise_train_ratio,temporal_train_ratio):
        """
            Required length is the actual length of the flow array, to be derived from grain and required_length_in_seconds
            The ratio_of_median_to_sample controls how aggresive the banancer is less is more aggresive
            Temporal train ratio is about splitting the start points into train and test after sorting
            sample_wise_train_ratio is about taking a sample and to directely put it for testing.
        """
        self.flow_config = flow_config
        self.required_length_in_seconds = required_length_in_seconds
        self.required_length_in_datapoints = math.ceil(self.required_length_in_seconds/self.flow_config.grain)
        self.ratio_of_median_to_sample = ratio_of_median_to_sample
        self.min_activity_for_start_point = min_activity_for_start_point
        self.sample_wise_train_ratio = sample_wise_train_ratio
        self.temporal_train_ratio = temporal_train_ratio


    def sampleAndCutToLength(self,data : List[FlowRepresentation]):
        class_to_chance_per_point = None
        
        class_to_chance_per_point = self.__generateSampleChancePerClass(flows= data)
        print(class_to_chance_per_point)

        def generateRequiredLengthFlowFromDataPoint(flow : FlowRepresentation,start_index: int) -> FlowRepresentation:
            up_packets = flow.up_packets[:,start_index:start_index+self.required_length_in_datapoints].copy()
            down_packets = flow.down_packets [:,start_index:start_index+ self.required_length_in_datapoints].copy()
            up_bytes = flow.up_bytes[:,start_index:start_index+ self.required_length_in_datapoints].copy()
            down_bytes =  flow.down_bytes[:,start_index:start_index+ self.required_length_in_datapoints].copy()
            new_flow = FlowRepresentation(up_bytes= up_bytes,down_bytes= down_bytes, up_packets= up_packets,down_packets= down_packets,flow_config= flow.flow_config, class_type= flow.class_type)
            return new_flow
    

        def generateFlowsBasedOnStartPoints(flow,start_points):
            generated = []
            for start_index in start_points:
                new_flow = generateRequiredLengthFlowFromDataPoint(flow= flow,start_index= start_index)
                generated.append(new_flow)
            return generated

        
        train_flows : List[FlowRepresentation] = []
        val_flows : List[FlowRepresentation] = []
        test_flows : List[FlowRepresentation] = []

        overlapping_train_points = 0
        for flow in data:
            data_point_length = flow.up_bytes.shape[1]

            if data_point_length < self.required_length_in_datapoints:
                continue
            else:
                start_points, _ = getValidInvalidStartingPointsForSubFlowStart(flow= flow, required_length= self.required_length_in_datapoints,min_activity= self.min_activity_for_start_point)
                if len(start_points) == 0:
                    continue
                

                number_of_cuts = math.ceil(class_to_chance_per_point[flow.class_type]*len(start_points))
                # now offsetting start points
                start_points = FixedLengthSampler.__offsetStartPoints(start_points= start_points,required_start_points= number_of_cuts)

                min_overlap_start_points = minimizeOverlaps(starting_points= start_points,requested_interval= self.required_length_in_datapoints,required_number_of_points= number_of_cuts)
                sampled_start_points = np.random.choice(a= np.array(min_overlap_start_points),size= number_of_cuts,replace= False).tolist()
                sampled_start_points.sort()  # very important to sort
                if random.random() > self.sample_wise_train_ratio:
                    # put all in the test
                    test_flows.extend(generateFlowsBasedOnStartPoints(flow= flow,start_points= sampled_start_points))
                    continue
                

                train_points_length = int(self.temporal_train_ratio*len(sampled_start_points))
                if train_points_length == 1:
                    train_start_points, test_start_points = sampled_start_points,[]
                else:
                    train_start_points, test_start_points = sampled_start_points[:train_points_length],sampled_start_points[train_points_length:]

                train_flows.extend(generateFlowsBasedOnStartPoints(flow = flow, start_points = train_start_points))
                test_flows.extend(generateFlowsBasedOnStartPoints(flow = flow, start_points = test_start_points))
                

                overlapping_train_points += FixedLengthSampler.__calculateOverlapingPoints(train_start_points= train_start_points, test_start_points= test_start_points,required_length= self.required_length_in_datapoints)
        print("overlapping points = {}".format(overlapping_train_points))
        return dict(train_flows = train_flows, test_flows = test_flows)
    

    def __generateSampleChancePerClass(self,flows : List[FlowRepresentation]):
        """
        This calculates the chance of sample given a starting point per class

        """

        # calculating the number of sample points per class
        class_to_valid_starting_points = dict()
        for flow in flows:
            if flow.class_type not in class_to_valid_starting_points:
                class_to_valid_starting_points[flow.class_type] = 0
            valid_starting_points_set, _ = getValidInvalidStartingPointsForSubFlowStart(flow= flow,required_length= self.required_length_in_datapoints,min_activity= self.min_activity_for_start_point)
            class_to_valid_starting_points[flow.class_type] += len(valid_starting_points_set)


        # getting the median of the number of starting points 
        values = list(class_to_valid_starting_points.values())
        values.sort()
        median = values[len(values)//2] if len(values)%2 == 1 else (values[len(values)//2] + values[len(values)//2 + 1])/2
        

        # calculating the points to sample by the median of the value
        points_to_sample = int(median*self.ratio_of_median_to_sample)
        if points_to_sample == 0:
            # incase median is 1
            points_to_sample = median


        # finally assigning each class sample a probablity to be sampled
        class_chance_per_sample = dict()
        for key in class_to_valid_starting_points:
            if class_to_valid_starting_points[key] <= points_to_sample:
                class_chance_per_sample[key] = 1
            else:
                if class_to_valid_starting_points[key] == 0:
                    print("warning class {} has zero points to sample".format(key))
                    class_chance_per_sample[key] = 1
                else:
                    class_chance_per_sample[key] = points_to_sample/class_to_valid_starting_points[key]

        return class_chance_per_sample




    @staticmethod
    def __calculateOverlapingPoints(train_start_points,test_start_points,required_length):
        if len(test_start_points) == 0 or len(train_start_points) == 0:
            return 0
        overlapping_train_points = 0

        for i in range(len(train_start_points) -1,-1,-1):
            if train_start_points[i] + required_length > test_start_points[0]:
                overlapping_train_points += 1
            else:
                break
        return overlapping_train_points
    
    @staticmethod
    def __offsetStartPoints(start_points,required_start_points):
        """
        The algorithm I use for removing windows can be biased based on the start point
        So this function changes the startpoint
        """
        if len(start_points) <= required_start_points:
            return start_points
        else:
            can_remove = len(start_points) - required_start_points
            offset = int(np.round((random.random()*can_remove)))

            return start_points[offset:]











class FixedLengthSimpleSampler:

    @staticmethod
    def sampleAndCutToLength(data : List[FlowRepresentation],flow_config : FlowConfig,required_length_in_seconds, train_ratio,
                             min_activity_for_start_point):
        """
        Required length is the actual length of the flow array, to be derived from grain and required_length_in_seconds
        The ratio_of_median_to_sample controls how aggresive the banancer is less is more aggresive
        """
        required_length = math.ceil(required_length_in_seconds/flow_config.grain)
       
        def generateRequiredLengthFlowFromDataPoint(flow : FlowRepresentation,start_index: int) -> FlowRepresentation:
            up_packets = flow.up_packets[:,start_index:start_index+required_length].copy()
            down_packets = flow.down_packets [:,start_index:start_index+required_length].copy()
            up_bytes = flow.up_bytes[:,start_index:start_index+required_length].copy()
            down_bytes =  flow.down_bytes[:,start_index:start_index+required_length].copy()

            new_flow = FlowRepresentation(up_bytes= up_bytes,down_bytes= down_bytes, up_packets= up_packets,down_packets= down_packets,flow_config= flow.flow_config, class_type= flow.class_type)
            return new_flow
        


        sampled_flows = []
        for flow in data:
            data_point_length = flow.up_bytes.shape[1]

            if data_point_length < required_length:
                continue
            else:
                start_points, _ = getValidInvalidStartingPointsForSubFlowStart(flow= flow, required_length= required_length,min_activity= min_activity_for_start_point)
                if len(start_points) == 0:
                    continue
            
                
                last_end_point = 0
                for start_point in start_points:
                    if start_point > last_end_point:
                        new_flow = generateRequiredLengthFlowFromDataPoint(flow= flow,start_index= start_point)
                        last_end_point = start_point + required_length - 1
                        sampled_flows.append(new_flow)
        
        labels = list(map(lambda x : x.class_type, sampled_flows))
        train_flows, test_flows, _, _ = train_test_split(sampled_flows, labels, test_size= 1 - train_ratio)

               
        return dict(train_flows = train_flows, test_flows = test_flows)
    

  