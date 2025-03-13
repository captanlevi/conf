from .flowRepresentation import FlowRepresentation
import numpy as np
from typing import List
from joblib import delayed, Parallel

class FeatureExtractor:
    """
    Faster vectorized feature extractor for single flow representation
    # TODO add a batch processor 
    Common convention for priority is 
    upload - bytes
    download - bytes
    upload - packets
    download - packets
    upload - packet length
    download - packet length

    all extracted features are per band so the output is 
    of shape (num_types_of_array,num_bands)
    num_types_of_array might vary with various features
    """
    really_small_number = 1e-8



    feature_to_index_map = dict(
            sparcity_u_0 = 0, sparcity_u_1 = 1, sparcity_u_2 = 2, sparcity_d_0 = 3, sparcity_d_1 = 4, sparcity_d_2 = 5,
            non_zero_mean_u_pl_0 = 6, non_zero_mean_u_pl_1 = 7, non_zero_mean_u_pl_2 = 8, non_zero_mean_d_pl_0 = 9, non_zero_mean_d_pl_1 = 10, non_zero_mean_d_pl_2 = 11, 
            non_zero_var_u_pl_0 = 12, non_zero_var_u_pl_1 = 13, non_zero_var_u_pl_2 = 14, non_zero_var_d_pl_0 = 15, non_zero_var_d_pl_1 = 16, non_zero_var_d_pl_2 = 17,
            change_count_u_0 = 18, change_count_u_1 = 19, change_count_u_2 = 20, change_count_d_0 = 21, change_count_d_1 = 22, change_count_d_2 = 23,
            fraction_per_band_u_b_0 = 24, fraction_per_band_u_b_1 = 25, fraction_per_band_u_b_2 = 26, fraction_per_band_d_b_0 = 27, fraction_per_band_d_b_1 = 28, fraction_per_band_d_b_2 = 29,
            fraction_per_band_u_p_0 = 30, fraction_per_band_u_p_1 = 31, fraction_per_band_u_p_2 = 32, fraction_per_band_d_p_0 = 33, fraction_per_band_d_p_1 = 34, fraction_per_band_d_p_2 = 35,
        )

    index_to_feature_map = dict((value,key) for key,value in feature_to_index_map.items())



    @staticmethod
    def fractionPerBandFeature(flow : FlowRepresentation):
        def calcFraction(array : np.ndarray):
            total_sum = array.sum(keepdims= True)
            band_wise_sum = array.sum(axis= 1, keepdims= True)
            band_fractions = (band_wise_sum/(total_sum + FeatureExtractor.really_small_number)).reshape(1,-1)
            
            return band_fractions
        
        upload_bytes_fraction = calcFraction(flow.up_bytes)
        download_bytes_fraction = calcFraction(flow.down_bytes)
        upload_packets_fraction = calcFraction(flow.up_packets)
        download_packets_fraction = calcFraction(flow.down_packets)
        feature = np.concatenate((upload_bytes_fraction,download_bytes_fraction,upload_packets_fraction,download_packets_fraction), axis= 0)
        return feature
            



    @staticmethod
    def changeCountFeature(flow : FlowRepresentation):
        """
        Calculates the number of times numbers in an array 
        switch from a zero to non zero value
        """
        def calcTransitions(array : np.ndarray):
            c1,c2 = array[:,:-1], array[:,1:]
            transitions = ((c1 == 0) & (c2 != 0)).sum(axis = 1) + ((c1 != 0) & (c2 == 0)).sum(axis = 1)
            return transitions.reshape(1,-1)
        
        upload_transitions = calcTransitions(flow.up_bytes)
        download_transitions = calcTransitions(flow.down_bytes)      
        feature = np.concatenate((upload_transitions,download_transitions), axis= 0)
        return feature
        
    
    @staticmethod
    def calculateSparcityFeature(flow : FlowRepresentation):

        def calculateSparcity(band_array : np.ndarray):
            sparcity = ((band_array == 0).sum(axis = 1))/band_array.shape[1]
            return sparcity.reshape(1,-1)
        
        upload_sparcity = calculateSparcity(flow.up_bytes)
        download_sparcity = calculateSparcity(flow.down_bytes)
        feature = np.concatenate((upload_sparcity,download_sparcity),axis= 0)
        return feature
    
    @staticmethod
    def calculateNonZeroMeanFeature(flow : FlowRepresentation):

        def calculateNonZeroMean(band_array : np.ndarray):
            non_zero_mean = np.ma.array(data= band_array, mask= (band_array == 0)).mean(axis= 1).data
            return non_zero_mean.reshape(1,-1)
        
        upload_packet_length_non_zero_mean = calculateNonZeroMean(flow.up_packets_length)
        download_packet_length_non_zero_mean = calculateNonZeroMean(flow.down_packets_length)
        feature = np.concatenate((upload_packet_length_non_zero_mean,download_packet_length_non_zero_mean), axis= 0)
        return feature



     
    @staticmethod
    def calculateNonZeroVarFeature(flow : FlowRepresentation):

        def calculateNonZeroVar(band_array : np.ndarray):
            non_zero_mean = np.ma.array(data= band_array, mask= (band_array == 0)).var(axis= 1).data
            return non_zero_mean.reshape(1,-1)
        
        upload_packet_length_non_zero_var = calculateNonZeroVar(flow.up_packets_length)
        download_packet_length_non_zero_var = calculateNonZeroVar(flow.down_packets_length)
        feature = np.concatenate((upload_packet_length_non_zero_var,download_packet_length_non_zero_var), axis= 0)
        return feature
    
    @staticmethod
    def extractFeaturesFromSingleFlow(flow : FlowRepresentation):

        # please do not change the sequence in this code. It is mapped to the feature_to_index_map
        feature = np.concatenate((FeatureExtractor.calculateSparcityFeature(flow), FeatureExtractor.calculateNonZeroMeanFeature(flow),
                                   FeatureExtractor.calculateNonZeroVarFeature(flow), FeatureExtractor.changeCountFeature(flow), FeatureExtractor.fractionPerBandFeature(flow)), axis=  0)

        return feature.reshape(-1)


    
    def extractFeaturesFromFlowList(flows : List[FlowRepresentation]):
        features = Parallel(n_jobs=min(8,len(flows)))(delayed(FeatureExtractor.extractFeaturesFromSingleFlow)(flow) for flow in flows)
        return features
    






class ActivityArrayFeatureExtractor:
    """
    Activity array is of shape (time_stamps,num_bands)
    """


    @staticmethod
    def changeCountFeature(activity_array : np.ndarray):
        """
        Calculates the number of times numbers in an array 
        switch from a zero to non zero value
        """
        def calcTransitions(array : np.ndarray):
            c1,c2 = array[:,:-1], array[:,1:]
            transitions = ((c1 == 0) & (c2 != 0)).sum(axis = 1) + ((c1 != 0) & (c2 == 0)).sum(axis = 1)
            return transitions.reshape(1,-1)
        
    
        feature = calcTransitions(activity_array.T)
        return feature
        
    
    @staticmethod
    def calculateSparcityFeature(activity_array : np.ndarray):

        def calculateSparcity(band_array : np.ndarray):
            sparcity = ((band_array == 0).sum(axis = 1))/band_array.shape[1]
            return sparcity.reshape(1,-1)
        
       
        feature = calculateSparcity(band_array= activity_array)
        return feature
    
    
    @staticmethod
    def extractFeaturesFromSingleFlow(activity_array : np.ndarray):

        # please do not change the sequence in this code. It is mapped to the feature_to_index_map
        feature = np.concatenate((ActivityArrayFeatureExtractor.calculateSparcityFeature(activity_array),
                                   ActivityArrayFeatureExtractor.changeCountFeature(activity_array)), axis=  0)

        return feature.reshape(-1)


    
    def extractFeaturesFromFlowList(activity_arrays: List[np.ndarray]):
        features = Parallel(n_jobs=min(8,len(activity_arrays)))(delayed(FeatureExtractor.extractFeaturesFromSingleFlow)(activity_array) for activity_array in activity_arrays)
        return features
    