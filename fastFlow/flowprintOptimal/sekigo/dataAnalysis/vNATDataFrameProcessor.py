from tqdm import tqdm
from typing import List, Dict
import datetime
from ..core.flowRepresentation import FlowRepresentation,PacketFlowRepressentation
from ..core.flowConfig import FlowConfig
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ..flowUtils.commons import normalizePacketRep


class VNATDataFrameProcessor:
    classes_to_include = [
        "nonvpn_ssh",            
        "nonvpn_scp_long",      
        "nonvpn_scp",             
        "nonvpn_rsync",           
        "nonvpn_sftp",            
        "nonvpn_skype_chat",      
        "nonvpn_vimeo",          
        "nonvpn_youtube",          
        "nonvpn_voip",                           
        "nonvpn_netflix"         
        ]

    @staticmethod
    def convertToFlowRepresentations(flow_config : FlowConfig,data_frame_path : str = "data/VNAT/vnat_dataframe.h5"):
        df = pd.read_hdf(data_frame_path)

        flows : List[FlowRepresentation] = []
        for i in tqdm(range(len(df))):
            flows.append(VNATDataFrameProcessor.processVNATRowToGetFlow(row = df.iloc[i],flow_config= flow_config))        
        return flows


    @staticmethod
    def processFileNames(filename : str):
        filename = filename.replace("-","_")
        filename = filename.split("_")
        filename = filename[:-1]
        return "_".join(filename)

    @staticmethod
    def normalizeTimestamps(timestamps : List[datetime.datetime]) -> List[datetime.timedelta]:
        """
        Generic function to subtract firest timestamp from the entire list
        """
        timestamps.sort()
        first_timestamp = timestamps[0]
        timestamps = list(map(lambda x : x-first_timestamp, timestamps))
        return timestamps

    @staticmethod
    def aggregateByTimeBins(grain : datetime.timedelta,normalized_timestamps : List[datetime.timedelta],direction : List[int],packet_sizes : List[float],band_thresholds):
        def aggregate(start_index : int,end_index : int):
            up_bytes = 0
            down_bytes = 0
            up_packets, down_packets = 0,0

            for i in range(start_index,end_index+1):
                if direction[i] == 0:
                    down_bytes += packet_sizes[i]
                    down_packets += 1
                else:
                    up_bytes += packet_sizes[i]
                    up_packets += 1
            
            return dict(up_packets = up_packets, down_packets = down_packets, up_bytes = up_bytes,
                        down_bytes = down_bytes)
        
        def aggregateBasedOnBands(start_index : int,end_index : int):

            def getIndexBasedOnBand(value):
                """
                logic to get the band index based on value and the band thresholds
                """
                for i in range(len(band_thresholds)):
                    if value <= band_thresholds[i]:
                        return i

                return len(band_thresholds)

            up_bytes,down_bytes = [0]*(len(band_thresholds)+1),[0]*(len(band_thresholds)+1)
            up_packets, down_packets = [0]*(len(band_thresholds)+1),[0]*(len(band_thresholds)+1)

        
            for i in range(start_index,end_index+1):
                band_index = getIndexBasedOnBand(packet_sizes[i])
                if direction[i] == 0:
                    down_bytes[band_index] += packet_sizes[i]
                    down_packets[band_index] += 1
                else:
                    up_bytes[band_index] += packet_sizes[i]
                    up_packets[band_index] += 1
            
            return dict(up_packets = up_packets, down_packets = down_packets, up_bytes = up_bytes,
                        down_bytes = down_bytes)


        def generateUniformGrainArray():
            array = [normalized_timestamps[0]]
            while normalized_timestamps[-1] >= array[-1]:
                array.append(array[-1] + grain)
            
            return array

        unifiorm_grain_array = generateUniformGrainArray()
        # handle case when len(uniform_grain_array) <= 1 no need to handle as it cannot be possible see generateUNifirmGrainArray

        start_index = 0
        end_index = 0

        aggregated_result = dict()
        for i in range(len(unifiorm_grain_array) - 1):
            time_slot_low, time_slot_high = unifiorm_grain_array[i], unifiorm_grain_array[i+1]

            while end_index < len(normalized_timestamps) and normalized_timestamps[end_index] < time_slot_high:
                end_index += 1
            
            aggregated_result[(time_slot_low,time_slot_high)] = aggregateBasedOnBands(start_index= start_index,end_index= end_index -1)
            start_index = end_index
            


        return aggregated_result

    @staticmethod
    def convertAggregatedDataToFlowRepresantation(aggregated_results : Dict[tuple,Dict[str,float]],file_name ,flow_config : FlowConfig):
        
        up_bytes,down_bytes,up_packets,down_packets = [],[],[],[]

        keys = list(aggregated_results.keys())
        keys.sort()
        sorted_values = [aggregated_results[i] for i in keys]
        up_bytes,down_bytes,up_packets,down_packets = [],[],[],[]


        for values in sorted_values:
            up_packets.append(values["up_packets"])
            down_packets.append(values["down_packets"])
            up_bytes.append(values["up_bytes"])
            down_bytes.append(values["down_bytes"])
        
        up_bytes,down_bytes,up_packets,down_packets = np.array(up_bytes).T,np.array(down_bytes).T,np.array(up_packets).T,np.array(down_packets).T
        label = VNATDataFrameProcessor.processFileNames(file_name)
        return FlowRepresentation(up_packets= up_packets, down_packets= down_packets,up_bytes= up_bytes, down_bytes= down_bytes,flow_config= flow_config,class_type= label)

            


    @staticmethod
    def processVNATRowToGetFlow(row : pd.Series, flow_config : FlowConfig):
        timestamps = row["timestamps"]
        packet_sizes = row["sizes"]
        directions = row["directions"]


        packet_sizes = [x for _, x in sorted(zip(timestamps, packet_sizes), key=lambda pair: pair[0])]
        directions = [x for _, x in sorted(zip(timestamps, directions), key=lambda pair: pair[0])]
        timestamps.sort()
        grain = datetime.timedelta(seconds= flow_config.grain)
        timestamps = list(map(lambda x : datetime.datetime.fromtimestamp(x), timestamps))
        normalized_timestamps = VNATDataFrameProcessor.normalizeTimestamps(timestamps= timestamps)
        aggregated_data = VNATDataFrameProcessor.aggregateByTimeBins(grain= grain,normalized_timestamps= normalized_timestamps, direction= directions,packet_sizes= packet_sizes,band_thresholds= flow_config.band_thresholds)
        flow = VNATDataFrameProcessor.convertAggregatedDataToFlowRepresantation(aggregated_results= aggregated_data, flow_config= flow_config, file_name= row["file_names"])
        
        return flow
            
            
    @staticmethod
    def __getTopLevelClass(class_type):
        lc_class = class_type.lower()

        if "nonvpn" not in lc_class:
            return "_unknown"
        if "vimeo" in lc_class or "netflix" in lc_class or "youtube" in lc_class:
            return "streaming"
        elif  "zoiper" in lc_class:
            return "voip"
        elif "skype" in lc_class:
            return "chat"
        elif "ssh" in lc_class or "rdp" in lc_class:
            return "control"
        elif "sftp" in lc_class or "rsync" in lc_class or "scp" in lc_class:
            return "FT"
        
        else:
            return "_unknown"
        
    
    @staticmethod
    def convertLabelsToTopLevel(flows : List[FlowRepresentation]):
        for flow in flows:
            flow.class_type = VNATDataFrameProcessor.__getTopLevelClass(flow.class_type)
        flows = list(filter(lambda x : x.class_type != "_unknown",flows))
        return flows

    
    @staticmethod
    def getPacketFlows(data_frame_path : str = "data/VNAT/vnat_dataframe.h5"):
        def getPacketFlowFromRow(row):
            timestamps = row["timestamps"]
            timestamps = list(map(lambda x : datetime.datetime.fromtimestamp(x), timestamps))
            lengths = row["sizes"]
            directions = row["directions"]

            lengths = [x for _, x in sorted(zip(timestamps, lengths), key=lambda pair: pair[0])]
            directions = [x for _, x in sorted(zip(timestamps, directions), key=lambda pair: pair[0])]
            timestamps.sort()

            lengths,inter_arrival_times,directions = normalizePacketRep(lengths= lengths,directions= directions,timestamps= timestamps)
            class_type = VNATDataFrameProcessor.processFileNames(filename= row["file_names"])
            return PacketFlowRepressentation(lengths= lengths,directions= directions,inter_arrival_times= inter_arrival_times,class_type= class_type)
        

        
        df = pd.read_hdf(data_frame_path)
        packet_flow_reps = Parallel(n_jobs=10)(delayed(getPacketFlowFromRow)(df.iloc[i]) for i in range(len(df)))
        """
        for i in tqdm(range(len(df))):
            row = df.iloc[0]

            if len(row["sizes"]) >= 30:
                packet_flow_reps.append(getPacketFlowFromRow(row= row))
        """
        return packet_flow_reps

        
