import pandas as pd
import numpy as np
import os
from ..core.flowRepresentation import PacketFlowRepressentation
from ..flowUtils.commons import normalizePacketRep
from tqdm import tqdm
from joblib import delayed, Parallel
import json
from typing import List


class BaseDataFrameProcessor:
    
    column_name_mapper = dict(
        up_bytes = "flowprint_upstream_byte_counts",
        down_bytes = "flowprint_downstream_byte_counts",
        up_packets = "flowprint_upstream_packet_counts",
        down_packets = "flowprint_downstream_packet_counts"
    )


    def __convertToFloatArrayFunction(self,x):
        return np.array(list(x),dtype= np.float32)



    def __init__(self,parquet_path):
        self.df = pd.read_parquet(parquet_path)
          # correcting the type and provider type to string if its in bytes
        self.df.loc[:,"type"] = self.df.type.apply(lambda x : x.decode("utf-8") if isinstance(x,bytes) else x)
        self.df.loc[:,"provider"] = self.df.provider.apply(lambda x : x.decode("utf-8") if isinstance(x,bytes) else x)
        self.df.loc[:,"sni"] = self.df.provider.apply(lambda x : x.decode("utf-8") if isinstance(x,bytes) else x)


        
        # adding bps features
        self.df["up_bps"] = (self.df.up_bytes*8)/ self.df.duration_sec
        self.df["down_bps"] = (self.df.down_bytes*8)/ self.df.duration_sec

        
        # converting flowprint arrays to floatarrays
        for _,column_name in BaseDataFrameProcessor.column_name_mapper.items():
            self.df.loc[:,column_name] = self.df[column_name].apply(self.__convertToFloatArrayFunction)
        
        

        # stripping and reindexing
        self.__filterBasedOnFirstAndLastNonZeroLengthDownBytes()
        self.df.reindex()
        

    def __filterBasedOnFirstAndLastNonZeroLengthDownBytes(self):

        def stripArray(row,col_name):
            return row[col_name][:,row["strip_indices"][0]:row["strip_indices"][1]  + 1]
        

        strip_indices = self.df[BaseDataFrameProcessor.column_name_mapper["down_bytes"]].apply(lambda x : BaseDataFrameProcessor.__getStripIndices(x.sum(axis = 0))).tolist()
        self.df["strip_indices"] = strip_indices
       

        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_bytes"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["up_bytes"]), axis = 1)
        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_bytes"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["down_bytes"]), axis = 1)
        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_packets"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["up_packets"]), axis = 1)
        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_packets"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["down_packets"]), axis = 1)
       
        self.df.drop(columns= ["strip_indices"], inplace= True)


    @staticmethod
    def __getStripIndices(arr):
        start_index = 0 
        end_index = len(arr) -1

        while start_index < len(arr) and arr[start_index] == 0:
            start_index += 1
        while end_index >= start_index and arr[end_index] == 0:
            end_index -= 1

        return start_index,end_index
    




class GamingDownloadDataFrameProcessor(BaseDataFrameProcessor):
    def __init__(self, parquet_path,gaming_download_down_bps_threshold = 1e6):
        super().__init__(parquet_path)
        #print(len(self.df[(self.df.type == "Gaming Download") & (self.df.down_bps < gaming_download_down_bps_threshold)]))
        #self.df.drop(self.df[(self.df.type == "Gaming Download") & (self.df.down_bps < gaming_download_down_bps_threshold)].index, inplace= True)
        self.df.drop(self.df[(self.df.type == "Gaming Download")].index, inplace= True)
        self.df.reindex()
        #self.df.rename(columns= {"Gaming Download" : "Download"}, inplace= True)



class SoftwareUpdateDataProcessor(BaseDataFrameProcessor):
    def __init__(self,parquet_path):
        super().__init__(parquet_path= parquet_path)
        print("initial software update length = {}".format(len(self.df)))
        self.df.loc[:,"type"] = self.df.type.apply(lambda x : "Download" if x == "Software Update" else x)
        self.df.drop(index= self.df[self.df.sni != "Apple iOSAppStore"].index, inplace= True)
        self.df.reindex()
        print("final software update length = {}".format(len(self.df)))
        self.generateUploadFromDownload()
        print("after adding uploads size = {}".format(len(self.df)))



    def generateUploadFromDownload(self):
        downloads_df = self.df[self.df.type == "Download"].copy(deep= True)

        downloads_df.rename(columns= {BaseDataFrameProcessor.column_name_mapper["up_bytes"] : BaseDataFrameProcessor.column_name_mapper["down_bytes"], BaseDataFrameProcessor.column_name_mapper["down_bytes"] : BaseDataFrameProcessor.column_name_mapper["up_bytes"]} , inplace= True)
        downloads_df.rename(columns= {BaseDataFrameProcessor.column_name_mapper["up_packets"] : BaseDataFrameProcessor.column_name_mapper["down_packets"], BaseDataFrameProcessor.column_name_mapper["down_packets"] : BaseDataFrameProcessor.column_name_mapper["up_packets"]} , inplace= True)
        
        downloads_df.rename(columns= {"up_bytes":"down_bytes", "down_bytes" : "up_bytes"}, inplace= True)
        downloads_df.rename(columns= {"up_packets":"down_packets", "down_packets" : "up_packets"}, inplace = True)
        downloads_df.loc[:,"type"] = "Upload"
        self.df = pd.concat([self.df,downloads_df], ignore_index = True)





class UTMobileNetProcessor:
    def __init__(self,base_path):
        self.csvs = []
        self.__getCSVPaths(csvs= self.csvs,path= base_path)


    def getProtocol(self,row):
        if not pd.isnull(row['tcp.len']):
            return 'TCP'
        elif not pd.isnull(row['udp.length']):
            return 'UDP'
        else:
            return 'Unknown'
    
    def getSrcPort(self,row):
        if not pd.isnull(row['tcp.len']):
            return row['tcp.srcport']
        elif not pd.isnull(row['udp.length']):
            return row['udp.srcport']
        else:
            return 'Unknown'
    
    def getDstPort(self,row):
        if not pd.isnull(row['tcp.len']):
            return row['tcp.dstport']
        elif not pd.isnull(row['udp.length']):
            return row['udp.dstport']
        else:
            return 'Unknown'


    def __getCSVPaths(self,csvs,path):
        
        for item in os.listdir(path):
            item_path = os.path.join(path,item)
            if os.path.isdir(item_path):
                self.__getCSVPaths(path= item_path,csvs= csvs)
            else:
                if item_path.endswith(".csv"):
                    csvs.append(item_path)

    
    def processData(self):
        flows = []

        
        csv_flows = Parallel(n_jobs=8)(delayed(self.processCSV)(csv_path) for csv_path in self.csvs)
        for csv_flow in csv_flows:
            flows.extend(csv_flow)
        """
        for csv_path in tqdm(self.csvs):
            try:
                csv_flows = self.processCSV(path= csv_path)
                flows.extend(csv_flows)
            except Exception as e:
                print(e)
        """

        return flows
    def processCSV(self,path):

        def processConnDf(conn_df):
            unique = list(set(conn_df["ip.src"].unique().tolist() + conn_df["ip.dst"].unique().tolist()))
            assert len(unique) <= 2,unique
            #mapping = dict()
            #for i in range(len(unique)):
            #    mapping[unique[i]] = i
            
        
            directions = conn_df["direction"].tolist()#conn_df["ip.src"].apply(lambda x : mapping[x]).tolist()
            lengths = conn_df["frame.len"].tolist()
            timestamps = conn_df["timestamp"].tolist()
            lengths,inter_arrival_times,directions = normalizePacketRep(lengths= lengths,timestamps= timestamps,directions= directions)
            file_name = os.path.basename(path).split("_")[0]

            return PacketFlowRepressentation(lengths= lengths,directions= directions,
                                            inter_arrival_times= inter_arrival_times,class_type= file_name)


        df = pd.read_csv(path,low_memory= False)
        df = df[df['ip.src'].notna()]
        df = df.apply(lambda row:self.__cleanUpDuplicate(row),axis=1)
        df = df[(df['ip.src']!='127.0.0.1') & (df['ip.dst']!='127.0.0.1')]
        df["timestamp"] = pd.to_datetime(df["frame.time"].apply(lambda x : x.replace("CDT","").replace("CST","").strip()),format = "mixed")

        flows = []
        conn_dfs = self.getConnDfs(df)
        flows = map(lambda x : processConnDf(x), conn_dfs)
        return flows


    def getConnDfs(self,df):
        # expects df to have timestamp column that is sortable, among other things
        df['protocal'] = df.apply(lambda row: self.getProtocol(row), axis=1)
        df['srcport'] = df.apply(lambda row: self.getSrcPort(row), axis=1)
        df['dstport'] = df.apply(lambda row: self.getDstPort(row), axis=1) 

        included = set()
        flow_dict = dict()
        conn_dfs = []
        flow_columns = ['ip.src', 'srcport', 'ip.dst', 'dstport', 'protocal']


        for flow, flow_df in df.groupby(by=flow_columns):
            if flow[0].split('.')[0] == '10':
                flow_df["direction"] = 0
            else:
                flow_df["direction"] = 1
            flow_dict[flow] = flow_df
        
        for key in flow_dict:
            if key in included:
                continue
            rev_key = (key[2],key[3],key[0],key[1],key[4])

            if rev_key in flow_dict:
                conn_df = pd.concat([flow_dict[key],flow_dict[rev_key]])
                conn_df.sort_values(by= "timestamp",inplace= True)

                if len(conn_df) >= 2:
                    conn_dfs.append(conn_df)
            
            included.add(rev_key)

        
        return conn_dfs


    def __cleanUpDuplicate(self,row):
        if len(row['ip.src'].split(','))>1:
            row['ip.src'] = row['ip.src'].split(',')[1]
        if len(row['ip.dst'].split(','))>1:
            row['ip.dst'] = row['ip.dst'].split(',')[1]
        return row
    




class MirageProcessor:


    label_to_app_mapping = {
    "com.pinterest" : "Pinterest",
    "com.facebook.katana" : "Facebook",
    "com.spotify.music" : "Spotify",
    "com.contextlogic.wish" : "Wish",
    "com.groupon" : "Groupon",
    "com.tripadvisor.tripadvisor" : "TripAdvisor",
    "com.dropbox.android" : "Dropbox",
    "com.trello" : "Trello",
    "com.viber.voip" : "Viber",
    "com.facebook.orca" : "Messenger",
    "com.twitter.android" : "Twitter",
    "com.google.android.youtube" : "Youtube",
    "de.motain.iliga" : "OneFootball",
    "com.accuweather.android" : "AccuWeather",
    "com.iconology.comics" : "Comics",
    "com.joelapenna.foursquared" : "FourSquare",
    "it.subito" : "Subito",
    "com.duolingo" : "Duolingo",
    "com.waze" : "Waze",
    "air.com.hypah.io.slither" : "Slither.io"
    }


    label_to_app_catagory_mapping = {
    "com.pinterest" : "Social",
    "com.facebook.katana" : "Social",
    "com.spotify.music" : "Music and Audio",
    "com.contextlogic.wish" : "Shopping",
    "com.groupon" : "Shopping",
    "com.tripadvisor.tripadvisor" : "Travel and Local",
    "com.dropbox.android" : "Productivity",
    "com.trello" : "Productivity",
    "com.viber.voip" : "Communication",
    "com.facebook.orca" : "Communication",
    "com.twitter.android" : "News and Magazines",
    "com.google.android.youtube" : "Video Players",
    "de.motain.iliga" : "Sports",
    "com.accuweather.android" : "Weather",
    "com.iconology.comics" : "Comics",
    "com.joelapenna.foursquared" : "Travel and Local",
    "it.subito" : "Lifestyle",
    "com.duolingo" : "Education",
    "com.waze" : "Maps & Navigation",
    "air.com.hypah.io.slither" : "Games"
    }

    
    def __init__(self,data_path):
        self.json_paths = []
        self.__getJSONPaths(path= data_path)


    def __getJSONPaths(self,path):
        if path.endswith(".json"):
            self.json_paths.append(path)
        elif os.path.isdir(path):
            for item in os.listdir(path):
                self.__getJSONPaths(path= os.path.join(path,item))

    
    def __correctLabels(self, packet_reps : List[PacketFlowRepressentation], mode):
        
        new_reps = []
        lookup_dict = None
        if mode == "app":
            lookup_dict = MirageProcessor.label_to_app_mapping
        elif mode == "app_catagory":
            lookup_dict = MirageProcessor.label_to_app_catagory_mapping


        for packet_rep in packet_reps:
            class_type = packet_rep.class_type

            if class_type not in lookup_dict:
                continue
            packet_rep.class_type = lookup_dict[class_type]

            new_reps.append(packet_rep)
        return new_reps
        

        


    def processSingleJSON(self,path):
        data = []
        with open(path, "r") as f:
            data = json.loads(f.read())
        keys = list(data.keys())

        packet_reps = []
        for conn, conn_data in data.items():
            packet_data = conn_data["packet_data"]
            
            directions = packet_data["packet_dir"]
            lengths = packet_data["L4_payload_bytes"]
            lengths = list(map(lambda x : x/1500, lengths))

            iat = np.array(packet_data["iat"])
            iat = np.log(1 + iat*1e6)/np.log(900000)  # converting to micro seconds and dividing by 9 seconds to normalize
            iat = iat.tolist()

            label = conn_data["flow_metadata"]["BF_label"]
            packet_rep = PacketFlowRepressentation(lengths= lengths, directions= directions, inter_arrival_times= iat, class_type= label)
            packet_reps.append(packet_rep)

        return packet_reps
    

    def getPacketReps(self, mode = "app"):
        packet_reps = []
        results = Parallel(n_jobs=10)(delayed(self.processSingleJSON)(path) for path in self.json_paths)
        for result in results:
            packet_reps.extend(result)

        return self.__correctLabels(packet_reps= packet_reps, mode= mode)

            
        


        