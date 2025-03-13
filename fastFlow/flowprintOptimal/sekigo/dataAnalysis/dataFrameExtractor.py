import copy
import numpy as np
import pandas as pd
from ..core.flowRepresentation import FlowRepresentation
from ..core.flowConfig import FlowConfig
from joblib import Parallel, delayed
from .dataFrameProcessor import BaseDataFrameProcessor
from typing import List




class DataFrameExtractor:

    column_name_mapper = dict(
        up_bytes = "flowprint_upstream_byte_counts",
        down_bytes = "flowprint_downstream_byte_counts",
        up_packets = "flowprint_upstream_packet_counts",
        down_packets = "flowprint_downstream_packet_counts"
    )

    @staticmethod
    def mergeDataProcessors(data_frame_processors : List[BaseDataFrameProcessor]):
        # Check if both DataFrames have the same columns and in the same order
        df_list = list(map(lambda x : x.df, data_frame_processors))
        if all(df_list[0].columns.equals(df.columns) for df in df_list):
            # Concatenate DataFrames
            merged_df = pd.concat(df_list, ignore_index=True)
            return merged_df
        else:
            assert False, "DataFrames do not have the same columns"



    @staticmethod
    def getData(data_frame_processors: List[BaseDataFrameProcessor],needed_flow_config : FlowConfig):
        df = DataFrameExtractor.mergeDataProcessors(data_frame_processors= data_frame_processors)

        flow_config = FlowConfig(grain= .5,band_thresholds= [1250])


        data = []
        for i in range(len(df)):
            row = df.iloc[i]
            if row[DataFrameExtractor.column_name_mapper["up_bytes"]].shape[1] < 5:
                continue
            row_flow_rep = FlowRepresentation(up_bytes = row[DataFrameExtractor.column_name_mapper["up_bytes"]][1:], down_bytes = row[DataFrameExtractor.column_name_mapper["down_bytes"]][1:],
                            up_packets = row[DataFrameExtractor.column_name_mapper["up_packets"]][1:], down_packets = row[DataFrameExtractor.column_name_mapper["down_packets"]][1:],
                            class_type = row["type"], provider_type = row["provider"], sni = row["sni"],flow_config = flow_config
                            )
            row_flow_rep.matchConfig(other_config= needed_flow_config)
            data.append(row_flow_rep)

        #data = DataFrameExtractor.__sampleDataToLength(data= data, required_length= cut_data_length, ratio_value_counts= sample_ratio_value_counts, start_with_invalid_points= start_with_invalid_points)
        return data