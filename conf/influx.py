from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

import requests
import json
import datetime
import pandas as pd
from zoneinfo import ZoneInfo

from .core import Connection
from .utils import convertUNIXToHumanReadable



def getTimestampsAndLengthAndDirection(connection: Connection):
    timestamps = []
    lengths = []
    direction = []
    for packet in connection.packet_stream:
        timestamps.append(packet.timestamp)
        lengths.append(packet.length)
        direction.append(packet.direction)
    timestamps = [convertUNIXToHumanReadable(timestamp) for timestamp in timestamps]
    return timestamps,lengths,direction

class InfluxImpl:
    def __init__(self, url = "http://localhost:8086", token = "iamadmin", org = "shadowForce", bucket = "bukbuk"):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
    
    def __getDfFromPacketStream(self,connection: Connection):
        timestamps,length,direction = getTimestampsAndLengthAndDirection(connection)
        df = pd.DataFrame(data= {"timestamps": timestamps, "length": length, "direction": direction})
        df.set_index("timestamps",inplace=True)
        df["direction"] = df["direction"].astype(int)
        return df


    def __writeDfToInflux(self,df,measurement_name,tag_columns):
        # Create the InfluxDB client
        # Create a write API instance
        client = InfluxDBClient(url=self.url, token=self.token)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        # Write points to InfluxDB
        write_api.write(bucket=self.bucket, org=self.org, record=df, data_frame_measurement_name= measurement_name, data_frame_tag_columns=tag_columns)
        # Close the client when done
        client.close()

    def write(self,connection : Connection,measurement_name,tag_columns):
        df = self.__getDfFromPacketStream(connection)
        self.__writeDfToInflux(df,measurement_name,tag_columns)


    def __deleteMeasurementFromInfluxdb(self, measurement,predicate=None):

        def get_current_time_rfc3339nano():
            current_time =  datetime.datetime.now(datetime.timezone.utc).isoformat("T") + "Z"
            current_time = current_time.split(".")[0] + "Z"
            return current_time
        headers = {
            'Authorization': f'Token {self.token}',
            'Content-Type': 'application/json'
        }

        
        data = {
            #"start": "2009-01-02T23:00:00Z",
            #"stop": get_current_time_rfc3339nano(),
            "start" : '1970-01-01T00:00:00Z',
            "stop" : '2100-01-01T00:00:00Z',

            "predicate": f'_measurement="{measurement}"'
        }
        
        if predicate:
            data["predicate"] += f' AND {predicate}'
        
        response = requests.post(
            f'{self.url}/api/v2/delete?org={self.org}&bucket={self.bucket}',
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 204:
            print("Data deleted successfully.")
        else:
            print(f"Failed to delete data: {response.status_code} - {response.text}")

    def delete(self,measurement, predicate=None):
        self.__deleteMeasurementFromInfluxdb(measurement,predicate)

    
    def getMeasurements(self):

        # Initialize the client
        client = InfluxDBClient(url= self.url, token= self.token, org= self.org)

        # Flux query to get all measurements
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "{self.bucket}")
        '''

        # Execute query
        query_api = client.query_api()
        tables = query_api.query(query)

        # Extract and print measurement names
        measurements = [record.values["_value"] for table in tables for record in table.records]

        # Close client
        client.close()

        return measurements
        