from .filters import *
from .enums import *
from .core import *
from typing import Dict
import pyshark
from .sniDetectors import SNIDetectorBase, sni_detector

class TraceConnection:
    def __init__(self,sni_detector : SNIDetectorBase = sni_detector):
        self.sni_detector = sni_detector

    def getConnections(self,pcap_path):        
        cap = pyshark.FileCapture(
        pcap_path,decode_as={'udp.port==3478': 'rtp'}
        )

        connection_dict : Dict[str:Connection] = dict()
       

        for packet in cap:
            
            conn = Connection.getConnFromPacket(packet= packet) # packet.ip.src, packet.ip.dst, packet.tcp.srcport, packet.tcp.dstport
           
            if conn == None:
                continue
            conn_type = conn.connection_type
            sni = self.sni_detector(packet= packet)
            packet_info = None

           
            if conn.getConnKey() not in connection_dict and conn.getRevKey() not in connection_dict:
                connection_dict[conn.getConnKey()] = conn
                packet_info = PacketInfo.getPacketInfoFromPacket(packet= packet,direction= 0)
            elif conn.getConnKey() in connection_dict:
                packet_info = PacketInfo.getPacketInfoFromPacket(packet= packet,direction= 0)
                conn = connection_dict[conn.getConnKey()]
            elif conn.getRevKey() in connection_dict:
                packet_info = PacketInfo.getPacketInfoFromPacket(packet= packet,direction= 1)
                conn = connection_dict[conn.getRevKey()]
              
            if conn.getSNI() == None and sni != None:
                conn.setSNI(sni)
            conn.addPacket(packet_info)


        return list(connection_dict.values())
            
            