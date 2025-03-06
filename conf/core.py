# Define PacketInfo class
from typing import List
from .enums import PacketType,TransPortType



def getPacketType(packet):
    if hasattr(packet, "tls"):
        return PacketType.TLS
    elif hasattr(packet, "dtls"):
        return PacketType.DTLS
    elif hasattr(packet, "rtp"):
        return PacketType.RTP
    elif hasattr(packet, "rtcp"):
        return PacketType.RTCP
    elif hasattr(packet, "stun"):
        return PacketType.STUN
    elif hasattr(packet, "turn"):
        return PacketType.TURN
    elif hasattr(packet, "quic"):
        return PacketType.QUIC
    
    elif hasattr(packet, "tcp"):
        return PacketType.TCP
    elif hasattr(packet, "udp"):
        return PacketType.UDP
    return PacketType.UNKNOWN


class PacketInfo:
    def __init__(self, length, timestamp, direction, packet_type : PacketType):
        self.length = length
        self.timestamp = timestamp
        self.direction = direction
        self.__packet_type = packet_type
    
    @staticmethod
    def getPacketInfoFromPacket(packet,direction):
        packet_type = getPacketType(packet)
        return PacketInfo(length= int(packet.length), timestamp = float(packet.sniff_time.timestamp()), direction= direction, packet_type= packet_type)
        

    def getPacketType(self):
        return self.__packet_type
    

class Connection:
    def __init__(self,src_ip,dst_ip,src_port,dst_port,transport_type : TransPortType,other_info = None):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.transport_type = transport_type
        self.connection_type : set[PacketType] = set()
        self.packet_stream : List[PacketInfo] = []
        self.other_info = other_info
        self.__sni = None

    def getConnKey(self):
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port,self.transport_type.name)
    def getRevKey(self):
        return (self.dst_ip, self.src_ip, self.dst_port, self.src_port,self.transport_type.name)
    

    @staticmethod
    def getTransPortType(packet):
        if hasattr(packet, "ip") == False:
            return TransPortType.UNKNOWN
        if hasattr(packet, "tcp"):
            return TransPortType.TCP
        elif hasattr(packet, "udp"):
            return TransPortType.UDP
        return TransPortType.UNKNOWN
    
  

    @staticmethod
    def getConnFromPacket(packet):
        transport_type = Connection.getTransPortType(packet)

        packet_type = getPacketType(packet)
        if packet_type == PacketType.UNKNOWN:
            return None

        if transport_type == TransPortType.TCP:
            return Connection(src_ip= packet.ip.src, 
                              dst_ip= packet.ip.dst, 
                              src_port= packet.tcp.srcport, 
                              dst_port =  packet.tcp.dstport,
                              transport_type= transport_type,
                              ) 
        elif transport_type == TransPortType.UDP:
             return Connection(
                        src_ip=packet.ip.src,
                        dst_ip=packet.ip.dst,
                        src_port=packet.udp.srcport,  #UDP instead of TCP
                        dst_port=packet.udp.dstport,
                        transport_type=transport_type,
                    )

        return None
    
    def addPacket(self,packet : PacketInfo):
        if packet.getPacketType() == PacketType.UNKNOWN:
            assert False , "Packet Type is Unknown"
        self.connection_type.add(packet.getPacketType())
        self.packet_stream.append(packet)

    def __len__(self):
        return len(self.packet_stream)
    
    def setSNI(self,sni):
        self.__sni = sni
    
    def getSNI(self):
        return self.__sni
    

    def getPacketsOfTypes(self,packet_types : set[PacketType]):
        return list(filter(lambda x : x.getPacketType() in packet_types, self.packet_stream))
    
    def getSubConnection(self,packet_types : List[PacketType]):
        packets = self.getPacketsOfTypes(packet_types= set(packet_types))

        new_conn = Connection(src_ip= self.src_ip,dst_ip= self.dst_ip,src_port= self.src_port,
                              dst_port= self.dst_port,transport_type= self.transport_type,
                              other_info= self.other_info
                              )
        for packet in packets:
            new_conn.addPacket(packet= packet)
        return new_conn
    
