from .core import Connection
from .utils import convertUNIXToHumanReadable

class ConnStats:
    def __init__(self,conn : Connection):
        self.conn = conn
    

    def getUpStreamBytes(self):
        packet_stream = self.conn.packet_stream

        up_stream_bytes = 0
        for packet in packet_stream:
            if packet.direction == 0:
                up_stream_bytes += packet.length
        return up_stream_bytes

    def getDownStreamBytes(self):
        packet_stream = self.conn.packet_stream

        down_stream_bytes = 0
        for packet in packet_stream:
            if packet.direction == 1:
                down_stream_bytes += packet.length
        return down_stream_bytes
    
    def getConnSpan(self):
        packet_stream = self.conn.packet_stream
        return packet_stream[-1].timestamp - packet_stream[0].timestamp
    
    def getPacketTypeCounts(self):
        packet_stream = self.conn.packet_stream
        packet_type_counts = dict()
        for packet in packet_stream:
            packet_type_counts[packet.getPacketType()] = 1 + packet_type_counts.get(packet.getPacketType(), 0)
        return packet_type_counts
    def getStartTime(self):
        return convertUNIXToHumanReadable(self.conn.packet_stream[0].timestamp)
    def getEndTime(self):
        return convertUNIXToHumanReadable(self.conn.packet_stream[-1].timestamp)