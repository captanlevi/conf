from abc import ABC, abstractmethod

# Chain of Responsibility Pattern


class SNIDetectorBase(ABC):
   
   def __init__(self, next = None):
      self.next : SNIDetectorBase = next
   
   @abstractmethod
   def getSNIFromPacket(self, packet):
      pass
   
   def __call__(self, packet):
       return self.getSNIFromPacket(packet)
   

class TLSSNIDetector(SNIDetectorBase):
    def getSNIFromPacket(self, packet):
        try:
            return packet.tls.handshake_extensions_server_name
        except AttributeError as e:
            if self.next != None:
                return self.next.getSNIFromPacket(packet)
            return None
        
class DTLSNIDetector(SNIDetectorBase):
    def getSNIFromPacket(self, packet):
        try:
            return packet.dtls.handshake_extensions_server_name
        except AttributeError as e:
            if self.next != None:
                return self.next.getSNIFromPacket(packet)
            return None
        
class QuickSNIDetector(SNIDetectorBase):
    pass





sni_detector = TLSSNIDetector(next=DTLSNIDetector(next=None))