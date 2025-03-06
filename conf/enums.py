from enum import Enum


class PacketType(Enum):
    TLS = 0
    DTLS = 1
    RTP = 2
    RTCP = 3
    STUN = 4
    TURN = 5
    QUIC = 6

    TCP = 7
    UDP = 8
    UNKNOWN = 9

class TransPortType(Enum):
    TCP = 0
    UDP = 1
    UNKNOWN = 2
