package commons

import (
	"fmt"
	"time"
)

type Flow struct {
	FiveTuple *FiveTuple
	StartTime time.Time
	EndTime   time.Time
	Packets   []PacketInfo
	HasRTP    bool
	HasSTUN   bool
	HasRTCP   bool
	HasTURN   bool
	HasDTLS   bool
	Closed    bool
}

type PacketInfo struct {
	PacketLength int
	Timestamp    time.Time
	Direction    uint8
	RTPHeader    *RTPHeader
	StunHeader   *StunHeader
	TurnHeader   *TurnHeader
	RTCPHeader   *RTCPHeader
	DTLSHeader   *DTLSHeader
}

type FiveTuple struct {
	ClientIP   string
	ServerIP   string
	ClientPort uint16
	ServerPort uint16
	Protocol   uint8
}

func (ft *FiveTuple) String() string {
	return fmt.Sprintf("%s:%d->%s:%d->(Protocol: %d)",
		ft.ClientIP, ft.ClientPort, ft.ServerIP, ft.ServerPort, ft.Protocol)
}

type StunHeader struct {
	MessageClass  uint8
	MessageMethod uint16
	TransactionId string
	MessageLength uint16
}

type RTPHeader struct {
	SequenceNumber uint16
	SSRC           uint32
	Timestamp      uint32
	PayloadType    uint8
	Version        uint8
}

type RTCPHeader struct {
	NumPackets uint16
}

type TurnHeader struct {
	ChannelNumber uint16
	DataLength    uint16
}

type DTLSHeader struct {
	ContentType    byte
	Version        string
	Epoch          uint16
	SequenceNumber uint64
	Length         uint16
}

type SavePacket struct {
	FlowID       int
	PacketLength int
	Timestamp    time.Time
	Direction    uint8
	RTP          bool
	STUN         bool
	DTLS         bool
}

type SaveFlow struct {
	FlowId    int
	FiveTuple *FiveTuple
	StartTime time.Time
	EndTime   time.Time
}
