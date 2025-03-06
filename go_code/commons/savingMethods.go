package commons

import "strconv"

type SaveStructCSV interface {
	GetStringArr() []string
	GetHeader() []string
}

func (flow SaveFlow) GetStringArr() []string {
	return []string{
		strconv.Itoa(flow.FlowId),
		flow.FiveTuple.String(),
		flow.StartTime.String(),
		flow.EndTime.String(),
	}

}

func (flow SaveFlow) GetHeader() []string {
	return []string{"FlowId", "FiveTuple", "StartTime", "EndTime"}
}

func (packet SavePacket) GetStringArr() []string {
	return []string{
		strconv.Itoa(packet.FlowID),
		strconv.Itoa(packet.PacketLength),
		packet.Timestamp.String(),
		strconv.Itoa(int(packet.Direction)),
		strconv.FormatBool(packet.RTP),
		strconv.FormatBool(packet.DTLS),
		strconv.FormatBool(packet.STUN),
	}
}

func (packet SavePacket) GetHeader() []string {
	return []string{"FlowID", "PacketLength", "Timestamp", "Direction", "RTP", "DTLS", "STUN"}
}
