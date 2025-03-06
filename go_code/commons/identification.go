package commons

import (
	"encoding/binary"
	"encoding/hex"

	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
)

func GetSTUNHeader(packet *gopacket.Packet) *StunHeader {
	udp_layer := (*packet).Layer(layers.LayerTypeUDP)
	if udp_layer == nil {
		return nil
	}
	udp_layer = udp_layer.(*layers.UDP)
	payload_bytes := udp_layer.LayerPayload()

	// If payload bytes are not atleast 20, STUN  header cannot exist
	if len(payload_bytes) < 20 {
		return nil
	}
	stun_first_2_bytes := binary.BigEndian.Uint16(payload_bytes[:2])
	message_length := binary.BigEndian.Uint16(payload_bytes[2:4])
	magic_cookie := hex.EncodeToString(payload_bytes[4:8])
	transaction_id := hex.EncodeToString(payload_bytes[8:20])
	// The first 2 bits should be zero and magic cookie should exist and match
	if (stun_first_2_bytes&0b1100000000000000 != 0) || (magic_cookie != STUN_MAGIC_COOKIE) {
		return nil
	}
	message_class := ExtractBitsFromMask(stun_first_2_bytes, STUN_MESSAGE_CLASS_MASK)
	message_method := ExtractBitsFromMask(stun_first_2_bytes, STUN_MESSAGE_METHOD_MASK)

	return &StunHeader{MessageClass: uint8(message_class), MessageMethod: message_method,
		TransactionId: transaction_id, MessageLength: message_length}

}

func GetRTPHeader(packet *gopacket.Packet) *RTPHeader {
	udp_layer := (*packet).Layer(layers.LayerTypeUDP)
	if udp_layer == nil {
		return nil
	}
	udp_layer = udp_layer.(*layers.UDP)
	payload := udp_layer.LayerPayload()
	rtpPacket := &rtp.Packet{}
	err := rtpPacket.Unmarshal(payload)
	if err != nil {
		// If unmarshaling fails, this is not an RTP packet
		return nil
	}
	header := rtpPacket.Header
	if header.Version != 2 {
		return nil
	}
	return &RTPHeader{SequenceNumber: header.SequenceNumber, SSRC: header.SSRC,
		Timestamp: header.Timestamp, Version: header.Version, PayloadType: header.PayloadType}
}

func GetRTCPHeader(packet *gopacket.Packet) *RTCPHeader {
	udp_layer := (*packet).Layer(layers.LayerTypeUDP)
	if udp_layer == nil {
		return nil
	}
	udp_layer = udp_layer.(*layers.UDP)
	payload := udp_layer.LayerPayload()
	rtcpPackets, err := rtcp.Unmarshal(payload)
	if err != nil {
		return nil
	}
	return &RTCPHeader{NumPackets: uint16(len(rtcpPackets))}
}

func GetTURNHeader(packet *gopacket.Packet) *TurnHeader {
	// Extract UDP layer
	udpLayer := (*packet).Layer(layers.LayerTypeUDP)
	if udpLayer == nil {
		return nil // Not a UDP packet, so not a Channel Data TURN message
	}
	udp, _ := udpLayer.(*layers.UDP)
	payloadBytes := udp.LayerPayload()
	// Channel Data TURN header requires at least 4 bytes
	if len(payloadBytes) < 4 {
		return nil
	}

	// Extract channel number and length from the first 4 bytes
	channelNumber := binary.BigEndian.Uint16(payloadBytes[:2])
	dataLength := binary.BigEndian.Uint16(payloadBytes[2:4])
	// Validate channel number range (0x4000 to 0x7FFF per RFC 5766)

	if channelNumber < 0x4000 {
		return nil // Invalid channel number, not a Channel Data TURN message
	}

	// Validate data length: should match remaining payload size
	// Note: TURN pads data to a 4-byte boundary, so payload length should be >= dataLength
	if int(dataLength) > len(payloadBytes)-4 {
		return nil // Data length exceeds payload, invalid TURN message
	}

	// Return the parsed TURN header
	return &TurnHeader{
		ChannelNumber: channelNumber,
		DataLength:    dataLength,
	}
}

func GetDTLSHeader(packet *gopacket.Packet) *DTLSHeader {
	udpLayer := (*packet).Layer(layers.LayerTypeUDP)
	if udpLayer == nil {
		return nil // Not a UDP packet, so not a DTLS
	}
	udp, _ := udpLayer.(*layers.UDP)
	payloadBytes := udp.LayerPayload()
	// DTLS header requires at least 13 bytes
	if len(payloadBytes) < 13 {
		return nil
	}

	seq_num_buffer := make([]byte, 0)
	seq_num_buffer = append(seq_num_buffer, 0, 0)
	for i := 0; i < 6; i++ {
		seq_num_buffer = append(seq_num_buffer, payloadBytes[i+5])
	}

	header := &DTLSHeader{
		ContentType:    payloadBytes[0],
		Version:        hex.EncodeToString(payloadBytes[1:3]),
		Epoch:          binary.BigEndian.Uint16(payloadBytes[3:5]),
		SequenceNumber: binary.BigEndian.Uint64(seq_num_buffer),
		Length:         binary.BigEndian.Uint16(payloadBytes[11:13]),
	}

	if _, exists := DTLS_VERSION_MAPPING[header.Version]; !exists {
		return nil
	}
	if int(header.Length) != len(payloadBytes)-13 {
		return nil
	}

	return header
}
