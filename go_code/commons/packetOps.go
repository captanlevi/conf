package commons

import (
	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
)

func GetFiveTupleFromPacket(packet *gopacket.Packet) (FiveTuple, bool) {
	var clientIP, serverIP string
	// Check for IPv4 layer

	ipv4Layer := (*packet).Layer(layers.LayerTypeIPv4)
	if ipv4Layer != nil {
		ipv4, _ := ipv4Layer.(*layers.IPv4)
		clientIP = ipv4.SrcIP.String()
		serverIP = ipv4.DstIP.String()
	} else {
		// Check for IPv6 layer
		ipv6Layer := (*packet).Layer(layers.LayerTypeIPv6)
		if ipv6Layer != nil {
			ipv6, _ := ipv6Layer.(*layers.IPv6)
			clientIP = ipv6.SrcIP.String()
			serverIP = ipv6.DstIP.String()
		} else {
			// Neither IPv4 nor IPv6 present, return false
			return FiveTuple{}, false
		}
	}

	// Extract TCP/UDP information
	var srcPort, dstPort uint16
	var protocol uint8
	if tcpLayer := (*packet).Layer(layers.LayerTypeTCP); tcpLayer != nil {
		tcp, _ := tcpLayer.(*layers.TCP)
		srcPort = uint16(tcp.SrcPort)
		dstPort = uint16(tcp.DstPort)
		protocol = 6 // TCP
	} else if udpLayer := (*packet).Layer(layers.LayerTypeUDP); udpLayer != nil {
		udp, _ := udpLayer.(*layers.UDP)
		srcPort = uint16(udp.SrcPort)
		dstPort = uint16(udp.DstPort)
		protocol = 17 // UDP
	} else {
		return FiveTuple{}, false
	}

	// Construct tuple
	tp := FiveTuple{
		ClientIP:   clientIP,
		ServerIP:   serverIP,
		ClientPort: srcPort,
		ServerPort: dstPort,
		Protocol:   protocol,
	}
	return tp, true

}

func GetRevFromFiveTuple(five_tuple FiveTuple) FiveTuple {
	return FiveTuple{
		ClientIP:   five_tuple.ServerIP,
		ServerIP:   five_tuple.ClientIP,
		ClientPort: five_tuple.ServerPort,
		ServerPort: five_tuple.ClientPort,
		Protocol:   five_tuple.Protocol,
	}
}
