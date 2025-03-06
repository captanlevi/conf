package core

import (
	"conf/p_filter/commons"
	"time"

	"github.com/google/gopacket"
)

func CleanUp(current_timestamp time.Time, mp *map[commons.FiveTuple]*[]commons.Flow, final bool, final_flows *[]commons.Flow) {
	// Step 1: Cleanup individual flow slices
	for _, flows := range *mp {
		// Create a new slice for filtered flows
		newFlows := make([]commons.Flow, 0, len(*flows)) // Preallocate capacity

		for i := range *flows {
			flow := &((*flows)[i])
			if final || (current_timestamp.After(flow.EndTime.Add(commons.TIME_BUFFER))) {
				flow.MarkFlowForMedia(commons.RTP_CHECK_THRESHOLD_FRACTION)
				flow.Closed = true
			}

			if flow.Closed {
				if flow.IsFlowMedia() {
					// Keep media flows
					*final_flows = append(*final_flows, *flow)
				}
			} else {
				// Keep ongoing flows
				newFlows = append(newFlows, *flow)
			}
		}

		// Update the original slice with the filtered values
		*flows = newFlows
	}

	// Step 2: Collect keys for removal (to avoid modifying the map while iterating)
	var keysToDelete []commons.FiveTuple
	for key, flows := range *mp {
		if len(*flows) == 0 {
			keysToDelete = append(keysToDelete, key)
		}
	}

	// Step 3: Remove keys from the map
	for _, key := range keysToDelete {
		delete(*mp, key)
	}
}

func GetPacketInfoFromGoPacket(packet *gopacket.Packet, direction uint8) commons.PacketInfo {
	current_packet_timestamp := (*packet).Metadata().Timestamp
	packet_info := commons.PacketInfo{RTPHeader: commons.GetRTPHeader(packet),
		StunHeader: commons.GetSTUNHeader(packet), RTCPHeader: commons.GetRTCPHeader(packet), TurnHeader: commons.GetTURNHeader(packet), DTLSHeader: commons.GetDTLSHeader(packet),
		Direction: direction, Timestamp: current_packet_timestamp, PacketLength: (*packet).Metadata().CaptureLength}

	return packet_info

}

func AddPacketToFlows(packet *gopacket.Packet, flows *[]commons.Flow, direction uint8, five_tuple *commons.FiveTuple) {
	current_packet_timestamp := (*packet).Metadata().Timestamp
	packet_info := GetPacketInfoFromGoPacket(packet, direction)

	added := false
	for i := range *flows {
		flow := &((*flows)[i])

		lower_range := flow.StartTime.Add(-commons.TIME_BUFFER)
		upper_range := flow.EndTime.Add(commons.TIME_BUFFER)

		if current_packet_timestamp.After(lower_range) && current_packet_timestamp.Before(upper_range) {
			flow.Packets = append(flow.Packets, packet_info)
			if current_packet_timestamp.Before(flow.StartTime) {
				flow.StartTime = current_packet_timestamp
			}
			if current_packet_timestamp.After(flow.EndTime) {
				flow.EndTime = current_packet_timestamp
			}
			added = true
			// Now mark if this flow has a STUN, TCP or other packets as part of the connection
			break
		}
	}

	if !added {
		flow := commons.Flow{StartTime: current_packet_timestamp, EndTime: current_packet_timestamp, Closed: false, FiveTuple: five_tuple}
		flow.Packets = append(flow.Packets, packet_info)
		// Now mark if this flow has a STUN, TCP or other packets as part of the connection
		*flows = append(*flows, flow)
	}

}

func ProcessPacket(packet *gopacket.Packet, mp *map[commons.FiveTuple]*[]commons.Flow) {

	five_tuple, success := commons.GetFiveTupleFromPacket(packet)
	if !success {
		return
	}
	if five_tuple.Protocol != 17 {
		return
	}

	rev_tuple := commons.GetRevFromFiveTuple(five_tuple)

	if flows, exists := (*mp)[five_tuple]; exists {
		AddPacketToFlows(packet, flows, 0, &five_tuple)

	} else if flows, exists := (*mp)[rev_tuple]; exists {
		AddPacketToFlows(packet, flows, 1, &rev_tuple)
	} else {
		new_flows := make([]commons.Flow, 0)
		(*mp)[five_tuple] = &new_flows
		AddPacketToFlows(packet, &new_flows, 0, &five_tuple)
	}
}
