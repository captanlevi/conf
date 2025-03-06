package commons

func (flow *Flow) MarkFlowForMedia(rtp_threshold_fraction float32) {
	if len(flow.Packets) == 0 {
		return
	}
	is_stun := false
	is_turn := false
	is_dtls := false

	// Count occurrences of each SSRC
	ssrc_mp := make(map[uint32]int)
	num_packets := len(flow.Packets)

	for _, packet := range flow.Packets {
		if packet.RTPHeader != nil {
			ssrc_mp[packet.RTPHeader.SSRC]++
		}
		if packet.StunHeader != nil {
			is_stun = true
		}
		if packet.TurnHeader != nil {
			is_turn = true
		}
		if packet.DTLSHeader != nil {
			is_dtls = true
		}
	}

	// Compute threshold count
	threshold_count := int(rtp_threshold_fraction * float32(num_packets))

	// Remove SSRCs that do not meet the threshold
	for ssrc, count := range ssrc_mp {
		if count < threshold_count {
			delete(ssrc_mp, ssrc)
		}
	}

	// Remove RTPHeader from packets that do not belong to valid SSRCs
	for i := range flow.Packets {
		if flow.Packets[i].RTPHeader != nil {
			if _, exists := ssrc_mp[flow.Packets[i].RTPHeader.SSRC]; !exists {
				flow.Packets[i].RTPHeader = nil
			}
		}
	}

	flow.HasSTUN = is_stun
	flow.HasTURN = is_turn
	flow.HasDTLS = is_dtls
	flow.HasRTP = len(ssrc_mp) > 0

}

func (flow *Flow) IsFlowMedia() bool {
	if flow.HasSTUN && (flow.HasRTP || flow.HasDTLS) {
		return true
	}
	return false
}
