package core

import (
	"conf/p_filter/commons"
	"fmt"
	"log"
	"time"

	"github.com/google/gopacket"
	"github.com/google/gopacket/pcap"
)

func Drive(pcap_file_path string, flow_save_path string, packets_save_path string) {

	handle, err := pcap.OpenOffline(pcap_file_path)
	if err != nil {
		log.Fatal("Failed to open PCAP file:", err)
	}
	defer handle.Close()
	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
	mp := make(map[commons.FiveTuple]*[]commons.Flow)
	final_flows := make([]commons.Flow, 0)
	index := 0
	flow_id := 0
	var last_timestamp time.Time
	for packet := range packetSource.Packets() {
		ProcessPacket(&packet, &mp)
		if index%500000 == 0 {
			fmt.Println("Cleaning up")
			// Removing ended connections, and moving ended media connections to final_flows
			CleanUp(packet.Metadata().Timestamp, &mp, false, &final_flows)
			fmt.Println("map size ", len(mp))
		}
		if len(final_flows) > 1000 {
			// Saving flows
			flow_id = Save(&final_flows, flow_id, packets_save_path, flow_save_path)
		}

		index += 1
		last_timestamp = packet.Metadata().Timestamp
	}

	CleanUp(last_timestamp, &mp, true, &final_flows)
	Save(&final_flows, flow_id, packets_save_path, flow_save_path)
}
