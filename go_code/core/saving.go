package core

import (
	"conf/p_filter/commons"
	"encoding/csv"
	"os"
)

// SaveFlows writes flow and packet data to CSV files and clears the in-memory slice.
func Save(flows *[]commons.Flow, flow_id int, packets_file_path string, flows_file_path string) int {
	// Prepare slices for saving
	save_packets := make([]commons.SaveStructCSV, 0)
	save_flows := make([]commons.SaveStructCSV, 0)

	// Populate save_flows and save_packets
	for _, flow := range *flows {
		save_flow := commons.SaveFlow{
			FlowId:    flow_id,
			FiveTuple: flow.FiveTuple,
			StartTime: flow.StartTime,
			EndTime:   flow.EndTime,
		}
		save_flows = append(save_flows, save_flow)

		truncated_slice := flow.Packets
		if len(truncated_slice) > commons.PACKETS_TRUNCATE_LENGTH {
			truncated_slice = truncated_slice[:commons.PACKETS_TRUNCATE_LENGTH]
		}
		for _, packet := range truncated_slice {
			save_packet := commons.SavePacket{
				FlowID:       flow_id,
				PacketLength: packet.PacketLength,
				Timestamp:    packet.Timestamp,
				Direction:    packet.Direction,
				RTP:          packet.RTPHeader != nil,
				DTLS:         packet.DTLSHeader != nil,
				STUN:         packet.StunHeader != nil,
			}
			save_packets = append(save_packets, save_packet)
		}

		flow_id++
	}

	// Clear the flows slice after saving
	*flows = (*flows)[:0]

	// Append data to CSV files
	appendDataToCSV(packets_file_path, save_packets)
	appendDataToCSV(flows_file_path, save_flows)
	return flow_id
}

// appendPacketsToCSV appends packet data to a CSV file.
func appendDataToCSV(filename string, data []commons.SaveStructCSV) {
	if len(data) == 0 {
		return
	}
	fileExists := fileExists(filename)

	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header only if file is newly created or empty
	if !fileExists {
		writer.Write(data[0].GetHeader())
	}

	for _, data_element := range data {
		writer.Write(data_element.GetStringArr())
	}
}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}
