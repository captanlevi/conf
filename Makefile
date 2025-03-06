source_pcap_file_path = "../data/gmeet2Members.pcapng"
output_flows_file_path = "../results/flows.csv"
output_packets_file_path = "../results/packets.csv"
truncate_length = 100


run:
	cd go_code;  go run . $(source_pcap_file_path) $(output_flows_file_path) $(output_packets_file_path)
all: run 