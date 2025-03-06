package main

import (
	"conf/p_filter/core"
	"os"
)

func main() {

	num_args, args := len(os.Args), os.Args
	if num_args < 4 {
		panic("Please use code with the input input_pcap_file, output_flows_path and output_packets_path")
	}

	input_pcap_file, output_flows_path, output_packets_path := args[1], args[2], args[3]
	core.Drive(input_pcap_file, output_flows_path, output_packets_path)

}
