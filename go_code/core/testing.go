package core

import (
	"conf/p_filter/commons"
	"fmt"
)

func PrintMp(mp *map[commons.FiveTuple]*[]commons.Flow) {

	for key, value := range *mp {

		for _, flow := range *value {
			fmt.Println(key, len(flow.Packets), flow.IsFlowMedia(), flow.HasTURN)

		}
	}

}

func PrintFlows(flows *[]commons.Flow) {

	for _, flow := range *flows {
		fmt.Println(flow.FiveTuple, len(flow.Packets), flow.IsFlowMedia(), flow.HasTURN)
	}
}
