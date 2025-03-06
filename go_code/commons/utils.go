package commons

func ExtractBits(source uint16, bits []int) uint16 {
	var result uint16 = 0

	for i, bit_index := range bits {
		bit := (source >> uint16(bit_index)) & 1
		result |= (bit << i)
	}
	return result

}

func ExtractBitsFromMask(source uint16, mask uint16) uint16 {
	var result uint16 = 0

	result_index := 0
	for i := 0; i < 16; i++ {

		mask_bit := (mask >> i) & 1
		source_bit := (source >> i) & 1
		if mask_bit == 1 {
			result |= (source_bit << uint16(result_index))
			result_index += 1
		}

	}
	return result
}
