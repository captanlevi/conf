package commons

import "time"

var PACKETS_TRUNCATE_LENGTH int = 100

var STUN_MAGIC_COOKIE = "2112a442"
var STUN_MESSAGE_CLASS_MASK uint16 = 0b0000000100010000
var STUN_MESSAGE_METHOD_MASK uint16 = 0b0011111011101111

var RTP_CHECK_THRESHOLD_FRACTION float32 = .1

var TIME_BUFFER time.Duration = 30 * time.Second

var DTLS_VERSION_MAPPING map[string]uint8 = map[string]uint8{
	"feff": 1,
	"fefd": 2,
	"0304": 3,
}
