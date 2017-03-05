package cudnn

import "C"

func safeIntToC(x int) C.int {
	if x > int(C.int(^C.uint(0)/2)) {
		panic("int value out of bounds")
	} else if x < int((-C.int(^C.uint(0)/2))-1) {
		panic("int value out of bounds")
	}
	return C.int(x)
}
