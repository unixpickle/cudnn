package cudnn

import (
	"testing"

	"github.com/unixpickle/cuda"
)

var testContext *cuda.Context

func setupTest(t *testing.T) *cuda.Context {
	if testContext == nil {
		devices, err := cuda.AllDevices()
		if err != nil {
			t.Fatal(err)
		}
		if len(devices) == 0 {
			t.Fatal("no CUDA devices")
		}
		testContext, err = cuda.NewContext(devices[0], -1)
		if err != nil {
			t.Fatal(err)
		}
	}
	return testContext
}
