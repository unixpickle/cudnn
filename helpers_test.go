package cudnn

import (
	"testing"

	"github.com/unixpickle/cuda"
)

var testContext *cuda.Context
var testHandle *Handle

func setupTest(t *testing.T) (*cuda.Context, *Handle) {
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
	if testHandle == nil {
		err := <-testContext.Run(func() (err error) {
			testHandle, err = NewHandle(testContext)
			if err != nil {
				return err
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return testContext, testHandle
}
