// Package cudnn is a binding for cuDNN, NVIDIA's deep
// neural network library.
package cudnn

/*
#include <cudnn.h>
*/
import "C"

import (
	"runtime"

	"github.com/unixpickle/cuda"
)

// A Handle is used to use the cuDNN API.
//
// It is recommended that you always use the handle from
// one cuda.Context.
type Handle struct {
	handle C.cudnnHandle_t
	ctx    *cuda.Context
}

// NewHandle creates a Handle.
//
// You should call this from the cuda.Context.
//
// The handle will be associated with the context's
// current device.
func NewHandle(ctx *cuda.Context) (*Handle, error) {
	res := &Handle{ctx: ctx}
	status := C.cudnnCreate(&res.handle)
	if err := newError("cudnnCreate", status); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(obj *Handle) {
		obj.ctx.Run(func() error {
			C.cudnnDestroy(obj.handle)
			return nil
		})
	})
	return res, nil
}
