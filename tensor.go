package cudnn

/*
#include <cudnn.h>

const cudnnTensorFormat_t goCudnnNCHW = CUDNN_TENSOR_NCHW;
const cudnnTensorFormat_t goCudnnNHWC = CUDNN_TENSOR_NHWC;

const cudnnDataType_t goCudnnFloat = CUDNN_DATA_FLOAT;
const cudnnDataType_t goCudnnDouble = CUDNN_DATA_DOUBLE;
const cudnnDataType_t goCudnnHalf = CUDNN_DATA_HALF;
*/
import "C"

import (
	"runtime"

	"github.com/unixpickle/cuda"
)

// TensorFormat describes the order in which a 4D tensor
// is layed out.
type TensorFormat int

const (
	TensorNCHW TensorFormat = iota
	TensorNHWC
)

func (t TensorFormat) cValue() C.cudnnTensorFormat_t {
	switch t {
	case TensorNCHW:
		return C.goCudnnNCHW
	case TensorNHWC:
		return C.goCudnnNHWC
	default:
		panic("invalid TensorFormat")
	}
}

// DataType specifies a type of floating point.
type DataType int

const (
	Float DataType = iota
	Double
	Half
)

func (d DataType) cValue() C.cudnnDataType_t {
	switch d {
	case Float:
		return C.goCudnnFloat
	case Double:
		return C.goCudnnDouble
	case Half:
		return C.goCudnnHalf
	default:
		panic("invalid DataType")
	}
}

// A TensorDesc describes a tensor's memory layout.
type TensorDesc struct {
	desc C.cudnnTensorDescriptor_t
	ctx  *cuda.Context
}

// NewTensorDesc creates a new TensorDesc.
//
// This should be called from the cuda.Context.
func NewTensorDesc(ctx *cuda.Context) (*TensorDesc, error) {
	res := &TensorDesc{ctx: ctx}
	status := C.cudnnCreateTensorDescriptor(&res.desc)
	if err := newError("cudnnCreateTensorDescriptor", status); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(t *TensorDesc) {
		t.ctx.Run(func() error {
			C.cudnnDestroyTensorDescriptor(t.desc)
			return nil
		})
	})
	return res, nil
}

// Set4D initializes the descriptor for a 4D tensor.
//
// This should be called from the cuda.Context.
func (t *TensorDesc) Set4D(format TensorFormat, dataType DataType, n, c, h, w int) error {
	status := C.cudnnSetTensor4dDescriptor(t.desc, format.cValue(), dataType.cValue(),
		safeIntToC(n), safeIntToC(c), safeIntToC(h), safeIntToC(w))
	return newError("cudnnSetTensor4dDescriptor", status)
}
