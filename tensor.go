package cudnn

/*
#include <cudnn.h>

const cudnnTensorFormat_t goCudnnNCHW = CUDNN_TENSOR_NCHW;
const cudnnTensorFormat_t goCudnnNHWC = CUDNN_TENSOR_NHWC;

const cudnnDataType_t goCudnnFloat = CUDNN_DATA_FLOAT;
const cudnnDataType_t goCudnnDouble = CUDNN_DATA_DOUBLE;
const cudnnDataType_t goCudnnHalf = CUDNN_DATA_HALF;

const size_t goCudnnIntSize = sizeof(int);
const int * goCudnnNullArray = NULL;
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

func newDataTypeC(val C.cudnnDataType_t) DataType {
	switch val {
	case C.goCudnnFloat:
		return Float
	case C.goCudnnDouble:
		return Double
	case C.goCudnnHalf:
		return Half
	default:
		panic("unable to create DataType")
	}
}

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

// Set4DEx is like Set4D, but with custom strides.
//
// This should be called from the cuda.Context.
func (t *TensorDesc) Set4DEx(dataType DataType, n, c, h, w, nStride, cStride,
	hStride, wStride int) error {
	status := C.cudnnSetTensor4dDescriptorEx(t.desc, dataType.cValue(),
		safeIntToC(n), safeIntToC(c), safeIntToC(h), safeIntToC(w),
		safeIntToC(nStride), safeIntToC(cStride), safeIntToC(hStride),
		safeIntToC(wStride))
	return newError("cudnnSetTensor4dDescriptorEx", status)
}

// Get4D gets the parameters of the 4D tensor.
//
// This should be called from the cuda.Context.
func (t *TensorDesc) Get4D() (dataType DataType, n, c, h, w, nStride, cStride,
	hStride, wStride int, err error) {
	var cn, cc, ch, cw, cnStride, ccStride, chStride, cwStride C.int
	var cType C.cudnnDataType_t
	status := C.cudnnGetTensor4dDescriptor(t.desc, &cType, &cn, &cc, &ch, &cw,
		&cnStride, &ccStride, &chStride, &cwStride)
	err = newError("cudnnGetTensor4dDescriptor", status)
	dataType = newDataTypeC(cType)
	n, c, h, w = int(cn), int(cc), int(ch), int(cw)
	nStride, cStride, hStride, wStride = int(cnStride), int(ccStride), int(chStride),
		int(cwStride)
	return
}

// Set sets variable-length tensor information.
//
// This should be called from the cuda.Context.
func (t *TensorDesc) Set(dataType DataType, dims, strides []int) error {
	if len(dims) != len(strides) {
		panic("dims and stride length mismatch")
	}
	cDims := make([]C.int, len(dims))
	cStrides := make([]C.int, len(strides))
	for i, x := range dims {
		cDims[i] = safeIntToC(x)
	}
	for i, x := range strides {
		cStrides[i] = safeIntToC(x)
	}
	status := C.cudnnSetTensorNdDescriptor(t.desc, dataType.cValue(),
		safeIntToC(len(dims)), (*C.int)(&cDims[0]), (*C.int)(&cStrides[0]))
	return newError("cudnnSetTensorNdDescriptor", status)
}

// Get gets the variable-length tensor information.
//
// This should be called from the cuda.Context.
func (t *TensorDesc) Get() (dataType DataType, dims, strides []int, err error) {
	var cType C.cudnnDataType_t
	var nDims C.int
	status := C.cudnnGetTensorNdDescriptor(t.desc, 0, &cType, &nDims,
		C.goCudnnNullArray, C.goCudnnNullArray)
	err = newError("cudnnGetTensorNdDescriptor", status)
	if err != nil {
		return
	}

	outDims := make([]C.int, int(nDims))
	outStrides := make([]C.int, int(nDims))
	status = C.cudnnGetTensorNdDescriptor(t.desc, nDims, &cType, &nDims,
		(*C.int)(&outDims[0]), (*C.int)(&outStrides[0]))
	err = newError("cudnnGetTensorNdDescriptor", status)
	if err != nil {
		return
	}

	dataType = newDataTypeC(cType)
	dims = make([]int, len(outDims))
	strides = make([]int, len(outStrides))
	for i := range dims {
		dims[i] = int(outDims[i])
		strides[i] = int(outStrides[i])
	}
	return
}
