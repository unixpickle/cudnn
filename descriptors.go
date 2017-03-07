package cudnn

/*
#include <cudnn.h>

const cudnnTensorFormat_t goCudnnNCHW = CUDNN_TENSOR_NCHW;
const cudnnTensorFormat_t goCudnnNHWC = CUDNN_TENSOR_NHWC;

const cudnnDataType_t goCudnnFloat = CUDNN_DATA_FLOAT;
const cudnnDataType_t goCudnnDouble = CUDNN_DATA_DOUBLE;
const cudnnDataType_t goCudnnHalf = CUDNN_DATA_HALF;

const cudnnConvolutionMode_t goCudnnConvolution = CUDNN_CONVOLUTION;
const cudnnConvolutionMode_t goCudnnCorrelation = CUDNN_CROSS_CORRELATION;

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

func newTensorFormatC(val C.cudnnTensorFormat_t) TensorFormat {
	switch val {
	case C.goCudnnNCHW:
		return TensorNCHW
	case C.goCudnnNHWC:
		return TensorNHWC
	default:
		panic("unable to create TensorFormat")
	}
}

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

// ConvMode specifies the type of convolution to perform.
type ConvMode int

const (
	Convolution ConvMode = iota
	CrossCorrelation
)

func newConvModeC(c C.cudnnConvolutionMode_t) ConvMode {
	switch c {
	case C.goCudnnConvolution:
		return Convolution
	case C.goCudnnCorrelation:
		return CrossCorrelation
	default:
		panic("unable to create ConvMode")
	}
}

func (c ConvMode) cValue() C.cudnnConvolutionMode_t {
	switch c {
	case Convolution:
		return C.goCudnnConvolution
	case CrossCorrelation:
		return C.goCudnnCorrelation
	default:
		panic("invalid ConvMode")
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
func (t *TensorDesc) Set4D(format TensorFormat, dataType DataType,
	n, c, h, w int) error {
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
func (t *TensorDesc) Get4D() (dataType DataType, n, c, h, w,
	nStride, cStride, hStride, wStride int, err error) {
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

// A FilterDesc describes the layout of a set of
// convolutional filters in memory.
type FilterDesc struct {
	desc C.cudnnFilterDescriptor_t
	ctx  *cuda.Context
}

// NewFilterDesc creates a new FilterDesc.
//
// This should be called from the cuda.Context.
func NewFilterDesc(ctx *cuda.Context) (*FilterDesc, error) {
	res := &FilterDesc{ctx: ctx}
	status := C.cudnnCreateFilterDescriptor(&res.desc)
	if err := newError("cudnnCreateFilterDescriptor", status); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(f *FilterDesc) {
		f.ctx.Run(func() error {
			C.cudnnDestroyFilterDescriptor(f.desc)
			return nil
		})
	})
	return res, nil
}

// Set4D initializes the descriptor for a 4D filter.
//
// This should be called from the cuda.Context.
func (f *FilterDesc) Set4D(dataType DataType, format TensorFormat,
	n, c, h, w int) error {
	status := C.cudnnSetFilter4dDescriptor(f.desc, dataType.cValue(), format.cValue(),
		safeIntToC(n), safeIntToC(c), safeIntToC(h), safeIntToC(w))
	return newError("cudnnSetFilter4dDescriptor", status)
}

// Get4D gets the parameters of the 4D tensor.
//
// This should be called from the cuda.Context.
func (f *FilterDesc) Get4D() (dataType DataType, format TensorFormat,
	n, c, h, w int, err error) {
	var cn, cc, ch, cw C.int
	var cType C.cudnnDataType_t
	var cFormat C.cudnnTensorFormat_t
	status := C.cudnnGetFilter4dDescriptor(f.desc, &cType, &cFormat,
		&cn, &cc, &ch, &cw)
	err = newError("cudnnGetFilter4dDescriptor", status)
	dataType = newDataTypeC(cType)
	format = newTensorFormatC(cFormat)
	n, c, h, w = int(cn), int(cc), int(ch), int(cw)
	return
}

// Set sets variable-length tensor information.
//
// This should be called from the cuda.Context.
func (f *FilterDesc) Set(dataType DataType, format TensorFormat, dims []int) error {
	cDims := make([]C.int, len(dims))
	for i, x := range dims {
		cDims[i] = safeIntToC(x)
	}
	status := C.cudnnSetFilterNdDescriptor(f.desc, dataType.cValue(), format.cValue(),
		safeIntToC(len(dims)), (*C.int)(&cDims[0]))
	return newError("cudnnSetFilterNdDescriptor", status)
}

// Get gets the variable-length tensor information.
//
// This should be called from the cuda.Context.
func (f *FilterDesc) Get() (dataType DataType, format TensorFormat, dims []int,
	err error) {
	var cType C.cudnnDataType_t
	var cFormat C.cudnnTensorFormat_t
	var nDims C.int

	// We get CUDNN_STATUS_BAD_PARAM if we ask for 0
	// dimensions like we do for TensorDesc.
	outDims := make([]C.int, 1)
	status := C.cudnnGetFilterNdDescriptor(f.desc, 1, &cType, &cFormat, &nDims,
		(*C.int)(&outDims[0]))
	err = newError("cudnnGetFilterNdDescriptor", status)
	if err != nil {
		return
	}

	outDims = make([]C.int, int(nDims))
	status = C.cudnnGetFilterNdDescriptor(f.desc, nDims, &cType, &cFormat, &nDims,
		(*C.int)(&outDims[0]))
	err = newError("cudnnGetFilterNdDescriptor", status)
	if err != nil {
		return
	}

	dataType = newDataTypeC(cType)
	format = newTensorFormatC(cFormat)
	dims = make([]int, len(outDims))
	for i := range dims {
		dims[i] = int(outDims[i])
	}
	return
}

// ConvDesc describes a convolution operation.
type ConvDesc struct {
	desc C.cudnnConvolutionDescriptor_t
	ctx  *cuda.Context
}

// NewConvDesc creates a new ConvDesc.
//
// This should be called from the cuda.Context.
func NewConvDesc(ctx *cuda.Context) (*ConvDesc, error) {
	res := &ConvDesc{ctx: ctx}
	status := C.cudnnCreateConvolutionDescriptor(&res.desc)
	if err := newError("cudnnCreateConvolutionDescriptor", status); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(c *ConvDesc) {
		c.ctx.Run(func() error {
			C.cudnnDestroyConvolutionDescriptor(c.desc)
			return nil
		})
	})
	return res, nil
}

// Set2D sets a 2-dimensional convolution.
//
// This should be called from the cuda.Context.
func (c *ConvDesc) Set2D(padH, padW, strideH, strideW, upscaleX, upscaleY int,
	mode ConvMode) error {
	status := C.cudnnSetConvolution2dDescriptor(c.desc, safeIntToC(padH),
		safeIntToC(padW), safeIntToC(strideH), safeIntToC(strideW),
		safeIntToC(upscaleX), safeIntToC(upscaleY), mode.cValue())
	return newError("cudnnSetConvolution2dDescriptor", status)
}

// Get2D gets info about a 2-dimensional convolution.
func (c *ConvDesc) Get2D() (padH, padW, strideH, strideW, upscaleX, upscaleY int,
	mode ConvMode, err error) {
	var cPadH, cPadW, cStrideH, cStrideW, cUpscaleX, cUpscaleY C.int
	var cMode C.cudnnConvolutionMode_t
	status := C.cudnnGetConvolution2dDescriptor(c.desc, &cPadH, &cPadW, &cStrideH,
		&cStrideW, &cUpscaleX, &cUpscaleY, &cMode)
	err = newError("cudnnGetConvolution2dDescriptor", status)
	if err != nil {
		return
	}
	mode = newConvModeC(cMode)
	padH, padW = int(cPadH), int(cPadW)
	strideH, strideW = int(cStrideH), int(cStrideW)
	upscaleX, upscaleY = int(cUpscaleX), int(cUpscaleY)
	return
}

// Set sets n-dimensional convolution information.
func (c *ConvDesc) Set(pad, stride, upscale []int, mode ConvMode, dataType DataType) error {
	if len(stride) != len(pad) || len(upscale) != len(pad) {
		panic("input slice lengths must match")
	}
	cPad := make([]C.int, len(pad))
	cStride := make([]C.int, len(stride))
	cUpscale := make([]C.int, len(upscale))
	for i, x := range pad {
		cPad[i] = safeIntToC(x)
		cStride[i] = safeIntToC(stride[i])
		cUpscale[i] = safeIntToC(upscale[i])
	}
	status := C.cudnnSetConvolutionNdDescriptor(c.desc, safeIntToC(len(pad)),
		&cPad[0], &cStride[0], &cUpscale[0], mode.cValue(), dataType.cValue())
	return newError("cudnnSetConvolutionNdDescriptor", status)
}

// Get gets n-dimensional convolution information.
func (c *ConvDesc) Get() (pad, stride, upscale []int, mode ConvMode,
	dataType DataType, err error) {
	var cType C.cudnnDataType_t
	var cMode C.cudnnConvolutionMode_t
	var trueLen C.int
	cPad := make([]C.int, 1)
	cStride := make([]C.int, 1)
	cUpscale := make([]C.int, 1)
	status := C.cudnnGetConvolutionNdDescriptor(c.desc, 1, &trueLen,
		&cPad[0], &cStride[0], &cUpscale[0], &cMode, &cType)
	err = newError("cudnnGetConvolutionNdDescriptor", status)
	if err != nil {
		return
	}
	cPad = make([]C.int, int(trueLen))
	cStride = make([]C.int, int(trueLen))
	cUpscale = make([]C.int, int(trueLen))
	status = C.cudnnGetConvolutionNdDescriptor(c.desc, trueLen, &trueLen,
		&cPad[0], &cStride[0], &cUpscale[0], &cMode, &cType)
	err = newError("cudnnGetConvolutionNdDescriptor", status)
	if err != nil {
		return
	}
	pad = make([]int, len(cPad))
	stride = make([]int, len(cStride))
	upscale = make([]int, len(cUpscale))
	for i, x := range cPad {
		pad[i] = int(x)
		stride[i] = int(cStride[i])
		upscale[i] = int(cUpscale[i])
	}
	mode = newConvModeC(cMode)
	dataType = newDataTypeC(cType)
	return
}
