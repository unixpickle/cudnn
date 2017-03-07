package cudnn

/*
#include <cudnn.h>

cudnnConvolutionFwdAlgo_t goCudnnAlgos[] = {
	CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
	CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
	CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
	CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
	CUDNN_CONVOLUTION_FWD_ALGO_FFT,
	CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
	CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
	CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
};
*/
import "C"

import (
	"runtime"
	"time"
	"unsafe"

	"github.com/unixpickle/cuda"
)

const numAlgos = 8

// ConvFwdAlgo is an algorithm for performing forward
// convolution.
type ConvFwdAlgo int

const (
	ConvFwdAlgoImplicitGemm ConvFwdAlgo = iota
	ConvFwdAlgoImplicitPrecompGemm
	ConvFwdAlgoGemm
	ConvFwdAlgoDirect
	ConvFwdAlgoFFT
	ConvFwdAlgoFFTTiling
	ConvFwdAlgoWinograd
	ConvFwdAlgoWinogradNonfused
)

func newConvFwdAlgoC(c C.cudnnConvolutionFwdAlgo_t) ConvFwdAlgo {
	for i := 0; i < numAlgos; i++ {
		if c == C.goCudnnAlgos[i] {
			return ConvFwdAlgo(i)
		}
	}
	panic("unable to create ConvFwdAlgo")
}

func (c ConvFwdAlgo) cValue() C.cudnnConvolutionFwdAlgo_t {
	if c < 0 || c >= numAlgos {
		panic("invalid ConvFwdAlgo")
	}
	return C.goCudnnAlgos[c]
}

// ConvFwdAlgoPerf stores the results of a benchmark for a
// forward convolution algorithm.
type ConvFwdAlgoPerf struct {
	Algo   ConvFwdAlgo
	Error  error
	Time   time.Duration
	Memory uintptr
}

func newConvFwdAlgoPerfC(c *C.cudnnConvolutionFwdAlgoPerf_t) *ConvFwdAlgoPerf {
	return &ConvFwdAlgoPerf{
		Algo:   newConvFwdAlgoC(c.algo),
		Error:  newError("ConvFwdAlgoPerf", c.status),
		Time:   time.Duration(float64(time.Millisecond) * float64(c.time)),
		Memory: uintptr(c.memory),
	}
}

// FindConvFwdAlgo runs performance benchmarks on every
// forward algorithm and returns the results, sorted from
// fastest to slowest.
//
// This must be called from the cuda.Context.
//
// This may attempt to allocate various chunks of memory.
// If you want to use your own allocator, use
// FindConvFwdAlgoEx.
func (h *Handle) FindConvFwdAlgo(xDesc *TensorDesc, wDesc *FilterDesc,
	convDesc *ConvDesc, yDesc *TensorDesc) ([]*ConvFwdAlgoPerf, error) {
	outStructs := make([]C.cudnnConvolutionFwdAlgoPerf_t, numAlgos)
	var algoCount C.int
	status := C.cudnnFindConvolutionForwardAlgorithm(h.handle, xDesc.desc, wDesc.desc,
		convDesc.desc, yDesc.desc, safeIntToC(numAlgos), &algoCount, &outStructs[0])
	runtime.KeepAlive(xDesc)
	runtime.KeepAlive(wDesc)
	runtime.KeepAlive(convDesc)
	runtime.KeepAlive(yDesc)
	err := newError("cudnnFindConvolutionForwardAlgorithm", status)
	if err != nil {
		return nil, err
	}
	res := make([]*ConvFwdAlgoPerf, int(algoCount))
	for i := range res {
		res[i] = newConvFwdAlgoPerfC(&outStructs[i])
	}
	return res, nil
}

// FindConvFwdAlgoEx is like FindConvFwdAlgo, but it
// takes manually-allocated buffers.
func (h *Handle) FindConvFwdAlgoEx(xDesc *TensorDesc, x cuda.Buffer,
	wDesc *FilterDesc, w cuda.Buffer, convDesc *ConvDesc, yDesc *TensorDesc,
	y cuda.Buffer, workspace cuda.Buffer) ([]*ConvFwdAlgoPerf, error) {
	var status C.cudnnStatus_t
	var algoCount C.int
	outStructs := make([]C.cudnnConvolutionFwdAlgoPerf_t, numAlgos)
	x.WithPtr(func(xPtr unsafe.Pointer) {
		w.WithPtr(func(wPtr unsafe.Pointer) {
			y.WithPtr(func(yPtr unsafe.Pointer) {
				workspace.WithPtr(func(workspacePtr unsafe.Pointer) {
					status = C.cudnnFindConvolutionForwardAlgorithmEx(
						h.handle, xDesc.desc, xPtr, wDesc.desc, wPtr,
						convDesc.desc, yDesc.desc, yPtr,
						safeIntToC(numAlgos), &algoCount, &outStructs[0],
						workspacePtr, C.size_t(workspace.Size()))
					runtime.KeepAlive(xDesc)
					runtime.KeepAlive(wDesc)
					runtime.KeepAlive(convDesc)
					runtime.KeepAlive(yDesc)
				})
			})
		})
	})
	err := newError("cudnnFindConvolutionForwardAlgorithmEx", status)
	if err != nil {
		return nil, err
	}
	res := make([]*ConvFwdAlgoPerf, int(algoCount))
	for i := range res {
		res[i] = newConvFwdAlgoPerfC(&outStructs[i])
	}
	return res, nil
}
