package cudnn

import "testing"

func TestFindConvFwdAlgo(t *testing.T) {
	ctx, handle := setupTest(t)
	<-ctx.Run(func() error {
		xDesc, err := NewTensorDesc(ctx)
		if err != nil {
			t.Error(err)
			return nil
		}
		if err := xDesc.Set4D(TensorNHWC, Float, 1, 3, 224, 224); err != nil {
			t.Error(err)
			return nil
		}
		wDesc, err := NewFilterDesc(ctx)
		if err != nil {
			t.Error(err)
			return nil
		}
		if err := wDesc.Set4D(Float, TensorNCHW, 16, 3, 3, 3); err != nil {
			t.Error(err)
			return nil
		}
		yDesc, err := NewTensorDesc(ctx)
		if err != nil {
			t.Error(err)
			return nil
		}
		if err := yDesc.Set4D(TensorNHWC, Float, 1, 16, 222, 222); err != nil {
			t.Error(err)
			return nil
		}
		convDesc, err := NewConvDesc(ctx)
		if err != nil {
			t.Error(err)
			return nil
		}
		if err := convDesc.Set2D(0, 0, 1, 1, 1, 1, Convolution); err != nil {
			t.Error(err)
			return nil
		}
		_, err = handle.FindConvFwdAlgo(xDesc, wDesc, convDesc, yDesc)
		if err != nil {
			t.Error(err)
		}
		return nil
	})
}
