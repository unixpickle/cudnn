package cudnn

import "testing"

func TestTensorDesc(t *testing.T) {
	ctx, _ := setupTest(t)
	<-ctx.Run(func() error {
		makers := []func(t *TensorDesc) error{
			func(t *TensorDesc) error {
				return t.Set4D(TensorNHWC, Half, 2, 3, 384, 512)
			},
			func(t *TensorDesc) error {
				return t.Set4DEx(Half, 2, 3, 384, 512,
					512*384*3, 1, 3*512, 3)
			},
			func(t *TensorDesc) error {
				return t.Set(Half, []int{2, 3, 384, 512},
					[]int{512 * 384 * 3, 1, 3 * 512, 3})
			},
		}
		for i, maker := range makers {
			tensor, err := NewTensorDesc(ctx)
			if err != nil {
				t.Error(err)
				return nil
			}

			if err := maker(tensor); err != nil {
				t.Errorf("maker %d: %s", i, err)
				continue
			}

			dataType, n, c, h, w, sn, sc, sh, sw, err := tensor.Get4D()
			if err != nil {
				t.Error(err)
				return nil
			}

			if dataType != Half {
				t.Errorf("maker %d: bad data type: %v", i, dataType)
			}

			nums := []int{n, c, h, w, sn, sc, sh, sw}
			expected := []int{2, 3, 384, 512, 512 * 384 * 3, 1, 3 * 512, 3}
			for j, x := range expected {
				a := nums[j]
				if a != x {
					t.Errorf("maker %d: parameter %d: expected %v but got %v", i, j, x, a)
				}
			}

			dataType, dims, strides, err := tensor.Get()
			if err != nil {
				t.Error(err)
				return nil
			}

			if dataType != Half {
				t.Errorf("maker %d: bad data type: %v", i, dataType)
			}

			nums = append(dims, strides...)
			for j, x := range expected {
				a := nums[j]
				if a != x {
					t.Errorf("maker %d: parameter %d: expected %v but got %v", i, j, x, a)
				}
			}
		}
		return nil
	})
}

func TestFilterDesc(t *testing.T) {
	ctx, _ := setupTest(t)
	<-ctx.Run(func() error {
		makers := []func(t *FilterDesc) error{
			func(f *FilterDesc) error {
				return f.Set4D(Half, TensorNHWC, 2, 3, 384, 512)
			},
			func(f *FilterDesc) error {
				return f.Set(Half, TensorNHWC, []int{2, 3, 384, 512})
			},
		}
		for i, maker := range makers {
			filter, err := NewFilterDesc(ctx)
			if err != nil {
				t.Error(err)
				return nil
			}

			if err := maker(filter); err != nil {
				t.Errorf("maker %d: %s", i, err)
				continue
			}

			dataType, format, n, c, h, w, err := filter.Get4D()
			if err != nil {
				t.Error(err)
				return nil
			}

			if dataType != Half {
				t.Errorf("maker %d: bad data type: %v", i, dataType)
			}
			if format != TensorNHWC {
				t.Errorf("maker %d: bad format: %v", i, format)
			}

			nums := []int{n, c, h, w}
			expected := []int{2, 3, 384, 512}
			for j, x := range expected {
				a := nums[j]
				if a != x {
					t.Errorf("maker %d: parameter %d: expected %v but got %v", i, j, x, a)
				}
			}

			dataType, format, dims, err := filter.Get()
			if err != nil {
				t.Error(err)
				return nil
			}

			if dataType != Half {
				t.Errorf("maker %d: bad data type: %v", i, dataType)
			}
			if format != TensorNHWC {
				t.Errorf("maker %d: bad format: %v", i, format)
			}

			nums = dims
			for j, x := range expected {
				a := nums[j]
				if a != x {
					t.Errorf("maker %d: parameter %d: expected %v but got %v", i, j, x, a)
				}
			}
		}
		return nil
	})
}

func TestConvDim(t *testing.T) {
	ctx, _ := setupTest(t)
	<-ctx.Run(func() error {
		desc, err := NewConvDesc(ctx)
		if err != nil {
			t.Error(err)
			return nil
		}
		err = desc.Set2D(2, 1, 3, 4, 1, 1, CrossCorrelation)
		if err != nil {
			t.Error(err)
			return nil
		}
		padH, padW, strideH, strideW, upX, upY, mode, err := desc.Get2D()
		if err != nil {
			t.Error(err)
			return nil
		}
		if mode != CrossCorrelation {
			t.Errorf("bad mode: %v", mode)
		}
		actual := []int{padH, padW, strideH, strideW, upX, upY}
		expected := []int{2, 1, 3, 4, 1, 1}
		for i, x := range expected {
			a := actual[i]
			if a != x {
				t.Errorf("parameter %d should be %d but got %d", i, x, a)
			}
		}

		desc, err = NewConvDesc(ctx)
		if err != nil {
			t.Error(err)
			return nil
		}
		err = desc.Set([]int{2, 1}, []int{3, 4}, []int{1, 1}, CrossCorrelation, Double)
		if err != nil {
			t.Error(err)
			return nil
		}
		pad, stride, up, mode, dt, err := desc.Get()
		if err != nil {
			t.Error(err)
			return nil
		}
		if mode != CrossCorrelation {
			t.Errorf("bad mode: %v", mode)
		}
		if dt != Double {
			t.Errorf("bad data type: %v", dt)
		}
		actual = append(append(pad, stride...), up...)
		for i, x := range expected {
			a := actual[i]
			if a != x {
				t.Errorf("parameter %d should be %d but got %d", i, x, a)
			}
		}
		return nil
	})
}
