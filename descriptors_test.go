package cudnn

import "testing"

func TestTensorDesc(t *testing.T) {
	ctx := setupTest(t)
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
