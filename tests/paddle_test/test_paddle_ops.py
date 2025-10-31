import paddle
import pytest
import numpy as np
paddle.compat.enable_torch_proxy()
import flag_gems

class TestPaddleOps:

    @pytest.mark.parametrize("shape", [[6, 8], [2, 4], [10]])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("axis", [-1, 1])
    def test_softmax(self, shape, dtype, axis):

        if len(shape) <= axis and axis != -1:
            pytest.skip(f"Axis {axis} not valid for shape {shape}")
            
        x = paddle.randn(shape, dtype=dtype) * 10000
        x.stop_gradient = False

        out0 = paddle.nn.functional.softmax(x, axis=axis)
        loss0 = out0.mean()
        loss0.backward()
        grad0 = x.grad.numpy().copy()
        
        x.clear_gradient()

        with flag_gems.use_gems():
            out1 = paddle.nn.functional.softmax(x, axis=axis)
            loss1 = out1.mean()
            loss1.backward()
            grad1 = x.grad.numpy().copy()

        np.testing.assert_allclose(out0.numpy(), out1.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad0, grad1, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[2, 3, 4], [4, 6, 8]])
    def test_bmm(self, shape):

        a = paddle.randn(shape)
        b_shape = shape[:-2] + [shape[-1], shape[-1] + 1]
        b = paddle.randn(b_shape)
        
        c = paddle.bmm(a, b)
        
        with flag_gems.use_gems():
            d = paddle.bmm(a, b)

        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[2, 3, 4], [10, 20]])
    def test_sum(self, shape):

        a = paddle.randn(shape)
        c = paddle.sum(a)

        with flag_gems.use_gems():
            d = paddle.sum(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[2, 3, 4], [10, 20]])
    def test_mean(self, shape):
        a = paddle.randn(shape)
        
        c = paddle.mean(a)

        with flag_gems.use_gems():
            d = paddle.mean(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[3, 4], [2, 3, 4]])
    @pytest.mark.parametrize("diagonal", [0, 1, -1])
    def test_triu(self, shape, diagonal):
        a = paddle.randn(shape)
        
        c = paddle.triu(a, diagonal=diagonal)

        with flag_gems.use_gems():
            d = paddle.triu(a, diagonal=diagonal)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[4, 3], [2, 5]])
    def test_addmm(self, shape):
        input_shape = [shape[0], shape[1]]
        mat1_shape = [shape[0], shape[1] + 2]
        mat2_shape = [shape[1] + 2, shape[1]]
        
        a = paddle.randn(input_shape)
        b = paddle.randn(mat1_shape)
        c = paddle.randn(mat2_shape)
        
        d = paddle.addmm(a, b, c)

        with flag_gems.use_gems():
            e = paddle.addmm(a, b, c)
            
        np.testing.assert_allclose(d.numpy(), e.numpy(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("shape", [[4, 3], [2, 3, 4]])
    def test_all(self, shape):
        a = paddle.randn(shape) > 0
        c = paddle.all(a)
        with flag_gems.use_gems():
            d = paddle.all(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[4, 3], [2, 3, 4]])
    def test_amax(self, shape):
        a = paddle.randn(shape)
        
        c = paddle.amax(a)

        with flag_gems.use_gems():
            d = paddle.amax(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[4, 3], [2, 3, 4]])
    def test_any(self, shape):
        a = paddle.randn(shape) > 0
        c = paddle.any(a)
        with flag_gems.use_gems():
            d = paddle.any(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("end", [10, 20])
    def test_arange(self, end):
        a = paddle.arange(end, dtype="int64")
        with flag_gems.use_gems():
            b = paddle.arange(end, dtype="int64")
        assert (a.numpy() == b.numpy()).all()

    @pytest.mark.parametrize("shape", [[4, 3], [2, 3, 4]])
    def test_argmax(self, shape):
        a = paddle.randn(shape)
        c = paddle.argmax(a)
        with flag_gems.use_gems():
            d = paddle.argmax(a)
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[4, 3], [2, 3, 4]])
    def test_argmin(self, shape):
        a = paddle.randn(shape)
        c = paddle.argmin(a)
        with flag_gems.use_gems():
            d = paddle.argmin(a)
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    def test_batch_norm(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 2, 2])
        running_mean = paddle.zeros([2])
        running_variance = paddle.ones([2])
        weight = paddle.to_tensor([2.0, 1.5])
        bias = paddle.to_tensor([1.0, 0.5])
        
        c = paddle.nn.functional.batch_norm(
            x, running_mean, running_variance, weight, bias, training=False
        )
        
        with flag_gems.use_gems():
            d = paddle.nn.functional.batch_norm(
                x, running_mean, running_variance, weight, bias, training=False
            )

        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_cat(self, axis):
        a = paddle.randn([2, 3, 4])
        b = paddle.randn([2, 3, 4])
        
        c = paddle.concat([a, b], axis=axis)

        with flag_gems.use_gems():
            d = paddle.concat([a, b], axis=axis)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    def test_count_nonzero(self):
        a = paddle.randn([2, 3, 4])
        
        c = paddle.count_nonzero(a)
        
        with flag_gems.use_gems():
            d = paddle.count_nonzero(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[3, 4], [4, 4]])
    def test_diag(self, shape):
        a = paddle.randn(shape)
        
        c = paddle.diag(a)

        with flag_gems.use_gems():
            d = paddle.diag(a)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    def test_dot(self):
        a = paddle.randn([100])
        b = paddle.randn([100])
        
        c = paddle.dot(a, b)

        with flag_gems.use_gems():
            d = paddle.dot(a, b)
            
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    def test_embedding(self):
        vocab_size = 10
        embedding_dim = 8
        batch_size = 5
        seq_len = 3
        
        indices = paddle.randint(0, vocab_size, [batch_size, seq_len]).astype("int32")
        weight = paddle.randn([vocab_size, embedding_dim])

        c = paddle.nn.functional.embedding(indices, weight)

        with flag_gems.use_gems():
            d = paddle.nn.functional.embedding(indices, weight)
        np.testing.assert_allclose(c.numpy(), d.numpy(), rtol=1e-5, atol=1e-6)

    def test_index_add(self):
        input_tensor = paddle.ones([3, 3], dtype="float32")
        index = paddle.to_tensor([0, 2], dtype="int32")
        value = paddle.to_tensor([[1, 1, 1], [1, 1, 1]], dtype="float32")
        
        b = paddle.index_add(input_tensor, index, axis=0, value=value)

        with flag_gems.use_gems():
            c = paddle.index_add(input_tensor, dim=0, index = index, src=value)
        np.testing.assert_allclose(b.numpy(), c.numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [[2, 3, 4], [10, 20]])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_ones(self, shape, dtype):
        a = paddle.ones(shape, dtype=dtype)
        with flag_gems.use_gems():
            b = paddle.ones(shape, dtype=dtype)
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-svx", __file__, "--noconftest"])