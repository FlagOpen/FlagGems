import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems

# flag_gems.enable()
unused = ['abs', 'add.Tensor', 'addmm', 'arange.start_step', 'arange.start', 'arange', 'batch_norm', 'bitwise_and.Tensor',
          'bitwise_and.Scalar', 'bitwise_and.Scalar_Tensor', 'bitwise_not', 'bitwise_or.Tensor', 'bitwise_or.Scalar',
          'bitwise_or.Scalar_Tensor', 'bmm', 'clamp', 'clamp.Tensor', 'cos', 'pad', 'constant_pad_nd', 'cumsum', 'cummin',
          'div.Tensor', 'div.Scalar', 'div.Tensor_mode', 'div.Scalar_mode', 'divide.Tensor', 'divide.Scalar',
          'divide.Tensor_mode', 'divide.Scalar_mode', 'true_divide.Tensor', 'true_divide.Scalar', 'floor_divide',
          'floor_divide.Scalar', 'remainder.Tensor', 'native_dropout', 'erf', 'embedding', 'eq.Tensor', 'eq.Scalar', 'exp',
          'exponential_', 'ge.Tensor', 'ge.Scalar', 'gelu', 'native_group_norm', '_weight_norm_interface', '_weight_norm',
          'gt.Tensor', 'gt.Scalar', 'instance_norm', 'isfinite', 'isin.Tensor_Tensor', 'isin.Scalar_Tensor',
          'isin.Tensor_Scalar', 'isinf', 'isnan', 'minimum', 'maximum', 'native_layer_norm', 'le.Tensor', 'le.Scalar',
          'lt.Tensor', 'lt.Scalar', 'rms_norm', 'rand', 'randn', 'rand_like', 'randn_like', 'zeros', 'ones', 'full',
          'zeros_like', 'ones_like', 'full_like', 'resolve_neg', 'resolve_conj', 'normal.Tensor_float', 'normal.float_Tensor',
          'normal.Tensor_Tensor', 'uniform_', 'mean', 'mean.dim', 'mm', 'mul.Tensor', 'multinomial', 'mv', 'ne.Tensor',
          'ne.Scalar', 'neg', 'pow.Scalar', 'pow.Tensor_Scalar', 'pow.Tensor_Tensor', 'reciprocal', 'relu', 'rsqrt',
          'sigmoid', 'silu', 'sin', 'softmax.int', 'sort', 'sub.Tensor', 'tanh', 'triu', 'var_mean.correction',
          'linalg_vector_norm', 'where.self_out', 'where.self', 'where.ScalarSelf', 'where.ScalarOther', 'max', 'max.dim',
          'min', 'min.dim', 'amax', 'argmax', 'argmin', 'prod', 'prod.dim_int', 'sum', 'sum.dim_IntList',
          'scaled_dot_product_attention', 'all', 'all.dim', 'all.dims', 'any', 'any.dim', 'any.dims', 'quantile',
          'log_softmax.int', 'outer', 'cross_entropy_loss', 'nll_loss_forward', 'nll_loss_backward', 'nll_loss2d_forward',
          'nll_loss2d_backward', 'scatter.src', 'scatter.reduce', 'gather', 'gather_backward', 'isclose', 'allclose',
          'fill.Scalar', 'fill.Tensor', 'flip', 'slice_scatter', 'select_scatter', 'index_select', 'tile',
          'masked_fill.Tensor', 'masked_fill.Scalar', 'masked_fill_.Tensor', 'masked_fill_.Scalar', '_unique2',
          '_upsample_bicubic2d_aa', 'upsample_nearest2d', 'nonzero', 'repeat', 'masked_select', 'stack', 'hstack', 'cat',
          'repeat_interleave.self_int', 'vstack', 'repeat_interleave.Tensor', 'repeat_interleave.self_Tensor', 'randperm',
          'diag', 'diag_embed', 'diagonal_backward', 'index_add', 'count_nonzero', 'logical_or', 'logical_and', 'logical_xor',
          'logical_not', 'index_put', 'log_sigmoid', 'vdot', 'mse_loss']

# print(flag_gems.all_ops())

device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("/Users/kongcw/workspace/Qwen2-7B")
model = AutoModelForCausalLM.from_pretrained("/Users/kongcw/workspace/Qwen2-7B")
model.to(device).eval()

def test_accuracy_Qwen(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profiler:
    # with torch.no_grad():
    #     ref_output = model.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.8, repetition_penalty=1.2,
    #                                 do_sample=True, num_beams=3, early_stopping=True)
    # print(profiler.key_averages().table(sort_by="cpu_time_total"))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profiler:
        with flag_gems.use_gems():
            with torch.no_grad():
                res_output = model.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.8, repetition_penalty=1.2,
                                                do_sample=True, num_beams=3, early_stopping=True)
    print(profiler.key_averages().table(sort_by="cpu_time_total"))

    generated_text = tokenizer.decode(res_output[0], skip_special_tokens=True)
    print(generated_text)

    # maxdiff = torch.max(torch.abs(ref_output - res_output))
    # assert torch.allclose(
    #     ref_output,
    #     res_output,
    #     atol=1e-3,
    #     rtol=1e-3,
    # ), f"Qwen FAIL with maxdiff {maxdiff} \nREF: {ref_output}\nRES: {res_output}"

print("prompt1:")
test_accuracy_Qwen("How are you today?")

# print("prompt2:")
# test_accuracy_Qwen("What is your name?")

# print("prompt3:")
# test_accuracy_Qwen("Who are you?")

# print("prompt4:")
# test_accuracy_Qwen("Where are you from?")