import torch
import numpy as np
import pytest
import triton

# set python path to current dir, instead of installed flag_gems
import sys
import os

# Ensure the project 'src' root is on sys.path so we can import 'flag_gems' package locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flag_gems.ops.attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)


if __name__ == "__main__": 
    batch = 1
    q_num_head = 8
    kv_num_head = 8

    seq_len = 64
    head_size = 128

    torch.manual_seed(0)

    np.random.seed(0)
    np_query = np.random.uniform(-0.2, 0.2, (batch, q_num_head, seq_len, head_size))
    np_key = np.random.uniform(-0.2, 0.2, (batch, kv_num_head, seq_len, head_size))
    np_value = np.random.uniform(-0.2, 0.2, (batch, kv_num_head, seq_len, head_size))

    query = torch.tensor(np_query, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.tensor(np_key, device="cuda", dtype=torch.float16, requires_grad=True)
    value = torch.tensor(np_value, device="cuda", dtype=torch.float16, requires_grad=True)

    is_causal = True 
    sm_scale = 1.3
    triton_result = scaled_dot_product_attention(query, key, value, None, is_causal=is_causal, scale=sm_scale, enable_gqa=True)


    torch_result = torch.nn.functional.scaled_dot_product_attention(
        query, 
        key, 
        value, 
        scale=sm_scale,
        is_causal=is_causal,
        enable_gqa=True
    )

    #print("triton result is: ", triton_result)
    #print("torch result is: ", torch_result)

    ###########################################
    #                   bwd                   #
    ###########################################

    dout = torch.randn_like(query)

    torch_result.backward(dout)
    torch_q_grad = query.grad.clone() if query.grad is not None else None
    torch_k_grad = key.grad.clone() if key.grad is not None else None 
    torch_v_grad = value.grad.clone() if value.grad is not None else None
    
    # torch.save(torch_q_grad, "/tmp/saved/wip/dout.pt")
    # torch.save(torch_q_grad, "/tmp/saved/wip/torch_query_grad.pt")
    # torch.save(torch_k_grad, "/tmp/saved/wip/torch_key_grad.pt")
    # torch.save(torch_v_grad, "/tmp/saved/wip/torch_value_grad.pt")

    # Clear gradients
    query.grad = None
    key.grad = None 
    value.grad = None

    #triton_result.backward(dout)
    # triton_q_grad = query.grad.clone() if query.grad is not None else None
    # triton_k_grad = key.grad.clone() if key.grad is not None else None
    # triton_v_grad = value.grad.clone() if value.grad is not None else None
    triton_q_grad, triton_k_grad, triton_v_grad, _, _, _ = \
    scaled_dot_product_attention_backward(dout, query, key, value, triton_result, None, None, is_causal=is_causal, scale=sm_scale, enable_gqa=True)
    
    # torch.save(triton_q_grad, "/tmp/saved/wip/triton_q_grad.pt")
    # torch.save(triton_k_grad, "/tmp/saved/wip/triton_k_grad.pt")
    # torch.save(triton_v_grad, "/tmp/saved/wip/triton_v_grad.pt")

    # Clear gradients  
    query.grad = None
    key.grad = None
    value.grad = None 

    # dv
    print("torch value grad is: ", torch_v_grad)
    print("triton value grad is: ", triton_v_grad)
    print("nan in triton value grad: ", torch.isnan(triton_v_grad).any())
    print("value grad diff is : ", triton_v_grad - torch_v_grad )
    print("torch value grad shape is: ", torch_v_grad.shape)
    print("triton value grad shape is: ", triton_v_grad.shape)
    print("nan count by head: ", torch.isnan(torch_v_grad - triton_v_grad).sum(dim=(0, 2, 3)))
    print("abs error by head: ", torch.abs(torch_v_grad - triton_v_grad).sum(dim=(0, 2, 3)))
    print("abs error by seq: ", torch.abs(torch_v_grad - triton_v_grad).sum(dim=(0, 1, 3)))
    print("abs error of head 0: ", torch.abs(torch_v_grad[:,0,:,:] - triton_v_grad[:,0,:,:]).sum(-1))
    print("abs error of head 1: ", torch.abs(torch_v_grad[:,1,:,:] - triton_v_grad[:,1,:,:]).sum(-1))
    print("abs error by seq: ", torch.abs(torch_v_grad - triton_v_grad).sum(dim=(0, 1, 3)))
    torch.testing.assert_close(torch_v_grad, triton_v_grad, atol=2e-3, rtol=2e-3)

    # dk
    # print("torch key grad is: ", torch_k_grad)
    # print("triton key grad is: ", triton_k_grad)
    # print("nan in triton key grad: ", torch.isnan(triton_k_grad).any())
    # print("key grad diff is : ", triton_k_grad - torch_k_grad )
    # print("torch key grad shape is: ", torch_k_grad.shape)
    # print("triton key grad shape is: ", triton_k_grad.shape)
    # print("nan count by head: ", torch.isnan(torch_k_grad - triton_k_grad).sum(dim=(0, 2, 3)))
    # print("abs error by head: ", torch.abs(torch_k_grad - triton_k_grad).sum(dim=(0, 2, 3)))
    # print("abs error by seq: ", torch.abs(torch_k_grad - triton_k_grad).sum(dim=(0, 1, 3)))
    # print("abs error by head: ", torch.abs(torch_k_grad - triton_k_grad).sum(dim=(0, 2, 3)))
    # print("abs error of head 0: ", torch.abs(torch_k_grad[:,0,:,:] - triton_k_grad[:,0,:,:]))
    # print("abs error of head 1: ", torch.abs(torch_k_grad[:,1,:,:] - triton_k_grad[:,1,:,:]))
    # print("abs error by seq: ", torch.abs(torch_k_grad - triton_k_grad).sum(dim=(0, 1, 3)))
    # torch.testing.assert_close(torch_k_grad, triton_k_grad, atol=2e-3, rtol=2e-3)



    # dq
    # print("torch query grad is: ", torch_q_grad)
    # print("triton query grad is: ", triton_q_grad)
    # print("nan in triton query grad: ", torch.isnan(triton_q_grad).any())
    # print("query grad diff is : ", triton_q_grad - torch_q_grad )
    # print("diff q grad by head: ", (triton_q_grad - torch_q_grad).abs().sum(dim=(0, 2, 3)))
    # print("diff q grad by seq id: ", (triton_q_grad - torch_q_grad).abs().sum(dim=(0, 1, 3)))
    # print("nan count by head: ", torch.isnan(triton_q_grad - torch_q_grad).sum(dim=(0, 2, 3)))
    # print("abs error by head: ", torch.abs(torch_q_grad - triton_q_grad).sum(dim=(0, 2, 3)))
    # print("abs error of head 0: ", torch.abs(torch_q_grad[:,0,:,:] - triton_q_grad[:,0,:,:]))
    # print("abs error of head 1: ", torch.abs(torch_q_grad[:,1,:,:] - triton_q_grad[:,1,:,:]))
    # print("abs error by seq: ", torch.abs(torch_q_grad - triton_q_grad).sum(dim=(0, 1, 3)))
    # torch.testing.assert_close(torch_q_grad, triton_q_grad, atol=2e-3, rtol=2e-3)
