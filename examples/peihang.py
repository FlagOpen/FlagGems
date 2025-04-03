import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems

device = 'cpu'

def test_accuracy_qwen(prompt):
    tokenizer = AutoTokenizer.from_pretrained("/Users/ph/.cache/modelscope/hub/models/Qwen/Qwen2-7B")
    model = AutoModelForCausalLM.from_pretrained("/Users/ph/.cache/modelscope/hub/models/Qwen/Qwen2-7B")

    model.to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
#    with torch.profiler.profile(
   # activities=[torch.profiler.ProfilerActivity.CPU],
   # record_shapes=True,
   # profile_memory=True
   # ) as profiler:
    with torch.no_grad():
        ref_output = model.generate(**inputs, max_new_tokens=256,       # 限制输出长度
                                            temperature=0.3,                    # 低随机性
                                            top_p=0.8,                        # 聚焦高概率候选
                                            repetition_penalty=1.2,             # 严格抑制重复
                                            do_sample=True,
                                            num_beams=3,                        # 轻量级束搜索
                                            early_stopping=True                 # 提前终止
                                            )
 #   print(profiler.key_averages().table(sort_by="cpu_time_total"))
    with flag_gems.use_gems():
        with torch.no_grad():
            res_output = model.generate(**inputs, max_new_tokens=256,       # 限制输出长度
                                        temperature=0.3,                    # 低随机性
                                        top_p=0.8,                          # 聚焦高概率候选
                                        repetition_penalty=1.2,             # 严格抑制重复
                                        do_sample=True,
                                        num_beams=3,                        # 轻量级束搜索
                                        early_stopping=True                 # 提前终止
                                        )
    
    generated_text1 = tokenizer.decode(ref_output[0], skip_special_tokens=True)
    print(generated_text1)
    generated_text2 = tokenizer.decode(res_output[0], skip_special_tokens=True)
    print(generated_text2)
    maxdiff = torch.max(torch.abs(ref_output - res_output))
    assert torch.allclose(
        ref_output,
        res_output,
        atol=1e-3,
        rtol=1e-3,
    ), f"qwen FAIL with maxdiff {maxdiff} \nREF: {ref_output}\nRES: {res_output}"
    
test_accuracy_qwen("Where are you from?")
