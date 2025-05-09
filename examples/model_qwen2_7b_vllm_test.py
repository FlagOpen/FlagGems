import pytest
import torch
from vllm import LLM, SamplingParams

import flag_gems

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
llm = LLM(
    model="/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen2-7B",
    max_model_len=256,
    enforce_eager=True,
)


@pytest.mark.parametrize(
    "prompt",
    ["How are you today?", "What is your name?", "Who are you?", "Where are you from?"],
)
def test_accuracy_Qwen(prompt):
    ref_output = llm.generate(prompt, sampling_params)

    with flag_gems.use_gems():
        res_output = llm.generate(prompt, sampling_params)

    generated_text_ref = ref_output[0].outputs[0].text
    generated_text_res = res_output[0].outputs[0].text

    print(f"Prompt:          {prompt!r}")
    print(f"Output torch:    {generated_text_ref!r}")
    print(f"Output FlagGems: {generated_text_res!r}")

    ref_output_tensor = torch.tensor(ref_output[0].outputs[0].token_ids)
    res_output_tensor = torch.tensor(res_output[0].outputs[0].token_ids)

    maxdiff = torch.max(torch.abs(ref_output_tensor - res_output_tensor))
    assert torch.allclose(
        ref_output_tensor,
        res_output_tensor,
        atol=1e-3,
        rtol=1e-3,
    ), f"Qwen FAIL with maxdiff {maxdiff} \nREF: {ref_output_tensor}\nRES: {res_output_tensor}"
