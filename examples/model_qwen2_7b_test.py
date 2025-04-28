import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen2-7B"
)
model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen2-7B"
)
model.to(device).eval()


@pytest.mark.parametrize(
    "prompt",
    ["What is your name?", "Where are you from?"],
)
def test_accuracy_Qwen(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)

    with torch.no_grad():
        ref_output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True,
            num_beams=3,
            early_stopping=True,
        )

    with flag_gems.use_gems():
        with torch.no_grad():
            res_output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.8,
                repetition_penalty=1.2,
                do_sample=True,
                num_beams=3,
                early_stopping=True,
            )

    generated_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)
    print(generated_text)
    generated_text = tokenizer.decode(res_output[0], skip_special_tokens=True)
    print(generated_text)

    maxdiff = torch.max(torch.abs(ref_output - res_output))
    assert torch.allclose(
        ref_output,
        res_output,
        atol=1e-3,
        rtol=1e-3,
    ), f"Qwen FAIL with maxdiff {maxdiff} \nREF: {ref_output}\nRES: {res_output}"
