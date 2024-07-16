from PIL import Image
import torch
import pytest
import flag_gems
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

@pytest.mark.parametrize("prompt", ["USER: <image>\nWhat's the content of the image? ASSISTANT:"])
@pytest.mark.parametrize("url", ["https://www.ilankelman.org/stopsigns/australia.jpg"])
def test_accuracy_llava(prompt, url):
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    torch.manual_seed(1234)
    model.to("cuda").eval()
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device="cuda")

    with torch.no_grad():
        ref_output = model(**inputs).logits

    with flag_gems.use_gems():
        res_output = model(**inputs).logits
    
    maxdiff = torch.max(torch.abs(ref_output - res_output))
    succeed = True
    if (
        torch.allclose(
            ref_output,
            res_output,
            atol=1e-3,
            rtol=1e-3,
        )
        is False
    ):
        score = torch.nn.functional.cosine_similarity(
            ref_output.flatten(),
            res_output.flatten(),
            dim=0,
            eps=1e-6,
        )
        succeed = score >= 0.99
    assert (
        succeed
    ), f"LLAVA_{dtype} FAIL with maxdiff {maxdiff} and score {score}\nREF: {ref_output}\nRES: {res_output}"
