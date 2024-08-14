import torch
import time
import triton
import copy, pytest
import flag_gems
from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    CPU_MODE,
)
from PIL import Image
import requests
from transformers import AutoTokenizer, BertConfig, BertModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration

WARMUP = 100
REPETITION = 1000
torch.backends.mlu.matmul.allow_tf32 = False


class ModelBenchmark:
    def __init__(
        self,
        model_name,
        model_ops,
        arg_func,
        dtype,
        batch,
        inputs,
        kwargs_func=None,
        model_runner=lambda x: x,
    ):
        self.model_name = model_name
        self.model_ops = model_ops
        self.model_runner = model_runner
        self.gems_op = None
        self.arg_func = arg_func
        self.kwargs_func = kwargs_func
        self.dtype = dtype
        self.batch = batch
        self.inputs = inputs

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def profile(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        if CPU_MODE:
            for i in range(WARMUP):
                fn()
            torch.mlu.synchronize()
            start = time.time()
            for i in range(REPETITION):
                fn()
            torch.mlu.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=WARMUP,
                rep=REPETITION,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def run(self):
        run_model = self.model_ops
        speedup = 0
        for input in self.inputs:
            args = ()
            if self.arg_func is not None:
                args = self.arg_func(self.batch, input, self.dtype)

            kwargs = {}
            if self.kwargs_func is not None:
                kwargs = self.kwargs_func(self.batch, input, self.dtype)
            # do perf torch
            with torch.no_grad():
                tep_args = copy.deepcopy(args)
                tep_kargs = copy.deepcopy(kwargs)
                torch_perf = self.profile(
                    self.model_runner(run_model), *tep_args, **tep_kargs
                )
            # perf gems
            if self.gems_op:
                tep_args = copy.deepcopy(args)
                tep_kargs = copy.deepcopy(kwargs)
                gems_perf = self.profile(
                    self.model_runner(self.gems_op), *tep_args, **tep_kargs
                )
            else:
                with flag_gems.use_gems():
                    with torch.no_grad():
                        tep_args = copy.deepcopy(args)
                        tep_kargs = copy.deepcopy(kwargs)
                        gems_perf = self.profile(
                            self.model_runner(run_model), *tep_args, **tep_kargs
                        )
            spd = torch_perf / gems_perf
            speedup += spd
        speedup /= len(self.inputs)
        print(
            f"\nOperator_Speedup_Test_Result({str(self.dtype)}):\t{self.model_name}\t{speedup}"
        )


PROMPTS = [
    "How are you today?",
    "What is your name?",
    "Who are you?",
    "Where are you from?",
]


# model bert test
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_model_bert(dtype):
    config = BertConfig()
    model = BertModel(config)
    model.to("cuda").to(dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def token_kargs(batch, inpstr, dtype):
        inputs = tokenizer(inpstr, return_tensors="pt").to("cuda").to(dtype)
        return inputs

    bench = ModelBenchmark(
        model_name="model_bert",
        model_ops=model,
        arg_func=None,
        dtype=dtype,
        batch=1,
        inputs=PROMPTS,
        kwargs_func=token_kargs,
    )
    bench.run()


# model llama test
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_model_llama(dtype):
    model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")
    model.to("cuda").to(dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")

    def model_run(model):
        return model.generate

    def token_kargs(batch, inpstr, dtype):
        inputs = tokenizer(inpstr, return_tensors="pt").to("cuda").to(dtype)
        inputs["max_length"] = 100
        inputs["num_beams"] = 5
        return inputs

    bench = ModelBenchmark(
        model_name="model_llama",
        model_ops=model,
        arg_func=None,
        dtype=dtype,
        batch=1,
        inputs=PROMPTS,
        kwargs_func=token_kargs,
        model_runner=model_run,
    )
    bench.run()


# model llava test
prompts_llava = ["USER: <image>\nWhat's the content of the image? ASSISTANT:"]
urls_llava = [
    "https://www.ilankelman.org/stopsigns/australia.jpg",
    "https://www.ilankelman.org/themes2/towerpisaleaning.jpg",
    "https://www.ilankelman.org/themes1/sunsetrainbowbb.jpg",
]

@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_model_llava(dtype):
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model.to("cuda").to(dtype).eval()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    def token_kargs(batch, inputs, dtype):
        prompt, url = inputs
        torch.manual_seed(1234)
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda").to(dtype)
        return inputs

    bench = ModelBenchmark(
        model_name="model_llava",
        model_ops=model,
        arg_func=None,
        dtype=dtype,
        batch=1,
        inputs=[(prompts_llava[0], x) for x in urls_llava],
        kwargs_func=token_kargs,
    )
    bench.run()
