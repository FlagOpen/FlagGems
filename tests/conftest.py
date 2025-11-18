import json
import logging
import os
from datetime import datetime
import pathlib

import pytest
import torch
import flag_gems

device = flag_gems.device

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_detail_and_result_{timestamp}.json"


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )
    parser.addoption(
        (
            "--mode" if flag_gems.vendor_name != "kunlunxin" else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args,
        action="store",
        default="normal",
        required=False,
        choices=["normal", "quick"],
        help="run tests on normal or quick mode",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="tests function param recorded in log files or not",
    )
    parser.addoption(
        "--compiler_mode",
        action="store",
        default="none",
        required=False,
        choices=["none", "flagtree"],
        help="use flagtree as the operator compiler.",
    )
    parser.addoption(
        "--limit-cases",
        type=int,
        default=None,
        help="only run first N parametrized cases",
    )

def _load_plan():
    path = os.path.join(os.path.dirname(__file__), "flagtreetest.json")
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    by_name = {}
    for it in data:
        key = it["id"]
        params = it.get("params")
        by_name[key] = params
    return by_name

def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"

    global QUICK_MODE
    QUICK_MODE = config.getoption("--mode") == "quick"

    global RECORD_LOG
    RECORD_LOG = config.getoption("--record") == "log"

    global COMPILER_MODE
    COMPILER_MODE = config.getoption("--compiler_mode") == "flagtree"

    if COMPILER_MODE:
        _load_plan()

    if RECORD_LOG:
        global RUNTEST_INFO, BUILTIN_MARKS, REGISTERED_MARKERS
        RUNTEST_INFO = {}
        BUILTIN_MARKS = {
            "parametrize",
            "skip",
            "skipif",
            "xfail",
            "usefixtures",
            "filterwarnings",
            "timeout",
            "tryfirst",
            "trylast",
        }
        REGISTERED_MARKERS = {
            marker.split(":")[0].strip() for marker in config.getini("markers")
        }
        cmd_args = [


            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return
    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKERS
        ]
        if len(op_marks) > 0:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                if op_mark not in RUNTEST_INFO:
                    RUNTEST_INFO[op_mark] = [params]
                else:
                    RUNTEST_INFO[op_mark].append(params)
        else:
            func_name = item.function.__name__
            logging.warning("There is no mark at {}".format(func_name))


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


test_results = {}


@pytest.hookimpl
def pytest_collection_modifyitems(config, items):
    limit = config.getoption("--limit-cases")
    if not limit:
        return

    counter = {}

    selected = []
    deselected = []

    for item in items:
        if not hasattr(item, "callspec"):
            selected.append(item)
            continue

        func_name = item.function.__name__

        # 初始化每个函数的计数器
        counter.setdefault(func_name, 0)

        if counter[func_name] < limit:
            selected.append(item)
            counter[func_name] += 1
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

@pytest.hookimpl
def pytest_generate_tests(metafunc):
    if not COMPILER_MODE:
        return

    plan = _load_plan()

    params = plan.get(metafunc.function.__name__)
    if not params:
        
        return


    cases = [params] if isinstance(params, dict) else list(params)
    if not cases:
        return

    json_keys_union = set().union(*(c.keys() for c in cases if isinstance(c, dict)))

    # 从 fixture names 里挑出和 json key 重合的（比如 shape / p / dtype）
    wanted = [a for a in metafunc.fixturenames if a in json_keys_union]

    def _canon(name, v):
        # 1) 形状参数：list -> tuple
        if name == "shape" or name.endswith("_shape"):
            if isinstance(v, (list, tuple)):
                return tuple(v)

        # 2) dtype 参数：字符串 -> torch.dtype
        if name == "dtype":
            if isinstance(v, str):
                mapping = {
                    "float32": torch.float32,
                    "float": torch.float32,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                }
                if v in mapping:
                    return mapping[v]
                else:
                    raise ValueError(f"Unknown dtype string in plan.json: {v!r}")

            # 如果本来就已经是 torch.dtype，就直接返回
            if isinstance(v, torch.dtype):
                return v

        # 3) 其他参数原样返回
        return v


    # 如果测试函数参数里没有这些字段，就退化成单参数 "param"
    if not wanted:
        metafunc.parametrize("param", cases)
        return

    argvals, ids = [], []
    for c in cases:
        if not isinstance(c, dict):
            argvals.append(tuple(c for _ in wanted))
            ids.append(f"param={c}")
            continue

        vals = tuple(_canon(name, c[name]) for name in wanted)
        argvals.append(vals)
        ids.append("-".join(f"{name}={_canon(name, c[name])}" for name in wanted))

    print("argvals=", argvals)
    print(
        "[gen]", metafunc.function.__name__,
        "wanted=", wanted,
        "n_cases=", len(cases),
        "n_argvals=", len(argvals),
        "ids=", ids,
    )
    metafunc.parametrize(wanted, argvals, ids=ids)

    
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    test_results[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    test_results[item.nodeid]["params"] = param_values
    # get all mark
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude marks，such as parametrize、skipif and so on
    exclude_marks = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "inplace"}
    operator_marks = [mark for mark in all_marks if mark not in exclude_marks]
    test_results[item.nodeid]["opname"] = operator_marks


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "call":
        test_results[report.nodeid]["result"] = report.outcome


def pytest_terminal_summary(terminalreporter):
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(test_results)
    else:
        existing_data = test_results

    with open("result.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4, default=str)
