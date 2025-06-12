import json
import logging
import os

import pytest
import torch

import flag_gems

from enum import Enum

class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"
    
    
class BenchConfig:
    def __init__(self):
        self.cpu_mode = False
        self.bench_level = BenchLevel.COMPREHENSIVE
        self.warm_up = 2
        self.repetition = 2
        self.record_log = False


Config = BenchConfig()

def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="record latency in cuda or cpu",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="Benchmark info recorded in log files or not",
    )
    
    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in BenchLevel],
        help="Specify the benchmark level: comprehensive, or core.",
    )

def pytest_configure(config):
    value = config.getoption("--mode")
    global CPU_MODE
    global Config
    CPU_MODE = value == "cpu"
    
    level_value = config.getoption("--level")
    Config.bench_level = BenchLevel(level_value)
    
    Config.record_log = config.getoption("--record") == "log"
    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-")
            file_handler = logging.FileHandler(filename, mode="w")
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(file_handler)
        else:

            logging.basicConfig(
                filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
                filemode="w",
                level=logging.INFO,
                format="[%(levelname)s] %(message)s",
            )  