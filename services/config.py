from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Prompts:
    summarize: str|None

class Config:
    _instance = None

    prompts: Prompts = None

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs) -> None:
        if self.prompts is not None:
            return

        prompts = Prompts(None)

        with Path(__file__).parent.parent.joinpath('config', 'prompts.yml').open('r') as f:
            raw_prompts = yaml.safe_load(f)
            prompts.summarize = raw_prompts['prompts']['summarize']

        self.prompts = prompts
