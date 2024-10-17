from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Prompts:
    summarize: str|None

@dataclass
class Models:
    whisper: str
    pyannote: str
    llama_summarize: str

class Config:
    _instance = None
    _initialized = False

    prompts: Prompts
    models: Models

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs) -> None:
        if self._initialized is True:
            return
    
        self._initialized = True

        with Path(__file__).parent.parent.joinpath('config', 'config.yml').open('r') as f:
            raw_prompts = yaml.safe_load(f)

            prompts = Prompts(summarize=raw_prompts['config']['prompts']['summarize'])
            models = Models(
                whisper=raw_prompts['config']['models']['whisper'],
                pyannote=raw_prompts['config']['models']['pyannote'],
                llama_summarize={
                    'model_name': raw_prompts['config']['models']['llama_summarize']['model_name'],
                    'file_name': raw_prompts['config']['models']['llama_summarize']['file_name'],
                    'starting_context_size': raw_prompts['config']['models']['llama_summarize']['starting_context_size'],
                }
            )

            self.prompts = prompts
            self.models = models
            