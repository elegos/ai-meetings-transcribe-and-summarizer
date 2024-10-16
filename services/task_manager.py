from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

class TaskState(Enum):
    INITIALIZED = "Initialized"
    PENDING = "Pending"
    AUDIO_CONVERT = "Converting in audio"
    TRANSCRIPTION = "Transcribing"
    SUMMARIZATION = "Summarizing"
    READY = "Ready"


@dataclass
class Task:
    id: str
    status: TaskState

    input_file: Path|None = field(default=None)
    output_lang: str|None = field(default=None)
    transcription: str|None = field(default=None)
    summary: str|None = field(default=None)

    def public_dict(self):
        return {
            'id': self.id,
            'status': self.status.name,
            'transcription': self.transcription,
            'summary': self.summary
        }

class TaskManager:
    _instance: 'TaskManager' = None
    task_list: list[Task] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self.task_list is None:
            self.task_list = []

    def create_task(self) -> str:
        task = Task(str(uuid.uuid4()), TaskState.INITIALIZED)

        self.task_list.append(task)

        return task.id
    
    def get_task(self, task_id: str) -> Task|None:
        for task in self.task_list:
            if task.id == task_id:
                return task
    
    def set_paths(self, task_id: str, intput_file: Path) -> None:
        task = self.get_task(task_id)

        if task is None:
            raise Exception(f'Task {task_id} not found')
        
        task.input_file = intput_file
        task.status = TaskState.PENDING
    
    def set_output_lang(self, task_id: str, output_lang: str) -> None:
        task = self.get_task(task_id)

        if task is None:
            raise Exception(f'Task {task_id} not found')
        
        task.output_lang = output_lang
    
    def get_next_task(self) -> Task|None:
        return next((task for task in self.task_list if task.status != TaskState.READY), None)

    def remove_task(self, task: str|Task) -> None:
        if isinstance(task, str):
            task = self.get_task(task)
        
        if task is not None:
            self.task_list.remove(task)