import asyncio
from pathlib import Path
from time import sleep
from typing import Any, Callable

from services import ai_tools, multimedia_convert
from services.task_manager import TaskManager, TaskState

def unlink(file_path: Path):
    if file_path.exists():
        file_path.unlink()

async def run_in_thread(func: Callable, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(None, func, *args, **kwargs)

async def process_file(file_path: Path, task_id: str):
    manager = TaskManager()
    while task := manager.get_next_task():
        if task is None:
            unlink(file_path)
            return

        if task.id != task_id:
            sleep(1)
            continue

        break

    if task is None:
        unlink(file_path)
        return

    # 1. convert to audio
    audio_file_path = file_path.with_suffix('.wav')
    task.status = TaskState.AUDIO_CONVERT
    await run_in_thread(multimedia_convert.convert_to_audio, file_path, audio_file_path)

    # 2. transcribe
    task.status = TaskState.TRANSCRIPTION
    task.transcription = await run_in_thread(ai_tools.transcribe, audio_file_path)

    unlink(file_path)
    unlink(audio_file_path)

    # 3. summarize
    task.status = TaskState.SUMMARIZATION
    task.summary = await run_in_thread(ai_tools.summarize, task.transcription, task.output_lang)

    # 4. ready
    task.status = TaskState.READY
