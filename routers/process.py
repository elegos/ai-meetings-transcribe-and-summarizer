from asyncio import sleep
from pathlib import Path
from fastapi import BackgroundTasks, UploadFile, WebSocket
from fastapi.routing import APIRouter

from services.process import process_file
from services.task_manager import TaskManager, TaskState

upload_path = Path(__file__).parent.parent.joinpath('uploads')
if upload_path.exists():
    for child in upload_path.iterdir():
        child.unlink()
upload_path.mkdir(parents=True, exist_ok=True)

process_router = APIRouter(prefix='')

@process_router.post('/summarize')
async def summarize(file: UploadFile, output_language: str, background_tasks: BackgroundTasks):
    """
    Summarize a audio and/or video file.

    Parameters
    ----------
    file : UploadFile
        Video file to be summarized.
    output_language : str
        Language of the summarization.

    Returns
    -------
    dict
        A JSON object containing the task id.
    """

    manager = TaskManager()
    task_id = manager.create_task()

    path = upload_path.joinpath(f'{task_id}.{file.filename.split(".")[-1]}')
    with path.open('wb') as f:
        f.write(await file.read())
    
    manager.set_paths(task_id, path)
    manager.set_output_lang(task_id, output_language)
    
    background_tasks.add_task(process_file, path, task_id)

    return {'task_id': task_id}

@process_router.websocket('/ws/{task_id}')
async def websocket_task_status(websocket: WebSocket, task_id: str):
    await websocket.accept()
    while True:
        task = TaskManager().get_task(task_id)
        if task is None:
            await websocket.send_json({'error': 'Task not found'})

            break

        await websocket.send_json(task.public_dict())

        if task.status != TaskState.READY:
            await sleep(1)
        else:
            break



@process_router.get('/result')
async def get_result(task_id: str):
    manager = TaskManager()
    task = manager.get_task(task_id)

    if task is None:
        return {'error': 'Task not found'}
    
    if task.status == TaskState.READY:
        manager.remove_task(task)

    return task.public_dict()
