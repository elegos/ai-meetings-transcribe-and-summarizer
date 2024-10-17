from datetime import timedelta
import gc
from pathlib import Path
from typing import Generator


from llama_cpp import Llama
from pyannote.audio import Pipeline
import torch
import torchaudio
import whisper

from services.config import Config
from services.io_tools import unlink

models_root = Path(__file__).parent.parent.joinpath('models')
models_root.mkdir(parents=True, exist_ok=True)

def seconds_to_timedelta(seconds: int) -> str:
    return str(timedelta(seconds=seconds))

def get_ordered_cuda_devices() -> list[str]:
    '''Returns a list of ordered CUDA device indexes, ordering by total memory DESC'''
    raw_result = []

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        raw_result.append({
            'index': i,
            'name': props.name,
            'total_memory': props.total_memory,
        })
    
    raw_result.sort(reverse=True, key=lambda x: x['total_memory'])

    return [result['index'] for result in raw_result]

def transcribe(audio_file_path: Path) -> Generator[float, None, str]:
    audio, sr = torchaudio.load(audio_file_path)

    # Diarize
    pyannote_pipeline = Pipeline.from_pretrained(Config().models.pyannote, cache_dir=models_root.joinpath('pyannote')).to(torch.device(f'cuda:{get_ordered_cuda_devices()[0]}'))
    diarization = pyannote_pipeline(audio_file_path)

    del pyannote_pipeline # unload model before the end of the function
    gc.collect()
    torch.cuda.empty_cache()

    # Transcribe
    whisper_model = whisper.load_model(Config().models.whisper, device=f'cuda:{get_ordered_cuda_devices()[0]}', download_root=str(models_root.joinpath('whisper')), in_memory=True)

    # Get the first minute of conversation to get the language
    sample_audio = audio[:, :int(60 * sr)]
    tmp_path = audio_file_path.with_suffix('.tmp.wav')
    torchaudio.save(tmp_path, sample_audio, sr)
    result = whisper_model.transcribe(str(tmp_path.absolute()), task="transcribe", without_timestamps=True)
    detected_language = result.get("language", "unknown")
    unlink(tmp_path)

    raw_turns = [{ 'speaker': speaker, 'start': turn.start, 'end': turn.end } for turn, _, speaker in diarization.itertracks(yield_label=True)]
    turns = []
    for i, elem in enumerate(raw_turns):
        if i == 0 or turns[-1]['speaker'] != elem['speaker']:
            turns.append(elem)
        else:
            turns[-1]['end'] = elem['end']
    
    num_turns = len(turns)
    transcription: list[dict] = []
    for i, turn in enumerate(turns):
        speaker = turn['speaker']
        start_time, end_time = turn['start'], turn['end']
        start_sample, end_sample = int(start_time * sr), int(end_time * sr)

        # Extract the segment of the audio
        tmp_path = audio_file_path.with_suffix('.tmp.wav')
        segment_audio = audio[:, start_sample:end_sample]
        torchaudio.save(tmp_path, segment_audio, sr)

        result = whisper_model.transcribe(str(tmp_path.absolute()), condition_on_previous_text=False, language=detected_language)
        unlink(tmp_path)

        transcription.append(f'{speaker}: {result['text']}')

        yield (i+1) / num_turns

    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    return '\n\n'.join(transcription)

def summarize(text: str, output_language: str) -> str:
    model: Llama = None
    llm_chat = None
    context_size = Config().models.llama_summarize['starting_context_size']

    gc.collect()
    torch.cuda.empty_cache()

    while True:
        try:
            model = Llama.from_pretrained(
                repo_id=Config().models.llama_summarize['model_name'],
                cache_dir=models_root,
                filename=Config().models.llama_summarize['file_name'],
                # verbose=False,
                n_gpu_layers=-1,
                n_ctx=context_size*1024,
                # main_gpu=1,
                main_gpu=int(get_ordered_cuda_devices()[0]),
            )
            break
        except Exception as e:
            context_size -= 2

    

    output = model.create_chat_completion(messages=[
        {'role': 'system', 'content': Config().prompts.summarize.format(output_language=output_language)},
        {'role': 'user', 'content': text},
    ])

    summary = output['choices'][0]['message']['content']

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return summary
