import gc
from pathlib import Path


from llama_cpp import Llama
import torch
import whisper

from services.config import Config

models_root = Path(__file__).parent.parent.joinpath('models')
models_root.mkdir(parents=True, exist_ok=True)

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

def transcribe(audio_file_path: Path) -> str:
    model = whisper.load_model('large-v3', device=f'cuda:{get_ordered_cuda_devices()[0]}', download_root=str(models_root.joinpath('whisper')), in_memory=True)
    result = model.transcribe(str(audio_file_path))

    # Useless?
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result['text']

def summarize(text: str, output_language: str) -> str:
    model_name = 'bartowski/Mistral-Nemo-Instruct-2407-GGUF'
    model = Llama.from_pretrained(
        model_name,
        cache_dir=models_root,
        filename='Mistral-Nemo-Instruct-2407-Q6_K_L.gguf',
        # verbose=False,
        n_gpu_layers=-1,
        n_ctx=10*1024,
        # main_gpu=1,
        main_gpu=int(get_ordered_cuda_devices()[0]),
    )

    raw_prompt = Config().prompts.summarize.format(output_language=output_language, text=text)
    output = model(prompt=f'[INST]{raw_prompt}[/INST]', echo=True, max_tokens=None)

    summary = output['choices'][0]['text']
    e_index = summary.find('[/INST]')
    summary = summary[e_index+7:]

    return summary
