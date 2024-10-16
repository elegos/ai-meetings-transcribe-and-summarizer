from pathlib import Path
import ffmpeg


def convert_to_audio(input_path: Path, output_path: Path) -> None:
    """
    Converts video file to audio file.

    Args:
        input_path (str): input video file path
        output_path (str): output audio file path
    """

    if not input_path.exists():
        raise Exception(f'File not found: {input_path}')

    input_path_str = str(input_path.absolute())
    output_path_str = str(output_path.absolute())

    # File is already a WAV audio file, copy and exit
    probe = ffmpeg.probe(input_path_str)
    if probe['streams'][0]['codec_name'] == 'pcm_s16le':
        if input_path == output_path:
            return

        with input_path.open('rb') as input_file:
            with output_path.open('wb') as output_file:
                while True:
                    data = input_file.read(4096)
                    if not data:
                        break
                    output_file.write(data)
        return

    res, err = ffmpeg \
        .input(input_path_str) \
        .output(output_path_str, ac=1) \
        .run(overwrite_output=True)
    
    if err is not None:
        raise Exception(err)