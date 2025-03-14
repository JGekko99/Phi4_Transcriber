# Audio Transcription with Phi-4-multimodal-instruct

This tool uses Microsoft's Phi-4-multimodal-instruct model to transcribe audio files into text. It's particularly optimized for Italian language transcription but supports other languages as well.

## Features

- Transcribe single audio files or entire directories
- Support for multiple audio formats (MP3, WAV, OGG, FLAC, M4A)
- Customizable output directory for saving transcriptions
- Language selection option (default: Italian)
- Robust audio processing with fallback mechanisms

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU recommended for faster processing

## Installation

1. Clone this repository or download the script files

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- torch (PyTorch)
- transformers (Hugging Face Transformers)
- librosa (Audio processing)
- numpy
- pydub (Additional audio format support)

## Usage

### Basic Usage

```bash
python transcribe_audio.py path/to/audio_file.mp3
```

This will transcribe the audio file and print the result to the console.

### Save Transcription to File

```bash
python transcribe_audio.py path/to/audio_file.mp3 --output-dir path/to/output_folder
```

This will save the transcription to a text file in the specified output directory.

### Process an Entire Directory

```bash
python transcribe_audio.py path/to/audio_directory --output-dir path/to/output_folder
```

This will process all supported audio files in the directory and save their transcriptions to the output folder.

### Change Language

```bash
python transcribe_audio.py path/to/audio_file.mp3 --language english
```

This will transcribe the audio in English instead of the default Italian.

## Command Line Arguments

- `input`: Path to audio file or directory containing audio files (required)
- `--output-dir`, `-o`: Directory to save transcriptions (optional)
- `--language`, `-l`: Language for transcription (default: italian)

## How It Works

The script uses Microsoft's Phi-4-multimodal-instruct model, which is a powerful multimodal model capable of processing both text and audio inputs. When you run the script:

1. The model and processor are loaded
2. Audio is prepared and normalized to the required format
3. A prompt is created asking the model to transcribe the audio in the specified language
4. The model generates the transcription
5. The result is either displayed or saved to a file

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- OGG (.ogg)
- FLAC (.flac)
- M4A (.m4a)

## Notes

- First-time use will download the model (approximately 4GB)
- Transcription quality may vary depending on audio clarity and language
- Processing time depends on audio length and hardware capabilities

## License

This project uses the Phi-4-multimodal-instruct model which is subject to Microsoft's license terms.