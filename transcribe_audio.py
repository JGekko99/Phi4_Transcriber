import os
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from pydub import AudioSegment
import argparse

def load_model_and_processor():
    """Load the Phi-4-multimodal-instruct model and processor."""
    print("Loading model and processor...")
    model_name = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def prepare_audio(audio_path):
    """Prepare audio file for the model."""
    print(f"Processing audio file: {audio_path}")
    
    # Load audio file using librosa
    try:
        # First try with librosa which handles various formats
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = torch.tensor(audio).unsqueeze(0)
    except Exception as e:
        print(f"Librosa failed: {e}, trying with pydub...")
        # Fallback to pydub for other formats
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            audio = torch.tensor(np.array(audio_segment.get_array_of_samples()) / 32768.0).float().unsqueeze(0)
        except Exception as e2:
            raise Exception(f"Failed to load audio file with both librosa and pydub: {e2}")
    
    return audio

def transcribe_audio(model, processor, audio_path, language="italian"):
    """Transcribe audio file using Phi-4-multimodal-instruct model."""
    audio = prepare_audio(audio_path)
    
    # Prepare prompt for the model
    prompt = f"Transcribe the following audio in {language}:"
    
    # Process inputs
    inputs = processor(
        text=prompt,
        audio=audio,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # Decode the generated text
    transcription = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the transcription part (remove the prompt)
    transcription = transcription.replace(prompt, "").strip()
    
    return transcription

def process_directory(model, processor, directory_path, output_dir=None, language="italian"):
    """Process all audio files in a directory."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if it's a file and has an audio extension
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in audio_extensions):
            try:
                print(f"\nProcessing {filename}...")
                transcription = transcribe_audio(model, processor, file_path, language)
                
                # Save transcription
                if output_dir:
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_dir, f"{base_name}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    print(f"Transcription saved to {output_path}")
                else:
                    print(f"Transcription for {filename}:\n{transcription}\n")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe Italian audio files using Phi-4-multimodal-instruct model")
    parser.add_argument("input", help="Path to audio file or directory containing audio files")
    parser.add_argument("--output-dir", "-o", help="Directory to save transcriptions (optional)")
    parser.add_argument("--language", "-l", default="italian", help="Language for transcription (default: italian)")
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor = load_model_and_processor()
    
    # Process input (file or directory)
    if os.path.isdir(args.input):
        process_directory(model, processor, args.input, args.output_dir, args.language)
    elif os.path.isfile(args.input):
        transcription = transcribe_audio(model, processor, args.input, args.language)
        
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}.txt")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"Transcription saved to {output_path}")
        else:
            print(f"Transcription:\n{transcription}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()