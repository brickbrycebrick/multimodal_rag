import whisper  # Installed via pip install openai-whisper - https://github.com/openai/whisper
import torch
import os
import json

class AudioTranscriber:
    def __init__(self, input_folder, output_folder="./data/transcriptions"):
        self.input_folder = os.path.abspath(input_folder)
        self.output_folder = os.path.abspath(output_folder)
        self.audio_files_dict = {}
        self.transcriptions_dict = {}
        self.whisper_model = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Scan for audio files
        self._scan_audio_files()
    
    def _scan_audio_files(self):
        """Scan the input folder for audio files and populate audio_files_dict"""
        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
            
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.mp3'):
                filepath = os.path.join(self.input_folder, filename)
                # Use filename as key since we don't have URLs
                self.audio_files_dict[filename] = filepath
                
        print(f"Found {len(self.audio_files_dict)} audio files")
        
    def transcribe_audio(self, audio_file, video_url=None):
        try:
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return None

            file_size = os.path.getsize(audio_file)
            if file_size == 0:
                print(f"Audio file is empty: {audio_file}")
                return None

            print(f"Transcribing {os.path.basename(audio_file)}...")
            result = self.whisper_model.transcribe(audio_file)
            transcription = result["text"]
            
            # Save transcription to JSON file
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            json_file = os.path.join(self.output_folder, f"{base_name}.json")
            
            # Create metadata dictionary
            metadata = {
                "video_url": video_url,
                "audio_file": audio_file,
                "transcription": transcription
            }
            
            print(f"Saving transcription to {json_file}")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            return metadata
            
        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
            return None

    def transcribe_all_audios(self, audio_files_dict=None):
        if audio_files_dict is not None:
            self.audio_files_dict = audio_files_dict
            
        if not self.audio_files_dict:
            print("No audio files found to transcribe")
            return self.transcriptions_dict

        for key, audio_path in self.audio_files_dict.items():
            if not audio_path.endswith('.mp3'):
                print(f"Skipping non-mp3 file: {audio_path}")
                continue

            # Extract video URL from the key if it's a URL
            video_url = key if key.startswith('http') else None
            result = self.transcribe_audio(audio_path, video_url)

            if result:
                self.transcriptions_dict[key] = result
            else:
                print(f"Failed to transcribe audio: {audio_path}")

        return self.transcriptions_dict

if __name__ == "__main__":
    # Use GPU if available - https://pytorch.org/docs/stable/notes/cuda.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Whisper model - model size: "tiny", "base", "small", "medium", "large"
    # fp16: https://pytorch.org/docs/stable/notes/amp_examples.html
    model = whisper.load_model("medium", device=device)

    audio_transcriber = AudioTranscriber(input_folder="./data/audios")
    audio_transcriber.whisper_model = model

    transcriptions_dict = audio_transcriber.transcribe_all_audios()
    
    for key, info in transcriptions_dict.items():
        print(f"\nProcessed: {key}")
        print(f"Audio file: {info['audio_file']}")
