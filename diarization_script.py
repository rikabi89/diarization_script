import torch
import torchaudio
from pathlib import Path
import whisperx

# Set your Hugging Face token here
hugging_face_token = 'YOUR_TOKEN_HERE'

# Set the device (CPU or CUDA) for processing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the audio file using torchaudio
audio_file_path = Path("C:/whisperX/sample.wav")
waveform, sample_rate = torchaudio.load(audio_file_path)

# Perform ASR using the Whisper model
model = whisperx.load_model("large-v2", device, compute_type='float16')
audio = whisperx.load_audio(audio_file_path)
result = model.transcribe(audio, batch_size=16)

# Perform speaker diarization using whisperx
diarize_model = whisperx.DiarizationPipeline(use_auth_token=hugging_face_token, device=device)

# Ensure the input audio is provided as a dictionary with the correct keys
audio_input = {
    "waveform": waveform,
    "sample_rate": sample_rate
}

diarize_segments = diarize_model(audio_input,max_speakers=2,min_speakers=2 )

# Assign word speakers
diarizedData = whisperx.assign_word_speakers(diarize_segments, result)

# Extract speaker segments
speaker_segments = {}
for segment in diarizedData["segments"]:
    speaker = segment["speaker"]
    start_time = segment["start"]
    end_time = segment["end"]
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    if speaker not in speaker_segments:
        speaker_segments[speaker] = []

    speaker_segment = waveform[:, start_sample:end_sample]
    speaker_segments[speaker].append(speaker_segment)

# Save each speaker's segments as separate audio files
output_dir = Path("speaker_segments")
output_dir.mkdir(exist_ok=True)

for speaker, segments in speaker_segments.items():
    speaker_waveform = torch.cat(segments, dim=1)
    output_filename = output_dir / f"{speaker}.wav"
    torchaudio.save(output_filename, speaker_waveform, sample_rate=sample_rate)
