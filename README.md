# whisply-replicate
Whisply, but as a replicate.com service

## Features

- Pre-downloaded Whisper v3 models for faster startup:
  - `large-v3`
  - `distil-large-v3`
  - `large-v3-turbo`
- Persistent HuggingFace model caching for speaker diarization
- GPU-accelerated transcription

## Usage

To run predictions locally using Cog:

```bash
# Basic transcription
cog predict -i audio_file=@path/to/audio.mp3

# With language specification (e.g. English)
cog predict -i audio_file=@path/to/audio.mp3 -i language=en

# With speaker annotation (requires HuggingFace token)
cog predict -i audio_file=@path/to/audio.mp3 -i annotate=true -i hf_token=your_token_here -i num_speakers=2

Note: The audio file will be passed to whisply using the --files flag internally.
```

## Model Caching

- Whisper v3 models are pre-downloaded during Docker build for faster startup
- When using speaker annotation (`annotate=true`):
  - You must provide a valid HuggingFace token (`hf_token`)
  - Diarization models will be downloaded on first use
  - Models are cached within the container for subsequent runs

The predictor will return a zip file containing all output files including:
- Transcription in multiple formats (txt, json, srt if enabled)
- Speaker annotations (if enabled)
- Translated text (if translation enabled)
