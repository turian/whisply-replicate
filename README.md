# whisply-replicate
Whisply, but as a replicate.com service

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

The predictor will return a zip file containing all output files including:
- Transcription in multiple formats (txt, json, srt if enabled)
- Speaker annotations (if enabled)
- Translated text (if translation enabled)
