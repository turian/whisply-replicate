# whisply-replicate

A Replicate.com service for audio transcription, translation, and speaker diarization using Whisper v3 models.

## Features

- **Multiple Whisper Models**: Support for various Whisper models including:
  - `large-v3` (default)
  - `distil-large-v3`
  - `large-v3-turbo`
  - And many more standard Whisper models

- **Advanced Audio Processing**:
  - Automatic audio format conversion and normalization
  - Support for various input audio/video formats
  - GPU-accelerated processing

- **Rich Output Options**:
  - Basic transcription (txt, json)
  - Subtitle generation (srt, vtt)
  - Speaker diarization with word-level timestamps
  - Translation to English

## Usage

The service accepts the following parameters:

### Required:
- `audio_file`: Input audio/video file to process

### Optional:
- `language`: Language code (e.g., 'en', 'fr', 'de') [default: auto-detect]
- `model`: Whisper model to use [default: 'large-v3']
- `subtitle`: Generate subtitles (.srt, .vtt) [default: false]
- `sub_length`: Words per subtitle segment [default: 5]
- `translate`: Translate to English [default: false]
- `annotate`: Enable speaker diarization [default: false]
- `num_speakers`: Number of speakers to detect [default: auto-detect]
- `hf_token`: HuggingFace token for speaker annotation
- `verbose`: Print progress during transcription [default: false]
- `post_correction`: YAML file for text corrections

### Example Usage with Cog:

```bash
# Basic transcription
cog predict -i audio_file=@path/to/audio.mp3

# Full features with speaker diarization
cog predict -i audio_file=@path/to/audio.mp3 \
           -i language=en \
           -i model=large-v3 \
           -i subtitle=true \
           -i translate=true \
           -i annotate=true \
           -i hf_token=your_token_here \
           -i num_speakers=2
```

## Output

The service returns a zip file containing:
- Transcription in requested formats (txt, json)
- Subtitle files if requested (srt, vtt)
- Speaker annotations if enabled (rttm format)
- Translated text if translation was enabled

## Technical Details

- Uses FFmpeg for audio preprocessing
- Automatic GPU detection and utilization
- Persistent model caching for faster startup
- Error handling and validation for all inputs
- Support for various audio formats through python-magic detection

## Model Caching

- Whisper v3 models are pre-downloaded during container build
- Speaker diarization models (when using `annotate=true`):
  - Require a valid HuggingFace token
  - Are cached after first use
  - Use persistent storage for subsequent runs
