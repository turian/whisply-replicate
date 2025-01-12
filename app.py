import os
import subprocess
from cog import BasePredictor, Input, Path
from typing import List

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        pass

    def predict(
        self,
        audio_file: Path = Input(description="Audio file to transcribe"),
        language: str = Input(
            description="Language code (e.g., 'en', 'fr', 'de')",
            default=None
        ),
        model: str = Input(
            description="Whisper model to use",
            default="large-v3-turbo",
            choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
        ),
        subtitle: bool = Input(
            description="Generate subtitles (.srt, .vtt)",
            default=False
        ),
        sub_length: int = Input(
            description="Subtitle segment length in words",
            default=5,
            ge=1
        ),
        translate: bool = Input(
            description="Translate to English",
            default=False
        ),
        annotate: bool = Input(
            description="Enable speaker annotation (requires HF token)",
            default=False
        ),
        num_speakers: int = Input(
            description="Number of speakers to annotate (auto-detection if None)",
            default=None,
            ge=2
        ),
        hf_token: str = Input(
            description="HuggingFace Access token for speaker annotation",
            default=None
        ),
        verbose: bool = Input(
            description="Print text chunks during transcription",
            default=False
        ),
        post_correction: Path = Input(
            description="Path to YAML file for post-correction",
            default=None
        ),
        export_format: str = Input(
            description="Export format",
            default="txt",
            choices=["all", "json", "txt", "rttm", "vtt", "webvtt", "srt"]
        )
    ) -> str:
        """Run whisply on the input audio file"""
        
        # Ensure the input file exists
        if not os.path.exists(audio_file):
            raise ValueError(f"Audio file not found: {audio_file}")
            
        # Build command with options
        cmd = ["whisply", "--device", "gpu", "--model", model]
        
        if language:
            cmd.extend(["--language", language])
        if subtitle:
            cmd.append("--subtitle")
            cmd.extend(["--sub_length", str(sub_length)])
        if translate:
            cmd.append("--translate")
        if annotate:
            cmd.append("--annotate")
            if num_speakers:
                cmd.extend(["--num_speakers", str(num_speakers)])
            if hf_token:
                cmd.extend(["--hf_token", hf_token])
        if verbose:
            cmd.append("--verbose")
        if post_correction:
            cmd.extend(["--post_correction", str(post_correction)])
        if export_format != "all":
            cmd.extend(["--export", export_format])
            
        cmd.append(str(audio_file))
        
        # Run whisply using subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Whisply failed: {e.stderr}")
