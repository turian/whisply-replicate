import os
import subprocess
import tempfile
import shutil
from pathlib import Path as PathLib
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
            choices=[
                "tiny", "tiny-en", "base", "base-en", "small", "small-en",
                "distil-small-en", "medium", "medium-en", "distil-medium-en",
                "large", "large-v1", "large-v2", "distil-large-v2",
                "large-v3", "distil-large-v3", "large-v3-turbo"
            ]
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
        )
    ) -> Path:
        """Run whisply on the input audio file"""
        
        # Ensure the input file exists
        if not os.path.exists(audio_file):
            raise ValueError(f"Audio file not found: {audio_file}")
            
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested output structure
            output_path = PathLib(temp_dir) / "output"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Build command with options using absolute path
            cmd = ["whisply", "--device", "gpu", "--model", model, "--output_dir", str(output_path.absolute())]
        
        if language:
            cmd.extend(["--lang", language])
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
        # Add input file with --files flag
        cmd.extend(["--files", str(audio_file)])
        
        # Run whisply using subprocess
        try:
            # Run whisply and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Create zip file of the output directory
            zip_path = PathLib(temp_dir) / "whisply_output.zip"
            shutil.make_archive(
                str(zip_path.with_suffix('')),  # Remove .zip as make_archive adds it
                'zip',
                output_path
            )
            
            return Path(str(zip_path))
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Whisply failed: {e.stderr}")
