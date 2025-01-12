import os
import subprocess
from cog import BasePredictor, Input, Path
from typing import List

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Set CUDA environment variables for GPU support
        os.environ["LD_LIBRARY_PATH"] = subprocess.check_output(
            "python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))'",
            shell=True
        ).decode().strip()

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
        translate: bool = Input(
            description="Translate to English",
            default=False
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
        if translate:
            cmd.append("--translate")
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
