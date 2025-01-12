import os
import subprocess
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        pass

    def predict(
        self,
        audio_file: Path = Input(description="Audio file to transcribe"),
        language: str = Input(
            description="Language code (e.g., 'en', 'fr', 'de')",
            default="en"
        ),
    ) -> str:
        """Run whisply on the input audio file"""
        
        # Ensure the input file exists
        if not os.path.exists(audio_file):
            raise ValueError(f"Audio file not found: {audio_file}")
            
        # Construct the whisply command
        cmd = [
            "whisply",
            "--model", "large-v3",
            "--language", language,
            str(audio_file)
        ]
        
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
