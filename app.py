import os
import subprocess
import tempfile
import shutil
import traceback
from pathlib import Path as PathLib
from cog import BasePredictor, Input, Path
from typing import List

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Create persistent cache directory for HF models
        os.makedirs("/root/.cache/huggingface", exist_ok=True)
        
        # Additions for diagnostics
        try:
            # Print Whisply version
            whisply_version = subprocess.run(
                ["whisply", "--version"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            print(f"Whisply version: {whisply_version}")
        except subprocess.CalledProcessError as e:
            print("Failed to get Whisply version:")
            print(e.stderr)
            raise RuntimeError(f"Failed to get Whisply version: {e.stderr}")
        
        # Print environment variables
        print(f"Environment PATH: {os.environ.get('PATH')}")
        
        # Verify /src directory existence and permissions
        if not PathLib("/src").exists():
            raise RuntimeError("The /src directory does not exist.")
        if not os.access("/src", os.W_OK):
            raise RuntimeError("No write permission to /src directory.")
        print("/src directory exists and is writable.")

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
        """Run Whisply on the input audio file"""

        # Print input parameters for diagnostics
        print("Prediction parameters:")
        print(f"  audio_file: {audio_file}")
        print(f"  language: {language}")
        print(f"  model: {model}")
        print(f"  subtitle: {subtitle}")
        print(f"  sub_length: {sub_length}")
        print(f"  translate: {translate}")
        print(f"  annotate: {annotate}")
        print(f"  num_speakers: {num_speakers}")
        print(f"  hf_token: {hf_token}")
        print(f"  verbose: {verbose}")
        print(f"  post_correction: {post_correction}")

        # Ensure the input file exists
        if not os.path.exists(audio_file):
            raise ValueError(f"Audio file not found: {audio_file}")
        else:
            print(f"Using audio file: {audio_file}")

        # Create a temporary directory for outputs under /src
        output_dir = PathLib(tempfile.mkdtemp(prefix="whisply_", dir="/src"))
        print(f"Temporary output directory created at: {output_dir}")

        try:
            # Build command with options
            cmd = ["whisply", "--device", "gpu", "--model", model, "--output_dir", str(output_dir)]
            
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

            # Add this print statement for diagnostics
            print(f"Executing command: {' '.join(cmd)}")

            # Run Whisply and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Add these print statements for diagnostics
            print("Whisply STDOUT:")
            print(result.stdout)
            print("Whisply STDERR:")
            print(result.stderr)

            # List contents of the output directory for diagnostics
            output_files = list(output_dir.iterdir())
            print("Output directory contents:")
            for f in output_files:
                print(f)
            
            # Verify Whisply output exists
            if not output_files:
                raise RuntimeError("No output files were generated by Whisply")

            # Create zip file
            zip_filename = "whisply_output.zip"
            final_path = PathLib(os.getcwd()) / zip_filename
            
            # Ensure any existing zip is removed
            if final_path.exists():
                final_path.unlink()
                
            # Create zip archive
            shutil.make_archive(
                str(final_path.with_suffix('')),
                'zip',
                root_dir=str(output_dir),
                base_dir='.'
            )
            
            # Verify zip was created and has content
            if not final_path.exists() or final_path.stat().st_size < 100:
                raise RuntimeError("Failed to create valid zip file")
                
            return Path(final_path)
            
        except subprocess.CalledProcessError as e:
            # Print error details for diagnostics
            print("Whisply failed with the following error:")
            print(e.stderr)
            raise RuntimeError(f"Whisply failed: {e.stderr}")
        except Exception as e:
            print("An unexpected error occurred:")
            traceback.print_exc()
            raise RuntimeError(f"Error processing output: {str(e)}")
        finally:
            # Clean up the temporary directory
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
