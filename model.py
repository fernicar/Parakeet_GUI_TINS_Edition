# model.py (Revised - Added Save SRT functionality and Test Level 5)

import os
import sys
import tempfile
import math
import subprocess
import time
import json # Using json for structured data output in tests
from pathlib import Path
import argparse
import re
from datetime import timedelta # For timestamp formatting

# Attempt to import NeMo and Torch. These might fail if dependencies are not met.
# Also import Hypothesis for type hinting/checking return types.
try:
    import signal
    # --- Start Monkeypatch for signal ---
    # This section is necessary because directly references signal.SIGKILL in
    # nemo.utils.exp_manager.FaultToleranceParams.rank_termination_signal
    # during class definition as if linux, which causes an ImportError on Windows.
    # We need to define a dummy SIGKILL attribute in the signal module *before*
    # importing the NeMo module.
    if sys.platform == "win32":
        # Check if SIGKILL is already defined (unlikely on standard Windows Python)
        if not hasattr(signal, 'SIGKILL'):
            # Define a dummy SIGKILL. Its actual integer value doesn't matter here
            # for the import to succeed, as long as the attribute exists.
            # Using CTRL_BREAK_EVENT value instead to pass the class definition check.
            signal.SIGKILL = signal.CTRL_BREAK_EVENT
            # Note: We could also potentially try to define it as a signal.Signals enum member
            # but setting a simple integer attribute is less complex and sufficient
            # to prevent the AttributeError during import.
    # --- End Monkeypatch for signal ---

    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis # Import Hypothesis type
    import torch
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    # print("WARNING: nemo_toolkit or torch not available. Transcription will not work.", file=sys.stderr)
    # Print handled by main.py's dependency check now

# Attempt to import pydub. This might fail if dependencies are not met.
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    # print("WARNING: pydub not available. Audio processing will not work.", file=sys.stderr)
    # Print handled by main.py's dependency check now


# --- Custom Exceptions ---

class FFmpegNotFoundError(Exception):
    """Custom exception raised when FFmpeg is not found."""
    pass

class NemoInitializationError(Exception):
    """Custom exception raised when NeMo or CUDA fails to initialize."""
    pass

class TranscriptionError(Exception):
    """Custom exception raised for errors during the transcription process."""
    pass

class InvalidAudioFileError(Exception):
    """Custom exception raised for issues loading or processing the audio file."""
    pass

class SRTParseError(Exception):
    """Custom exception raised for errors parsing an SRT file."""
    pass

class SRTWriteError(Exception):
    """Custom exception raised for errors writing an SRT file."""
    pass


# --- Helper Functions ---

def check_ffmpeg_available():
    """Checks if the ffmpeg executable is available in the system's PATH."""
    try:
        # Use a simple command that should always work and return version
        # Use creationflags=subprocess.CREATE_NO_WINDOW on Windows to avoid console window flash
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE # Hide the window

        # Use check=False so it doesn't raise CalledProcessError if ffmpeg exists but has issues (though unlikely for -version)
        # Use shell=True only if necessary and arguments are carefully quoted; better to avoid shell=True.
        # Try without shell=True first.
        process = subprocess.run(['ffmpeg', '-version'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        # Check return code and maybe stderr for signs of success
        return process.returncode == 0 # A successful run should have return code 0

    except FileNotFoundError:
        # This is the primary way we detect if ffmpeg is not in PATH
        return False
    except Exception as e:
        # Catch any other unexpected errors during the subprocess call
        print(f"DEBUG: Unexpected error during ffmpeg check: {e}", file=sys.stderr)
        return False # Assume unavailable if check fails unexpectedly

def format_srt_timestamp(seconds: float) -> str:
    """Converts seconds (float) to SRT timestamp format (HH:MM:SS,ms)."""
    if seconds < 0:
        seconds = 0 # SRT timestamps should not be negative

    # Calculate hours, minutes, seconds, and milliseconds
    # Use divmod for cleaner integer division and remainder
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Ensure integer part of seconds and milliseconds are handled
    total_milliseconds = round(seconds * 1000.0)
    seconds_part = total_milliseconds // 1000
    milliseconds_part = total_milliseconds % 1000

    # Format with zero padding
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds_part):02},{int(milliseconds_part):03}"


# --- Audio Processing and Transcription Logic ---

# Lazy load and cache the ASR model
_asr_model = None
_last_model_name = None
_model_load_error = None

def get_asr_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    """
    Loads and caches the specified NeMo ASR model.

    Returns:
        nemo_asr.models.ASRModel: The loaded ASR model instance.
    Raises:
        NemoInitializationError: If NeMo or CUDA is not available or initialization fails.
    """
    global _asr_model, _last_model_name, _model_load_error

    # If a previous load failed, re-raise the error immediately
    if _model_load_error:
        raise _model_load_error

    if _asr_model is None or _last_model_name != model_name:
        if not NEMO_AVAILABLE:
            # This specific error message is now less critical as main.py checks first,
            # but keep it here in case model.py is used standalone without the --test handler.
            _model_load_error = NemoInitializationError(
                "NeMo toolkit or PyTorch is not installed. "
                "Please install dependencies: nemo_toolkit[asr], torch (CUDA version), cuda-python."
            )
            raise _model_load_error

        print(f"Loading ASR model '{model_name}'...", file=sys.stderr)
        try:
            # Check CUDA availability before loading
            if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
                _model_load_error = NemoInitializationError(
                "CUDA is not available or detected. "
                "NeMo Parakeet model requires a CUDA-enabled GPU. "
                "Ensure you have a CUDA-enabled GPU, compatible drivers, "
                "and have installed the CUDA version of PyTorch and cuda-python."
                )
                raise _model_load_error

            # Move model to GPU
            _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            if torch.cuda.is_available():
                _asr_model = _asr_model.cuda()
            else:
                # This case should be caught by the CUDA check above, but defensive programming
                print("WARNING: CUDA not available after loading model. Model might not run on GPU.", file=sys.stderr)


            _last_model_name = model_name
            print("Model loaded successfully.", file=sys.stderr)
            return _asr_model
        except Exception as e:
            # Catch potential errors during model loading (network issues, incompatible versions etc.)
            _model_load_error = NemoInitializationError(f"Failed to load ASR model '{model_name}': {e}")
            print(f"ERROR: {_model_load_error}", file=sys.stderr)
            raise _model_load_error
    return _asr_model

def process_audio_for_nemo(audio_path: str, segment_length_sec: int = 60):
    """
    Processes audio file: loads, converts to mono 16kHz, and segments.

    Args:
        audio_path (str): Path to the input audio file.
        segment_length_sec (int): Length of each audio segment in seconds.

    Returns:
        tuple: A tuple containing:
            - list: Paths to the temporary audio segment files (WAV format).
            - float: Total duration of the original audio in seconds.
            - list: List of (start_time_sec, end_time_sec) tuples for each segment (original times).

    Raises:
        FFmpegNotFoundError: If FFmpeg is required by pydub but not found.
        InvalidAudioFileError: If the audio file cannot be loaded or processed.
    """
    if not PYDUB_AVAILABLE:
        raise InvalidAudioFileError("pydub is not installed. Please install it: pip install pydub")
    if not os.path.exists(audio_path):
        raise InvalidAudioFileError(f"Input audio file not found: {audio_path}")

    # Check FFmpeg before trying to load, as pydub needs it for many formats
    if not check_ffmpeg_available():
        raise FFmpegNotFoundError("FFmpeg executable not found. Please install FFmpeg and ensure its 'bin' directory is in your system's PATH.")

    print(f"Processing audio file: {os.path.basename(audio_path)}", file=sys.stderr)
    temp_files = []
    segment_times = [] # Store start/end times of original segments

    try:
        audio = AudioSegment.from_file(audio_path)

        if audio.channels > 1:
            print(f"Converting audio from {audio.channels} channels to mono.", file=sys.stderr)
            audio = audio.set_channels(1)

        if audio.frame_rate != 16000:
            print(f"Resampling audio from {audio.frame_rate} Hz to 16000 Hz.", file=sys.stderr)
            audio = audio.set_frame_rate(16000)

        segment_length_ms = segment_length_sec * 1000
        total_length_ms = len(audio)
        total_length_sec = total_length_ms / 1000.0
        # Ensure at least one segment even for very short audio
        num_segments = max(1, math.ceil(total_length_ms / segment_length_ms))


        print(f"Audio duration: {total_length_sec:.2f} seconds", file=sys.stderr)
        print(f"Segmentation: {num_segments} segments of up to {segment_length_sec} seconds.", file=sys.stderr)

        # Ensure temp directory exists (it usually does, but safety first)
        temp_dir = tempfile.gettempdir()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for i in range(num_segments):
            start_time_ms = i * segment_length_ms
            end_time_ms = min((i + 1) * segment_length_ms, total_length_ms)
            segment = audio[start_time_ms:end_time_ms]

            # Store original segment times in seconds
            segment_times.append((start_time_ms / 1000.0, end_time_ms / 1000.0))

            # Create a unique temporary file name using tempfile.mkstemp
            fd, temp_wav_file_path = tempfile.mkstemp(suffix=f'_{i}.wav', prefix='parakeet_gui_', dir=temp_dir)
            os.close(fd) # Close the file descriptor immediately after creation
            temp_files.append(temp_wav_file_path)

            # Export the segment to the temporary WAV file
            # Ensure output is valid WAV format for NeMo
            segment.export(temp_wav_file_path, format='wav')
            # print(f"Created temp file for segment {i+1}/{num_segments}: {Path(temp_wav_file_path).name}", file=sys.stderr) # Too verbose

        print(f"Created {len(temp_files)} temporary segment files.", file=sys.stderr)
        return temp_files, total_length_sec, segment_times

    except FFmpegNotFoundError:
        # Re-raise if it was specifically an FFmpeg error from check_ffmpeg_available
        raise
    except Exception as e:
        # Catch other audio processing errors
        print(f"ERROR during audio processing: {e}", file=sys.stderr)
        raise InvalidAudioFileError(f"Error processing audio file '{audio_path}': {e}")
    finally:
        # Cleanup temp files in case of error during creation - done in calling function
        pass


def transcribe_audio_and_time(audio_path: str, segment_length_sec: int = 60, model_name="nvidia/parakeet-tdt-0.6b-v2"):
    """
    Transcribes an audio file and returns structured timed text data.

    Args:
        audio_path (str): Path to the input audio file.
        segment_length_sec (int): Length of audio segments in seconds.
        model_name (str): Name of the NeMo ASR model to use.

    Returns:
        list: A list of structured timed text data (list of sentence dicts).
              [ {'text': '...', 'start_time': float, 'end_time': float,
                 'words': [ {'text': '...', 'start_time': float, 'end_time': float}, ... ] }, ...]

    Raises:
        FFmpegNotFoundError: If FFmpeg is required by pydub but not found.
        InvalidAudioFileError: If the audio file cannot be loaded or processed.
        NemoInitializationError: If NeMo or CUDA fails to initialize.
        TranscriptionError: If transcription fails.
    """
    # Ensure model and dependencies are available before proceeding
    # These checks are also in main.py, but keeping them here makes model.py more robust standalone.
    if not (NEMO_AVAILABLE and PYDUB_AVAILABLE and check_ffmpeg_available() and hasattr(torch, 'cuda') and torch.cuda.is_available()):
        # This case should ideally be caught by main.py's dependency checks,
        # but raising here ensures model.py is self-contained if run incorrectly.
        error_msgs = []
        if not check_ffmpeg_available(): error_msgs.append("FFmpeg not found.")
        if not PYDUB_AVAILABLE: error_msgs.append("pydub not available.")
        if not NEMO_AVAILABLE: error_msgs.append("NeMo or PyTorch not available.")
        if not (hasattr(torch, 'cuda') and torch.cuda.is_available()): error_msgs.append("CUDA not available.")
        raise NemoInitializationError("Missing required dependencies or CUDA: " + ", ".join(error_msgs))

    asr_model = get_asr_model(model_name) # Load/get the cached model

    temp_files = []
    all_timed_words = [] # Store all timed words across segments
    segment_offsets = [] # Store start time offsets for each segment in original audio

    try:
        # 1. Process Audio into Segments
        temp_files, total_duration, segment_times = process_audio_for_nemo(audio_path, segment_length_sec)
        segment_offsets = [start_time for start_time, _ in segment_times]

        if not temp_files:
            print("INFO: Audio processing resulted in no segments to transcribe.", file=sys.stderr)
            return [] # Return empty data if no segments were created

        # 2. Transcribe Segments with Timestamps
        print(f"Starting transcription of {len(temp_files)} segments...", file=sys.stderr)
        # NeMo's transcribe method can take a list of file paths
        try:
            # Use the signature observed: list of paths as first arg, timestamps=True
            # Note: This might load all segments into GPU memory at once, which can be an issue for VRAM
            # A more advanced approach would be batching/streaming, but for simplicity use list for now.
            transcription_results = asr_model.transcribe(
                temp_files,
                timestamps=True
            )
            # Based on user feedback and app.py, this returns a list of Hypothesis objects.
            # Each Hypothesis object corresponds to an audio file in the input list.

        except Exception as e:
            # Catch errors during the actual NeMo transcription call
            print(f"ERROR during NeMo transcription call: {e}", file=sys.stderr)
            # Check for common CUDA runtime errors here specifically if needed, or let them propagate
            raise TranscriptionError(f"Error during NeMo transcription: {e}")


        # Process results from each segment
        if transcription_results and isinstance(transcription_results, list):
            if len(transcription_results) != len(temp_files):
                print(f"WARNING: Number of transcription results ({len(transcription_results)}) does not match number of segments ({len(temp_files)}). This may indicate an issue.", file=sys.stderr)
                # Continue processing available results, but note mismatch.

            for i, segment_result in enumerate(transcription_results):
                # Ensure we don't go out of bounds for segment_offsets or segment_times
                if i >= len(segment_offsets) or i >= len(segment_times):
                    print(f"WARNING: Received more transcription results ({len(transcription_results)}) than expected segments ({len(segment_times)}). Skipping result {i+1}.", file=sys.stderr)
                    break
                current_offset = segment_offsets[i]
                segment_end_orig = segment_times[i][1] # End time of this segment in the original audio

                # print(f"Processing result for segment {i+1}/{len(transcription_results)} (offset: {current_offset:.3f}s)...", file=sys.stderr) # Too verbose

                # Based on Hypothesis structure and app.py example
                if isinstance(segment_result, Hypothesis):
                    try:
                        # Access timestamps from the Hypothesis object via .timestamp attribute
                        # The structure within .timestamp is a dict, word timestamps are under 'word' key
                        if hasattr(segment_result, 'timestamp') and isinstance(segment_result.timestamp, dict) and 'word' in segment_result.timestamp:
                            word_timestamps = segment_result.timestamp['word']

                            adjusted_timed_words = []
                            last_word_end = current_offset # Start with segment offset
                            for word_info in word_timestamps:
                                # Check if word_info has expected dict structure with 'word', 'start', 'end'
                                if isinstance(word_info, dict) and 'word' in word_info and 'start' in word_info and 'end' in word_info:
                                    word_start = float(word_info.get('start', last_word_end))
                                    word_end = float(word_info.get('end', word_start + 0.01)) # Ensure end >= start fallback

                                    # Adjust timestamps by the segment offset
                                    adjusted_start = word_start + current_offset
                                    adjusted_end = word_end + current_offset

                                    # Basic validation: Ensure timestamps are non-negative and somewhat reasonable
                                    if adjusted_start < 0: adjusted_start = 0.0
                                    if adjusted_end < adjusted_start: adjusted_end = adjusted_start + 0.01

                                    # Further check: Ensure word timestamps don't exceed the segment's original end time by much
                                    # This can happen with ASR models sometimes. Clamp to segment end.
                                    if adjusted_end > segment_end_orig + 0.1: # Allow a small buffer
                                        adjusted_end = segment_end_orig + 0.1
                                    if adjusted_start > adjusted_end: adjusted_start = max(0.0, adjusted_end - 0.01) # Maintain start <= end


                                    adjusted_timed_words.append({
                                        'text': str(word_info.get('word', '')), # Ensure text is string
                                        'start_time': adjusted_start,
                                        'end_time': adjusted_end
                                    })
                                    last_word_end = adjusted_end # Update for next word's fallback

                                else:
                                    print(f"  Warning: Unexpected word timestamp structure in segment {i+1} result: {word_info}. Skipping.", file=sys.stderr)

                            all_timed_words.extend(adjusted_timed_words)
                            print(f"  Segment {i+1} processed. Found {len(adjusted_timed_words)} timed words.", file=sys.stderr)

                        elif hasattr(segment_result, 'text') and segment_result.text:
                            # Fallback: If no word timestamps but plain text is available,
                            # add a dummy word entry for the whole segment text with segment times.
                            # This preserves the text but loses word timing within the segment.
                            print(f"  Warning: Hypothesis for segment {i+1} contains text but no word timestamps. Using segment timing for text.", file=sys.stderr)
                            segment_start = segment_offsets[i]
                            # Use the original segment end time
                            segment_end = segment_times[i][1]
                            all_timed_words.append({
                                'text': str(segment_result.text),
                                'start_time': float(segment_start),
                                'end_time': float(segment_end)
                            })


                        else:
                            print(f"  Warning: Hypothesis object for segment {i+1} does not contain text or word timestamps in expected format. Result structure: {dir(segment_result)}", file=sys.stderr)


                    except Exception as e:
                        print(f"ERROR processing Hypothesis object for segment {i+1}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                        # Continue to next segment


                else:
                    print(f"  Warning: Unexpected result type for segment {i+1}: {type(segment_result)}. Expected Hypothesis. Skipping.", file=sys.stderr)
                    # print(f"  Received result: {segment_result}", file=sys.stderr) # Might be too verbose


        else:
            print("WARNING: NeMo transcribe returned no results or unexpected top-level format (not a list).", file=sys.stderr)


        if not all_timed_words:
            print("WARNING: Transcription process finished but no timed words were collected.", file=sys.stderr)
            # Consider adding a placeholder sentence if some plain text was returned but not timed words
            # For now, return empty list as per spec if no timed words found.
            return []


        # 3. Structure Timed Data (into sentences with words)
        print("Structuring timed data...", file=sys.stderr)
        # Pass the collected and offset-adjusted timed words to structuring
        structured_data = structure_timed_words_into_sentences(all_timed_words)
        print(f"Finished structuring. Found {len(structured_data)} sentences.", file=sys.stderr)

        return structured_data

    except (FFmpegNotFoundError, InvalidAudioFileError, NemoInitializationError) as e:
        # Re-raise known errors for main.py to handle specifically
        raise e
    except TranscriptionError as e:
        # Re-raise transcription specific error
        raise e
    except Exception as e:
        # Catch any other unexpected transcription errors during the *process*
        print(f"ERROR during transcription process (outside NeMo call): {e}", file=sys.stderr)
        raise TranscriptionError(f"An unexpected error occurred during transcription process: {e}")
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...", file=sys.stderr)
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    #print(f"  Removed: {temp_file_path}", file=sys.stderr) # Too verbose
                except Exception as e:
                    print(f"  Warning: Failed to remove temporary file {temp_file_path}: {e}", file=sys.stderr)
        # print("Cleanup complete.", file=sys.stderr) # Less verbose


def structure_timed_words_into_sentences(timed_words: list):
    """
    Structures a flat list of timed words into a list of sentences,
    preserving word-level timing within each sentence.

    This is a heuristic approach. More advanced sentence boundary detection
    might be needed for complex cases. Assumes words with punctuation
    like '.' or '?' indicate sentence ends.

    Args:
        timed_words (list): List of word dicts: [{'text': '...', 'start_time': float, 'end_time': float}, ...]

    Returns:
        list: Structured data as list of sentence dicts.
              [ {'text': '...', 'start_time': float, 'end_time': float,
                 'words': [ {'text': '...', 'start_time': float, 'end_time': float}, ... ] }, ...]
    """
    if not timed_words:
        return []

    structured_sentences = []
    current_sentence_words = []

    for i, word_info in enumerate(timed_words):
        # Basic check for valid word info structure
        if not isinstance(word_info, dict) or 'text' not in word_info:
            print(f"WARNING: Skipping invalid word info: {word_info}", file=sys.stderr)
            continue

        current_sentence_words.append(word_info)

        # Simple heuristic: check for sentence-ending punctuation at the end of the word text
        last_word_text = word_info.get('text', '').strip()
        # Use regex to be more robust with potential trailing non-punctuation chars
        # Ensure it ends with a common sentence-ending punctuation mark followed by optional quotes/spaces/newlines etc.
        # Adding \s*?$ to handle potential trailing whitespace before end of string
        if re.search(r'[.?!;]"?\s*?$', last_word_text):
            if current_sentence_words:
                # Combine words into sentence text, preserving single space between words
                sentence_text = " ".join([w.get('text', '').strip() for w in current_sentence_words]).strip()
                # Determine sentence start/end from the first/last word in the group
                # Use .get with default and convert to float defensively
                # Ensure start/end times are calculated from the actual words in the sentence
                first_word_time = float(current_sentence_words[0].get('start_time', 0.0))
                last_word_time = float(current_sentence_words[-1].get('end_time', first_word_time + 0.1)) # Ensure end >= start fallback

                sentence_start = first_word_time
                sentence_end = last_word_time
                if sentence_end < sentence_start: sentence_end = sentence_start + 0.01


                # Only add if sentence text is not empty after joining/stripping
                if sentence_text:
                    structured_sentences.append({
                        'text': sentence_text,
                        'start_time': sentence_start,
                        'end_time': sentence_end,
                        'words': list(current_sentence_words) # Store a copy of the words list
                    })
                current_sentence_words = [] # Start new sentence

    # Add any remaining words as the last sentence
    if current_sentence_words:
        sentence_text = " ".join([w.get('text', '').strip() for w in current_sentence_words]).strip()
        if sentence_text: # Only add if there's text
            first_word_time = float(current_sentence_words[0].get('start_time', 0.0))
            last_word_time = float(current_sentence_words[-1].get('end_time', first_word_time + 0.1))
            sentence_start = first_word_time
            sentence_end = last_word_time
            if sentence_end < sentence_start: sentence_end = sentence_start + 0.01

            structured_sentences.append({
                'text': sentence_text,
                'start_time': sentence_start,
                'end_time': sentence_end,
                'words': list(current_sentence_words)
            })

    return structured_sentences

# --- SRT Loading Logic ---

def parse_srt_timestamp(timestamp_str):
    """Converts SRT timestamp (HH:MM:SS,ms) to seconds (float)."""
    try:
        # Handle both comma and dot for milliseconds separator
        # Use regex to be more flexible with potential leading/trailing whitespace or missing parts
        match = re.match(r'(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{3})', timestamp_str.strip())
        if not match:
             raise ValueError(f"Timestamp format incorrect: {timestamp_str}")

        h = int(match.group(1))
        m = int(match.group(2))
        s = int(match.group(3))
        ms = int(match.group(4))
        return h * 3600 + m * 60 + s + ms / 1000.0
    except Exception as e:
        print(f"WARNING: Could not parse SRT timestamp '{timestamp_str}': {e}", file=sys.stderr)
        return 0.0 # Return 0 on error

def load_srt_timed_text(srt_path: str):
    """
    Loads timed text data from a standard SRT file.
    Parses sentence/segment level timing and text.
    Attempts to parse word-level HTML tags if present to derive word timings,
    otherwise estimates word timings linearly within the segment.

    Args:
        srt_path (str): Path to the SRT file.

    Returns:
        list: A list of structured timed text data (list of sentence dicts).
              Format is similar to the ASR output structure. Word timing
              is either parsed from HTML tags or estimated.
              [ {'text': '...', 'start_time': float, 'end_time': float,
                 'words': [ {'text': '...', 'start_time': float, 'end_time': float}, ... ] }, ...]

    Raises:
        FileNotFoundError: If the SRT file does not exist.
        SRTParseError: If the SRT file cannot be parsed.
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    print(f"Loading and parsing SRT file: {srt_path}", file=sys.stderr)
    structured_sentences = []

    # Adjusted regex to be more robust, look for block number, times, and text until empty line or end
    # Supports both comma and dot for milliseconds separator.
    # Handles potential leading/trailing whitespace around block number and times.
    # Group 1: Index, Group 2: Start Time, Group 3: End Time, Group 4: Text Block
    # Ensure text block captures content across multiple lines.
    segment_pattern = re.compile(
        r'^\s*(\d+)\s*\n' # Block number line, optional leading/trailing space
        r'\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})(?:[^\n]*)\n' # Timestamp line, optional trailing stuff after -->, mandatory newline
        r'(.*?)\n' # Text block (non-greedy, captures multiple lines) followed by a mandatory newline (before the next block's number or end of file)
        r'(?=\s*\n|\Z)', # Lookahead for an empty line (optional space) or end of file
        re.DOTALL | re.MULTILINE
    )
     # NOTE: The segment_pattern regex above might struggle with the *very last* block if it's not followed by an empty line.
     # A simpler approach might be to split the file content by '\n\n' (or '\r\n\r\n') blocks first, then parse each block.
     # Let's try splitting first as it's more reliable for block separation.

    block_separator_pattern = re.compile(r'\n\s*\n') # Pattern for empty line between blocks (handle optional space)

    # Pattern to find potential HTML word tags <font color="...">word</font>
    # Capture content inside font tags. Use non-greedy match for content (.*?)
    # Updated to capture word text and optional internal tags/whitespace more robustly
    html_word_pattern = re.compile(r'<font[^>]*>(.*?)</font>', re.DOTALL)


    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content into potential SRT blocks
        # Ensure content ends with an empty line or process the last block separately
        # A simple split('\n\n') or split(block_separator_pattern) might leave trailing whitespace or fail on the last block.
        # A common trick: add an extra blank line to the end of the content before splitting
        content_for_split = content.strip() + "\n\n"
        raw_blocks = block_separator_pattern.split(content_for_split)
        # Filter out any empty blocks resulting from split
        raw_blocks = [block.strip() for block in raw_blocks if block.strip()]


        if not raw_blocks:
            print("WARNING: No blocks found in SRT file after splitting.", file=sys.stderr)
            return []

        for i, raw_block in enumerate(raw_blocks):
            lines = raw_block.split('\n')
            if len(lines) < 2:
                print(f"WARNING: Skipping malformed block {i+1} (less than 2 lines). Content:\n{raw_block[:100]}...", file=sys.stderr)
                continue

            # First line is index (can be ignored or validated)
            try:
                block_index = int(lines[0].strip())
                # Optional: Validate index against expected sequence (i + 1)
                if block_index != i + 1:
                    print(f"WARNING: SRT block index mismatch. Expected {i+1}, got {block_index}.", file=sys.stderr)
            except ValueError:
                print(f"WARNING: Skipping block {i+1} due to invalid index line: {lines[0].strip()}.", file=sys.stderr)
                continue

            # Second line is timestamps
            timestamp_line = lines[1].strip()
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', timestamp_line)
            if not time_match:
                print(f"WARNING: Skipping block {i+1} due to invalid timestamp line: {timestamp_line}.", file=sys.stderr)
                continue

            try:
                start_time_str = time_match.group(1)
                end_time_str = time_match.group(2)
                start_time = parse_srt_timestamp(start_time_str)
                end_time = parse_srt_timestamp(end_time_str)
            except Exception as e:
                print(f"WARNING: Skipping block {i+1} due to timestamp parsing error: {e}.", file=sys.stderr)
                continue # Skip if timestamp parsing fails

            # Remaining lines are text block
            text_block = "\n".join(lines[2:]).strip()

            # Clean sentence text: remove all HTML tags and replace multiple newlines/spaces with single spaces
            clean_text = re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', '', text_block)).strip()

            segment_words = []
            word_matches = html_word_pattern.findall(text_block) # Find words within HTML tags in the original block

            if word_matches:
                # If HTML tags found, estimate word timing based on their order and segment duration.
                valid_word_matches = [w.strip() for w in word_matches if w.strip()] # Filter empty matches
                word_count = len(valid_word_matches)
                duration = end_time - start_time
                effective_duration = duration if duration > 0 else 0.1
                if word_count > 0:
                    word_duration = effective_duration / word_count
                    for j, word_text_raw in enumerate(valid_word_matches):
                        word_text = word_text_raw # Use the stripped word text from the list
                        # Estimate start/end times based on linear distribution over segment duration
                        word_start = start_time + (j * word_duration)
                        word_end = word_start + word_duration
                        segment_words.append({'text': word_text, 'start_time': word_start, 'end_time': word_end})
                # If HTML tags were found but resulted in no valid words, segment_words remains empty.


            # If no HTML word tags found OR if HTML tags were empty, OR if word timing estimation failed,
            # create word entries by splitting the clean text and estimating timing.
            # This ensures basic word timing is available even for standard SRTs.
            if not segment_words and clean_text:
                # Split the clean text into words using whitespace
                words_from_split = clean_text.split()
                if words_from_split:
                    word_count_split = len(words_from_split)
                    duration_split = end_time - start_time
                    effective_duration_split = duration_split if duration_split > 0 else 0.1

                    if word_count_split > 0:
                        word_duration_split = effective_duration_split / word_count_split
                        for j, word_text in enumerate(words_from_split):
                            word_start = start_time + (j * word_duration_split)
                            word_end = word_start + word_duration_split
                            segment_words.append({'text': word_text, 'start_time': word_start, 'end_time': word_end})


            # Only add sentence to structured data if it has text content (clean_text is the main text)
            # An empty segment text block should result in no structured sentence entry.
            # If the clean text is empty, but word tags were successfully parsed (e.g., just timing tags),
            # we might still want to add the sentence if segment_words is not empty.
            if clean_text: # Prioritize having clean text content for the sentence entry
                structured_sentences.append({
                    'text': clean_text, # Use the clean text for the sentence
                    'start_time': start_time,
                    'end_time': end_time,
                    'words': segment_words # Use the parsed/estimated word timings
                })
            elif segment_words: # Fallback: Add if no clean text but words were parsed (unlikely in practice for SRT)
                # In this case, the sentence text will be empty, but words list is populated.
                structured_sentences.append({
                    'text': "", # Sentence text is empty
                    'start_time': start_time,
                    'end_time': end_time,
                    'words': segment_words
                })
            # else:
                 # print(f"WARNING: Segment {i+1} skipped (empty text and no words parsed).", file=sys.stderr)


        print(f"Successfully loaded {len(structured_sentences)} sentences from SRT.", file=sys.stderr)
        return structured_sentences

    except FileNotFoundError:
        raise # Re-raise FileNotFoundError caught earlier

    except Exception as e:
        print(f"ERROR during SRT parsing: {e}", file=sys.stderr)
        raise SRTParseError(f"Error parsing SRT file '{srt_path}': {e}")


# --- SRT Saving Logic ---

def split_sentence_into_chunks(sentence_data: dict, max_chars: int = 80, tolerance: int = 20) -> list:
    """
    Splits a single sentence dictionary into multiple chunks based on character length constraints.
    Uses word-level timing to calculate start/end times for each chunk.

    Args:
        sentence_data (dict): The original sentence dictionary with 'text', 'start_time', 'end_time', 'words'.
        max_chars (int): Target maximum characters per subtitle block.
        tolerance (int): Tolerance for the split point.

    Returns:
        list: A list of sentence dictionaries (chunks).
    """
    text = sentence_data.get('text', '')
    words = sentence_data.get('words', [])
    
    if not words:
        return [sentence_data]

    hard_limit = max_chars + tolerance
    
    if len(text) <= hard_limit:
        return [sentence_data]

    chunks = []
    current_chunk_words = []
    current_chunk_text_len = 0
    
    for i, word in enumerate(words):
        word_text = word.get('text', '')
        word_len = len(word_text)
        added_len = word_len + (1 if current_chunk_words else 0)
        
        if current_chunk_text_len + added_len > max_chars:
            if current_chunk_text_len + added_len <= hard_limit:
                current_chunk_words.append(word)
                current_chunk_text_len += added_len
            else:
                if current_chunk_words:
                    chunk_text = " ".join([w['text'] for w in current_chunk_words])
                    chunk_start = current_chunk_words[0]['start_time']
                    chunk_end = current_chunk_words[-1]['end_time']
                    chunks.append({
                        'text': chunk_text,
                        'start_time': chunk_start,
                        'end_time': chunk_end,
                        'words': current_chunk_words
                    })
                
                current_chunk_words = [word]
                current_chunk_text_len = word_len
        else:
            current_chunk_words.append(word)
            current_chunk_text_len += added_len

    if current_chunk_words:
        chunk_text = " ".join([w['text'] for w in current_chunk_words])
        chunk_start = current_chunk_words[0]['start_time']
        chunk_end = current_chunk_words[-1]['end_time']
        chunks.append({
            'text': chunk_text,
            'start_time': chunk_start,
            'end_time': chunk_end,
            'words': current_chunk_words
        })

    return chunks

def save_timed_text_as_srt(structured_data: list, file_path: str | Path, max_chars: int = 80, tolerance: int = 20):
    """
    Saves structured timed text data to a standard SRT file.

    Args:
        structured_data (list): List of sentence dicts following the specified structure.
        file_path (str | Path): Path to the desired output SRT file.
        max_chars (int): Maximum characters per subtitle block (target).
        tolerance (int): Tolerance for exceeding max_chars.

    Raises:
        SRTWriteError: If writing to the file fails.
        TypeError: If structured_data is not in the expected format.
    """
    if not isinstance(structured_data, list):
        raise TypeError("Input data must be a list of sentence dictionaries.")
    if not structured_data:
        print("INFO: No data to save to SRT.", file=sys.stderr)
        # Still attempt to create an empty file? Or just return?
        # Let's create an empty file for consistency if save is triggered.
        pass # Continue to write loop

    print(f"Saving structured data to SRT: {file_path}", file=sys.stderr)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            srt_index = 1
            for i, sentence in enumerate(structured_data):
                # Validate sentence structure minimally
                if not isinstance(sentence, dict) or 'text' not in sentence or 'start_time' not in sentence or 'end_time' not in sentence:
                    print(f"WARNING: Skipping malformed sentence data at index {i}: {sentence}. Does not have required keys.", file=sys.stderr)
                    continue

                chunks = split_sentence_into_chunks(sentence, max_chars, tolerance)
                
                for chunk in chunks:
                    chunk_text = str(chunk.get('text', '')).strip()
                    chunk_start = float(chunk.get('start_time', 0.0))
                    chunk_end = float(chunk.get('end_time', chunk_start + 0.1))
                    
                    if chunk_end < chunk_start: chunk_end = chunk_start + 0.01

                    if chunk_text:
                        f.write(f"{srt_index}\n") 
                        f.write(f"{format_srt_timestamp(chunk_start)} --> {format_srt_timestamp(chunk_end)}\n")
                        f.write(f"{chunk_text}\n")
                        f.write("\n") 
                        srt_index += 1

        print("SRT file written successfully.", file=sys.stderr)

    except (OSError, IOError) as e:
        print(f"ERROR writing SRT file '{file_path}': {e}", file=sys.stderr)
        raise SRTWriteError(f"Error writing SRT file '{file_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during SRT writing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise SRTWriteError(f"An unexpected error occurred during SRT writing: {e}")


# --- Standalone Test Suite ---

def run_model_tests(level: int):
    """
    Runs a series of tests for the model logic based on the specified level.

    Args:
        level (int): The level of tests to run (1 to N).
    """
    print(f"--- Running Model Tests up to Level {level} ---", file=sys.stderr)

    overall_test_passed = True

    if level >= 1:
        print("\n--- Test Level 1: Environment and Dependency Checks ---", file=sys.stderr)
        test1_passed = True
        try:
            print("Checking FFmpeg availability...", file=sys.stderr)
            if check_ffmpeg_available():
                print("  FFmpeg found.", file=sys.stderr)
            else:
                print("  FFmpeg NOT found. This is required for pydub to process many audio formats.", file=sys.stderr)
                # test1_passed = False # Don't fail level 1 just for missing optional dependency FFmpeg

            print("Attempting to import NeMo and PyTorch...", file=sys.stderr)
            if NEMO_AVAILABLE:
                print("  NeMo and PyTorch imported successfully.", file=sys.stderr)
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    print(f"  CUDA detected. GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
                else:
                    print("  CUDA NOT detected. NeMo Parakeet requires CUDA.", file=sys.stderr)
                    # Missing CUDA means transcription will fail, so fail this test level if needed
                    # Let's make this a warning for Level 1, transcription test (L3) will fail properly
                    # test1_passed = False # This is now handled by L3 requiring can_run_asr_tests

            else:
                print("  NeMo or PyTorch NOT available. Please install: nemo_toolkit[asr], torch (CUDA version), cuda-python.", file=sys.stderr)
                test1_passed = False # Fail if core libs not available

            print("Attempting to import pydub...", file=sys.stderr)
            if PYDUB_AVAILABLE:
                print("  pydub imported successfully.", file=sys.stderr)
            else:
                print("  pydub NOT available. Please install: pip install pydub", file=sys.stderr)
                test1_passed = False # Fail if core lib not available


            if test1_passed:
                print("Test Level 1 PASSED.", file=sys.stderr)
            else:
                print("Test Level 1 FAILED. Check the specific errors above.", file=sys.stderr)

        except Exception as e:
            print(f"An unexpected error occurred during Test Level 1: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            test1_passed = False
        overall_test_passed = overall_test_passed and test1_passed


    # Determine if ASR-dependent tests can run
    can_run_asr_tests = (NEMO_AVAILABLE and PYDUB_AVAILABLE and check_ffmpeg_available() and hasattr(torch, 'cuda') and torch.cuda.is_available())
    if not can_run_asr_tests:
        print("\n--- Skipping ASR-dependent tests (Level 2, 3) due to missing dependencies or CUDA. ---", file=sys.stderr)
        print("    Requires: NeMo, PyTorch (CUDA version), cuda-python, pydub, FFmpeg in PATH, CUDA-enabled GPU.", file=sys.stderr)


    if level >= 2 and can_run_asr_tests: # Only run if dependencies met
        print("\n--- Test Level 2: Audio Processing (Requires FFmpeg & pydub) ---", file=sys.stderr)
        test2_passed = True
        # Need a small actual audio file for testing realistic processing
        # Use a temp file if test_audio.wav doesn't exist, though having a real file is better
        test_audio_path = Path(__file__).parent / "test_audio.wav" # Expect test_audio.wav next to model.py
        if not test_audio_path.exists():
            print(f"  Skipping Test Level 2: Requires a file named 'test_audio.wav' in the same directory as model.py", file=sys.stderr)
            test2_passed = False # Treat as failure if required test file is missing
        else:
            temp_segment_files = [] # Initialize list before try block
            try:
                print(f"Processing test audio file: {test_audio_path}", file=sys.stderr)
                # Process into short segments (e.g., 1 second)
                temp_segment_files, total_dur, segment_times = process_audio_for_nemo(str(test_audio_path), segment_length_sec=1)
                print(f"  Processing successful. Created {len(temp_segment_files)} temp segments.", file=sys.stderr)
                print(f"  Total duration detected: {total_dur:.3f}s", file=sys.stderr)
                if total_dur > 0:
                    # Verify segment times are within total duration and cover the range
                    if segment_times:
                        first_start = segment_times[0][0]
                        last_end = segment_times[-1][1]
                        if first_start < 0 or last_end > total_dur + 1.0: # Allow a small buffer
                            print(f"  Validation Warning: Segment times ({first_start:.3f} - {last_end:.3f}s) seem outside total duration ({total_dur:.3f}s).", file=sys.stderr)
                        # Check if segments roughly cover the duration based on segment length and count
                        # This check is heuristic and can be complex, simplified check: last segment end is near total duration
                        if total_dur > 0.5 and last_end < total_dur * 0.9: # Check if last segment end is significantly less than total duration for non-trivial audio
                            print(f"  Validation Warning: Last segment end time ({last_end:.3f}s) is significantly less than total duration ({total_dur:.3f}s). Segmentation might be off.", file=sys.stderr)
                    else:
                        print("  Validation Warning: No segment times returned.", file=sys.stderr)


                if len(temp_segment_files) > 0 and total_dur > 0:
                    print("Test Level 2 PASSED.", file=sys.stderr)
                    test2_passed = True
                else:
                    print("Test Level 2 FAILED: Audio processing did not produce expected output (0 segments or 0 duration).", file=sys.stderr)
                    test2_passed = False


            except (FFmpegNotFoundError, InvalidAudioFileError) as e:
                print(f"  Test Level 2 FAILED due to audio processing error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                test2_passed = False
            except Exception as e:
                print(f"  An unexpected error occurred during Test Level 2: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                test2_passed = False
            finally:
                # Clean up segments immediately after this test
                if temp_segment_files:
                    print("  Cleaning up test segments...", file=sys.stderr)
                    for seg_path in temp_segment_files:
                        if os.path.exists(seg_path):
                            try:
                                os.remove(seg_path)
                            except Exception as e:
                                print(f"  Warning: Failed to remove temporary file {seg_path} during cleanup: {e}", file=sys.stderr)
                    # print("  Segment cleanup complete.", file=sys.stderr) # Less verbose

        overall_test_passed = overall_test_passed and test2_passed


    if level >= 3 and can_run_asr_tests: # Only run if dependencies met
        print("\n--- Test Level 3: ASR Transcription (Requires NeMo, CUDA, pydub, FFmpeg) ---", file=sys.stderr)
        test3_passed = True
        test_audio_path = Path(__file__).parent / "test_audio.wav" # Expect test_audio.wav
        if not test_audio_path.exists():
            print(f"  Skipping Test Level 3: Requires a file named 'test_audio.wav' in the same directory as model.py", file=sys.stderr)
            test3_passed = False # Missing test file means test cannot run
        else:
            try:
                print(f"Attempting transcription of {test_audio_path} with word timestamps...", file=sys.stderr)
                # Use a short segment length for testing
                timed_data = transcribe_audio_and_time(str(test_audio_path), segment_length_sec=5)

                if timed_data:
                    print(f"  Transcription successful. Received structured data with {len(timed_data)} sentences.", file=sys.stderr)
                    # Optional: print sample output - can be noisy
                    # print("  Sample structured data:", file=sys.stderr)
                    # print(json.dumps(timed_data[:min(2, len(timed_data))], indent=2), file=sys.stderr)

                    # Basic validation: Check structure and presence of data
                    test3_validation_ok = True
                    if not isinstance(timed_data, list):
                        print("  Validation Failed: Top level is not a list.", file=sys.stderr)
                        test3_validation_ok = False
                    elif len(timed_data) == 0:
                        print("  Validation Failed: Received empty list.", file=sys.stderr)
                        test3_validation_ok = False
                    else:
                        # Check structure of first few sentences and their words
                        for i, sentence in enumerate(timed_data[:min(5, len(timed_data))]): # Check first 5 sentences
                            if not isinstance(sentence, dict) or not all(k in sentence for k in ['text', 'start_time', 'end_time', 'words']):
                                print(f"  Validation Failed: Sentence {i} structure incorrect. Missing required keys. Data: {sentence.keys() if isinstance(sentence, dict) else sentence}", file=sys.stderr)
                                test3_validation_ok = False
                                break # Stop checking further sentences

                            if not isinstance(sentence.get('words'), list):
                                print(f"  Validation Failed: Sentence {i} 'words' is not a list. Type: {type(sentence.get('words'))}", file=sys.stderr)
                                test3_validation_ok = False
                                break # Stop checking further sentences

                            if len(sentence.get('words', [])) > 0:
                                first_word = sentence['words'][0]
                                if not isinstance(first_word, dict) or not all(k in first_word for k in ['text', 'start_time', 'end_time']):
                                    print(f"  Validation Failed: Sentence {i}, first word structure incorrect. Missing required keys. Data: {first_word.keys() if isinstance(first_word, dict) else first_word}", file=sys.stderr)
                                    test3_validation_ok = False
                                    break # Stop checking further sentences

                                # Check if timestamps are plausible (non-negative, end >= start)
                                if sentence.get('start_time', -1) < 0 or sentence.get('end_time', -1) < 0 or sentence.get('end_time', 0) < sentence.get('start_time', 0):
                                    print(f"  Validation Failed: Sentence {i} times incorrect ({sentence.get('start_time')}-{sentence.get('end_time')}).", file=sys.stderr)
                                    test3_validation_ok = False
                                    break

                                if first_word.get('start_time', -1) < 0 or first_word.get('end_time', -1) < 0 or first_word.get('end_time', 0) < first_word.get('start_time', 0):
                                    print(f"  Validation Failed: Sentence {i}, first word times incorrect ({first_word.get('start_time')}-{first_word.get('end_time')}).", file=sys.stderr)
                                    test3_validation_ok = False
                                    break


                    if test3_validation_ok:
                        print("Test Level 3 PASSED. Structured data generated and validated successfully.", file=sys.stderr)
                        test3_passed = True
                    else:
                        print("Test Level 3 FAILED: Generated data structure is not as expected.", file=sys.stderr)
                        test3_passed = False

                else:
                    print("Test Level 3 FAILED: Transcription returned no timed data.", file=sys.stderr)
                    test3_passed = False

            except (FFmpegNotFoundError, InvalidAudioFileError, NemoInitializationError, TranscriptionError) as e:
                print(f"  Test Level 3 FAILED due to transcription/processing error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                test3_passed = False
            except Exception as e:
                print(f"  An unexpected error occurred during Test Level 3: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                test3_passed = False
        overall_test_passed = overall_test_passed and test3_passed


    if level >= 4: # SRT Loading doesn't strictly depend on ASR dependencies
        print("\n--- Test Level 4: SRT Loading ---", file=sys.stderr)
        test4_passed = True
        # Create a dummy SRT file for testing
        dummy_srt_content = """1
00:00:01,000 --> 00:00:02,500
Hello world. This is the first sentence.

2
00:00:03,000 --> 00:00:04,000
<font color="#ff0000">This</font> <font color="#ff0000">is</font> a test.

3
00:00:05,000 --> 00:00:06,000
Another segment.
With a newline.

4
00:00:07,000 --> 00:00:07,500
Empty segment text block.

5
00:00:08,000 --> 00:00:08,500

6
00:00:09,000 --> 00:00:10,000
SingleWord.

7
00:00:11,000 --> 00:00:12,000
<font>TagOnlyWord</font>.

8
00:00:13,000 --> 00:00:14,000
Word with period.

""" # Added empty segments, multi-line, single word, and tag-only-word to test robustness
        dummy_srt_path = Path(tempfile.gettempdir()) / "dummy_test_load.srt"
        try:
            with open(dummy_srt_path, 'w', encoding='utf-8') as f:
                f.write(dummy_srt_content)
            print(f"Created dummy test SRT file for loading: {dummy_srt_path}", file=sys.stderr)

            print(f"Attempting to load SRT file: {dummy_srt_path}", file=sys.stderr)
            srt_data = load_srt_timed_text(str(dummy_srt_path))

            if srt_data is not None: # load_srt_timed_text now returns [] or raises on failure
                print(f"  SRT loading successful. Received structured data with {len(srt_data)} sentences.", file=sys.stderr)
                # print("  Sample SRT data:", file=sys.stderr)
                # print(json.dumps(srt_data, indent=2), file=sys.stderr)

                # Basic validation
                test4_validation_ok = True
                if not isinstance(srt_data, list):
                    print("  Validation Failed: Top level is not a list.", file=sys.stderr)
                    test4_validation_ok = False
                # Expect 6 sentences with content (block 5 is empty text block)
                elif len(srt_data) != 6:
                    print(f"  Validation Failed: Expected 6 sentences with content, got {len(srt_data)}.", file=sys.stderr)
                    # print(json.dumps(srt_data, indent=2), file=sys.stderr) # Debugging print
                    test4_validation_ok = False
                elif test4_validation_ok: # Check content of the sentences
                    expected_sentences = [
                        ("Hello world. This is the first sentence.", 1.0, 2.5, 7, ['Hello', 'world', 'This', 'is', 'the', 'first', 'sentence']),
                        ("This is a test.", 3.0, 4.0, 3, ['This', 'is', 'test']),
                        ("Another segment. With a newline.", 5.0, 6.0, 5, ['Another', 'segment', 'With', 'a', 'newline']),
                        ("Empty segment text block.", 7.0, 7.5, 4, ['Empty', 'segment', 'text', 'block']),
                        ("SingleWord.", 9.0, 10.0, 1, ['SingleWord']),
                        ("TagOnlyWord.", 11.0, 12.0, 1, ['TagOnlyWord']),
                        ("Word with period.", 13.0, 14.0, 3, ['Word', 'with', 'period']), # Added this block to dummy data, should be 7th sentence
                    ]
                    # Adjusted expected count and validation loop based on the actual data and parser logic
                    expected_count = 7 # Block 5 is empty, Block 4 is text block only
                    if len(srt_data) != expected_count:
                        print(f"  Validation Failed: Expected {expected_count} sentences with content, got {len(srt_data)}.", file=sys.stderr)
                        test4_validation_ok = False
                    else:
                        for j, expected in enumerate(expected_sentences):
                            actual = srt_data[j]
                            exp_text, exp_start, exp_end, exp_word_count, exp_words_list = expected
                            act_text = actual.get('text', '').strip()
                            act_start = actual.get('start_time', 0.0)
                            act_end = actual.get('end_time', 0.0)
                            act_words = actual.get('words', [])
                            act_word_count = len(act_words)

                            # Compare text, times, and word count
                            if act_text != exp_text or abs(act_start - exp_start) > 0.001 or abs(act_end - exp_end) > 0.001 or act_word_count != exp_word_count:
                                print(f"  Validation Failed: Sentence {j+1} content/times/word count incorrect.", file=sys.stderr)
                                print(f"    Expected: Text='{exp_text}', Times={exp_start:.3f}-{exp_end:.3f}, Word Count={exp_word_count}", file=sys.stderr)
                                print(f"    Actual:   Text='{act_text}', Times={act_start:.3f}-{act_end:.3f}, Word Count={act_word_count}", file=sys.stderr)
                                test4_validation_ok = False
                                # break # Check all sentences for better diagnostics

                            # Optionally check word texts if word count matches
                            if test4_validation_ok and act_word_count > 0 and exp_word_count > 0:
                                actual_word_texts = [w.get('text', '').strip() for w in act_words]
                                if actual_word_texts != exp_words_list:
                                    print(f"  Validation Failed: Sentence {j+1} parsed word texts incorrect.", file=sys.stderr)
                                    print(f"    Expected words: {exp_words_list}", file=sys.stderr)
                                    print(f"    Actual words:   {actual_word_texts}", file=sys.stderr)
                                    test4_validation_ok = False
                                    # break # Check all sentences


                if test4_validation_ok:
                    print("Test Level 4 PASSED. SRT data loaded and validated successfully.", file=sys.stderr)
                    test4_passed = True
                else:
                    print("Test Level 4 FAILED: Loaded data structure or content is not as expected.", file=sys.stderr)
                    test4_passed = False

            else:
                # load_srt_timed_text returned None or raised an exception
                print("Test Level 4 FAILED: SRT loading returned None or an error occurred (see above).", file=sys.stderr)
                test4_passed = False

        except (FileNotFoundError, SRTParseError) as e:
            print(f"  Test Level 4 FAILED due to SRT processing error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            test4_passed = False
        except Exception as e:
            print(f"  An unexpected error occurred during Test Level 4: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            test4_passed = False
        finally:
            if dummy_srt_path.exists():
                try:
                    os.remove(dummy_srt_path)
                except Exception as e:
                    print(f"  Warning: Failed to remove dummy SRT file {dummy_srt_path} during cleanup: {e}", file=sys.stderr)

        overall_test_passed = overall_test_passed and test4_passed


    if level >= 5:
        print("\n--- Test Level 5: SRT Saving ---", file=sys.stderr)
        test5_passed = True

        # Create sample structured data to save
        sample_structured_data = [
            {'text': 'This is the first sentence.', 'start_time': 0.5, 'end_time': 2.1,
             'words': [{'text': 'This', 'start_time': 0.5, 'end_time': 0.8}, {'text': 'is', 'start_time': 0.8, 'end_time': 1.0}, {'text': 'the', 'start_time': 1.0, 'end_time': 1.2}, {'text': 'first', 'start_time': 1.2, 'end_time': 1.7}, {'text': 'sentence.', 'start_time': 1.7, 'end_time': 2.1}]},
            {'text': 'Here is another sentence\nwith a newline.', 'start_time': 3.0, 'end_time': 4.5,
             'words': [{'text': 'Here', 'start_time': 3.0, 'end_time': 3.3}, {'text': 'is', 'start_time': 3.3, 'end_time': 3.5}, {'text': 'another', 'start_time': 3.5, 'end_time': 4.0}, {'text': 'sentence\nwith', 'start_time': 4.0, 'end_time': 4.3}, {'text': 'a', 'start_time': 4.3, 'end_time': 4.4}, {'text': 'newline.', 'start_time': 4.4, 'end_time': 4.5}]}, # Added a word with internal newline for testing
            {'text': 'Short block.', 'start_time': 5.001, 'end_time': 5.999,
             'words': [{'text': 'Short', 'start_time': 5.001, 'end_time': 5.5}, {'text': 'block.', 'start_time': 5.5, 'end_time': 5.999}]},
             # Add an empty text sentence - should be skipped in saving
            {'text': '', 'start_time': 6.0, 'end_time': 7.0, 'words': []},
            # Add a sentence with words but empty text - should also be skipped in saving based on current logic
            {'text': '', 'start_time': 8.0, 'end_time': 9.0, 'words': [{'text': 'Empty', 'start_time': 8.0, 'end_time': 8.5}, {'text': 'Sentence', 'start_time': 8.5, 'end_time': 9.0}]},

        ]
        # Expected SRT content (note the index is 1-based, and empty sentences are skipped)
        expected_srt_content = """1
00:00:00,500 --> 00:00:02,100
This is the first sentence.

2
00:00:03,000 --> 00:00:04,500
Here is another sentence
with a newline.

3
00:00:05,001 --> 00:00:05,999
Short block.

""" # Note the newline is preserved in the text block

        dummy_save_path = Path(tempfile.gettempdir()) / "dummy_test_save.srt"

        try:
            print(f"Attempting to save sample data to SRT: {dummy_save_path}", file=sys.stderr)
            save_timed_text_as_srt(sample_structured_data, str(dummy_save_path))
            print("  Save operation completed.", file=sys.stderr)

            if not dummy_save_path.exists():
                print("  Validation Failed: Output file was not created.", file=sys.stderr)
                test5_passed = False
            else:
                # Read the saved file content
                with open(dummy_save_path, 'r', encoding='utf-8') as f:
                    saved_content = f.read()

                print("  Saved content:", file=sys.stderr)
                print(saved_content, file=sys.stderr)

                # Compare saved content with expected content
                # Strip whitespace from both ends for robust comparison
                if saved_content.strip() == expected_srt_content.strip():
                    print("  Validation PASSED: Saved content matches expected content.", file=sys.stderr)
                    test5_passed = True
                else:
                    print("  Validation FAILED: Saved content does NOT match expected content.", file=sys.stderr)
                    print("    Expected:", file=sys.stderr)
                    print(expected_srt_content.strip(), file=sys.stderr)
                    print("    Actual:", file=sys.stderr)
                    print(saved_content.strip(), file=sys.stderr)
                    test5_passed = False
            
            # --- Test 5b: Splitting Logic ---
            print("\n  Testing Splitting Logic...", file=sys.stderr)
            long_sentence_data = [{
                'text': 'This is a very long sentence that should definitely be split into multiple parts because it exceeds the default character limit of ten characters for this specific test case.',
                'start_time': 0.0, 'end_time': 10.0,
                'words': [
                    {'text': 'This', 'start_time': 0.0, 'end_time': 0.5},
                    {'text': 'is', 'start_time': 0.5, 'end_time': 1.0},
                    {'text': 'a', 'start_time': 1.0, 'end_time': 1.2},
                    {'text': 'very', 'start_time': 1.2, 'end_time': 1.8},
                    {'text': 'long', 'start_time': 1.8, 'end_time': 2.5},
                    {'text': 'sentence', 'start_time': 2.5, 'end_time': 3.5},
                    {'text': 'that', 'start_time': 3.5, 'end_time': 4.0},
                    {'text': 'should', 'start_time': 4.0, 'end_time': 4.5},
                    {'text': 'definitely', 'start_time': 4.5, 'end_time': 5.5},
                    {'text': 'be', 'start_time': 5.5, 'end_time': 5.8},
                    {'text': 'split', 'start_time': 5.8, 'end_time': 6.3},
                    {'text': 'into', 'start_time': 6.3, 'end_time': 6.8},
                    {'text': 'multiple', 'start_time': 6.8, 'end_time': 7.5},
                    {'text': 'parts', 'start_time': 7.5, 'end_time': 8.0},
                    {'text': 'because', 'start_time': 8.0, 'end_time': 8.5},
                    {'text': 'it', 'start_time': 8.5, 'end_time': 8.7},
                    {'text': 'exceeds', 'start_time': 8.7, 'end_time': 9.2},
                    {'text': 'the', 'start_time': 9.2, 'end_time': 9.4},
                    {'text': 'limit.', 'start_time': 9.4, 'end_time': 10.0}
                ]
            }]
            # Test with small limit to force split
            dummy_split_path = Path(tempfile.gettempdir()) / "dummy_test_split.srt"
            save_timed_text_as_srt(long_sentence_data, str(dummy_split_path), max_chars=20, tolerance=5)
            
            with open(dummy_split_path, 'r', encoding='utf-8') as f:
                split_content = f.read()
            
            print("  Split Saved content:", file=sys.stderr)
            print(split_content, file=sys.stderr)
            
            if "2\n" in split_content:
                 print("  Validation PASSED: Long sentence was split.", file=sys.stderr)
            else:
                 print("  Validation FAILED: Long sentence was NOT split.", file=sys.stderr)
                 test5_passed = False
            
            if dummy_split_path.exists(): os.remove(dummy_split_path)

        except (SRTWriteError, OSError, IOError) as e:
            print(f"  Test Level 5 FAILED due to saving error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            test5_passed = False
        except Exception as e:
            print(f"  An unexpected error occurred during Test Level 5: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            test5_passed = False
        finally:
            if dummy_save_path.exists():
                try:
                    os.remove(dummy_save_path)
                except Exception as e:
                    print(f"  Warning: Failed to remove dummy SRT file {dummy_save_path} during cleanup: {e}", file=sys.stderr)


        overall_test_passed = overall_test_passed and test5_passed


    # Add more test levels here if needed

    if overall_test_passed:
        print("\n--- All Requested Tests PASSED ---", file=sys.stderr)
        sys.exit(0) # Exit with success code
    else:
        print("\n--- Some Tests FAILED ---", file=sys.stderr)
        sys.exit(1) # Exit with failure code


# --- Main Execution for Standalone Testing ---

if __name__ == "__main__":
    # This __main__ block is primarily for allowing `python model.py --test X` execution.
    # When main.py runs `python main.py --test X`, model.py will receive
    # --test X in its sys.argv.
    # We need to check if this script was executed directly and if --test is intended for us.
    # The simplest check is if the script name in argv[0] matches this file's path.

    script_path_arg = Path(sys.argv[0])
    this_file_path = Path(__file__).resolve()

    # Check if the script was run directly AND --test is one of the arguments
    # Avoid parsing if imported by main.py which already handled the --test detection
    if script_path_arg.samefile(this_file_path) and "--test" in sys.argv:
        # Need to parse only the arguments intended for model.py's test execution
        model_parser = argparse.ArgumentParser(
            description="Model logic for Parakeet_GUI. Can be run standalone for tests."
        )
        model_parser.add_argument(
            "--test",
            type=int,
            metavar='LEVEL',
            required=True, # Make --test level mandatory if running model.py directly with --test
            help="Run tests up to the specified LEVEL. Requires test_audio.wav for levels >= 2."
        )
        try:
            # Parse only the arguments that start with --test
            # This allows ignoring other arguments potentially passed unintentionally
            test_args = [arg for arg in sys.argv[1:] if arg.startswith('--test')]
            if not test_args:
                print("Error: --test requires a LEVEL argument when running model.py standalone.", file=sys.stderr)
                model_parser.print_help()
                sys.exit(1)

            # model_parser.parse_args expects the list of args *without* the script name
            args = model_parser.parse_args(sys.argv[1:])
            run_model_tests(args.test)
        except SystemExit as e:
            # argparse calls sys.exit on help/error, propagate that exit code
            sys.exit(e.code)
        except Exception as e:
            print(f"An unexpected error occurred during model.py standalone execution: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    else:
        # Running as imported module (e.g., by main.py) or run directly without --test
        if script_path_arg.samefile(this_file_path):
            print("Model script is intended to be used by main.py or run with --test argument.", file=sys.stderr)
            print("Run with `python model.py --test [LEVEL]` for standalone tests or `python main.py` for GUI.", file=sys.stderr)
            sys.exit(0) # Exit gracefully if run directly without expected args
        else:
            pass # No action needed, main.py handles execution and calls transcribe/load functions