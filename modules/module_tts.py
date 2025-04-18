"""
module_tts.py

Text-to-Speech (TTS) module for TARS-AI application.

Handles TTS functionality to convert text into audio using:
- Azure Speech SDK
- Local tools (e.g., espeak-ng)
- Server-based TTS systems

"""

# === Standard Libraries ===
import requests
import os 
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO
import asyncio

from modules.module_piper import text_to_speech_with_pipelining_piper
from modules.module_silero import text_to_speech_with_pipelining_silero
from modules.module_espeak import text_to_speech_with_pipelining_espeak
from modules.module_alltalk import text_to_speech_with_pipelining_alltalk
from modules.module_elevenlabs import text_to_speech_with_pipelining_elevenlabs
from modules.module_azure import text_to_speech_with_pipelining_azure
from modules.module_messageQue import queue_message

def update_tts_settings(ttsurl):
    """
    Updates TTS settings using a POST request to the specified server.

    Parameters:
    - ttsurl: The URL of the TTS server.
    """

    url = f"{ttsurl}/set_tts_settings"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    payload = {
        "stream_chunk_size": 100,
        "temperature": 0.75,
        "speed": 1,
        "length_penalty": 1.0,
        "repetition_penalty": 5,
        "top_p": 0.85,
        "top_k": 50,
        "enable_text_splitting": True
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            queue_message(f"LOAD: TTS Settings updated successfully.")
        else:
            queue_message(f"ERROR: Failed to update TTS settings. Status code: {response.status_code}")
            queue_message(f"INFO: Response: {response.text}")
    except Exception as e:
        queue_message(f"ERROR: TTS update failed: {e}")

def play_audio_stream(tts_stream, samplerate=22050, channels=1, gain=1.0, normalize=False):
    """
    Play the audio stream through speakers using SoundDevice with volume/gain adjustment.
    
    Parameters:
    - tts_stream: Stream of audio data in chunks.
    - samplerate: The sample rate of the audio data.
    - channels: The number of audio channels (e.g., 1 for mono, 2 for stereo).
    - gain: A multiplier for adjusting the volume. Default is 1.0 (no change).
    - normalize: Whether to normalize the audio to use the full dynamic range.
    """
    try:
        with sd.OutputStream(samplerate=samplerate, channels=channels, dtype='int16', blocksize=4096) as stream:
            for chunk in tts_stream:
                if chunk:
                    # Convert bytes to int16 using numpy
                    audio_data = np.frombuffer(chunk, dtype='int16')
                    
                    # Normalize the audio (if enabled)
                    if normalize:
                        max_value = np.max(np.abs(audio_data))
                        if max_value > 0:
                            audio_data = audio_data / max_value * 32767
                    
                    # Apply gain adjustment
                    audio_data = np.clip(audio_data * gain, -32768, 32767).astype('int16')

                    # Write the adjusted audio data to the stream
                    stream.write(audio_data)
                else:
                    queue_message(f"ERROR: Received empty chunk.")
    except Exception as e:
        queue_message(f"ERROR: Error during audio playback: {e}")


async def generate_tts_audio(text, ttsoption, azure_api_key=None, azure_region=None, ttsurl=None, toggle_charvoice=True, tts_voice=None):
    """
    Generate TTS audio for the given text using the specified TTS system.

    Parameters:
    - text (str): The text to convert into speech.
    - ttsoption (str): The TTS system to use (Azure, server-based, or local).
    - ttsurl (str): The base URL of the TTS server (for server-based TTS).
    - toggle_charvoice (bool): Flag indicating whether to use character voice for TTS.
    - tts_voice (str): The TTS speaker/voice configuration.
    """
    try:
        # Azure TTS generation
        if ttsoption == "azure":
           async for chunk in text_to_speech_with_pipelining_azure(text):
                yield chunk

        # Local TTS generation using `espeak-ng`
        elif ttsoption == "espeak":
            async for chunk in text_to_speech_with_pipelining_espeak(text):
                yield chunk

        elif ttsoption == "alltalk":
            async for chunk in text_to_speech_with_pipelining_alltalk(text):
                yield chunk
                
        # Local TTS generation using local onboard PIPER TTS
        elif ttsoption == "piper":
            async for chunk in text_to_speech_with_pipelining_piper(text):
                yield chunk  

        elif ttsoption == "elevenlabs":
            async for chunk in text_to_speech_with_pipelining_elevenlabs(text):
                yield chunk

        elif ttsoption == "silero":
            async for chunk in text_to_speech_with_pipelining_silero(text):
                yield chunk 

        else:
            raise ValueError(f"ERROR: Invalid TTS option.")

    except Exception as e:
        queue_message(f"ERROR: Text-to-speech generation failed: {e}")

async def play_audio_chunks(text, config):
    """
    Plays audio chunks sequentially from the generate_tts_audio function.
    """
    async for audio_chunk in generate_tts_audio(text, config):
        try:
            # Read the audio chunk into a format playable by sounddevice
            data, samplerate = sf.read(audio_chunk, dtype='float32')
            sd.play(data, samplerate)
            await asyncio.sleep(len(data) / samplerate)  # Wait for playback to finish
        except Exception as e:
            queue_message(f"ERROR: Failed to play audio chunk: {e}")