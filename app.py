"""
app.py

Main entry point for the TARS-AI application.

Initializes modules, loads configuration, and manages key threads for functionality such as:
- Speech-to-text (STT)
- Text-to-speech (TTS)
- Bluetooth control
- AI response generation

Run this script directly to start the application.
"""

# === Standard Libraries ===
import os
import sys
import threading
from datetime import datetime

# === Custom Modules ===
from modules.module_config import load_config
from modules.module_character import CharacterManager
from modules.module_memory import MemoryManager
from modules.module_stt import STTManager
from modules.module_tts import update_tts_settings
from modules.module_btcontroller import *
from modules.module_main import initialize_managers, wake_word_callback, utterance_callback, post_utterance_callback, start_bt_controller_thread, start_discord_bot, process_discord_message_callback
from modules.module_vision import initialize_blip
from modules.module_llm import initialize_manager_llm
import modules.module_chatui

# === Constants and Globals ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.append(os.getcwd())

CONFIG = load_config()

# === Helper Functions ===
def init_app():
    """
    Performs initial setup for the application
    """
    
    queue_message(f"LOAD: Script running from: {BASE_DIR}")
    #queue_message(f"DEBUG: init_app() called")
    
    # Load the configuration
    CONFIG = load_config()
    if CONFIG['TTS']['ttsoption'] == 'xttsv2':
        update_tts_settings(CONFIG['TTS']['ttsurl'])

def start_discord_in_thread():
    """
    Start the Discord bot in a separate thread to prevent blocking.
    """
    discord_thread = threading.Thread(target=start_discord_bot, args=(process_discord_message_callback,), daemon=True)
    discord_thread.start()
    queue_message("INFO: Discord bot started in a separate thread.")

# === Main Application Logic ===
if __name__ == "__main__":
    # Perform initial setup
    init_app()

    # Create a shutdown event for global threads
    shutdown_event = threading.Event()

    # Initialize CharacterManager, MemoryManager
    char_manager = CharacterManager(config=CONFIG)
    memory_manager = MemoryManager(config=CONFIG, char_name=char_manager.char_name, char_greeting=char_manager.char_greeting)
   
    # Initialize STTManager
    stt_manager = STTManager(config=CONFIG, shutdown_event=shutdown_event)
    stt_manager.set_wake_word_callback(wake_word_callback)
    stt_manager.set_utterance_callback(utterance_callback)
    stt_manager.set_post_utterance_callback(post_utterance_callback)

    #DISCORD Callback
    if CONFIG['DISCORD']['enabled'] == 'True':
        start_discord_in_thread()

    # Pass managers to main module
    initialize_managers(memory_manager, char_manager, stt_manager)
    initialize_manager_llm(memory_manager, char_manager)

    # Start necessary threads
    if CONFIG['CONTROLS']['enabled'] == 'True':
        bt_controller_thread = threading.Thread(target=start_bt_controller_thread, name="BTControllerThread", daemon=True)
        bt_controller_thread.start()

    # Create a thread for the Flask app
    if CONFIG['CHATUI']['enabled'] == "True":
        queue_message(f"LOAD: ChatUI starting on port 5012...")
        flask_thread = threading.Thread(target=modules.module_chatui.start_flask_app, daemon=True)
        flask_thread.start()
    
    # Initilize BLIP to speed up initial image capture
    if CONFIG['VISION']['server_hosted'] != "True":
        initialize_blip()
    
    try:
        queue_message(f"LOAD: TARS-AI v1.03a running.")
        # Start the STT thread
        stt_manager.start()

        while not shutdown_event.is_set():
            time.sleep(0.1) # Sleep to reduce CPU usage

    except KeyboardInterrupt:
        queue_message(f"INFO: Stopping all threads and shutting down executor...")
        shutdown_event.set()  # Signal global threads to shutdown
        # executor.shutdown(wait=True)

    finally:
        stt_manager.stop()
        bt_controller_thread.join()
        queue_message(f"INFO: All threads and executor stopped gracefully.")