"""
module_character.py

Character Management Module for TARS-AI Application.

This module manages character attributes and dynamic properties for the TARS-AI application.
"""

# === Standard Libraries ===
import json
from datetime import datetime
import configparser
import os

from modules.module_messageQue import queue_message

class CharacterManager:
    """
    Manages character attributes and dynamic properties for TARS-AI.
    """
    def __init__(self, config):
        self.config = config
        self.character_card_path =  os.path.join("..", self.config['CHAR']['character_card_path'])
        self.character_card = None
        self.char_name = None
        self.description = None
        self.personality = None
        self.world_scenario = None
        self.char_greeting = None
        self.example_dialogue = None
        self.voice_only = config['TTS']['voice_only']
        self.load_character_attributes()
        self.load_persona_traits()

    def load_character_attributes(self):
        """
        Load character attributes from the character card file specified in the config.
        """
        try:
            with open(self.character_card_path, "r") as file:
                data = json.load(file)

            self.char_name = data.get("char_name", "")
            self.description = data.get("description", "")
            self.personality = data.get("personality", "")
            self.scenario = data.get("scenario", "")
            self.char_greeting = data.get("first_mes", "")
            self.example_dialogue = data.get("mes_example", "")

            # Format the greeting with placeholders
            if self.char_greeting:
                self.char_greeting = self.char_greeting.replace("{{user}}", self.config['CHAR']['user_name'])
                self.char_greeting = self.char_greeting.replace("{{char}}", self.char_name)
                self.char_greeting = self.char_greeting.replace("{{time}}", datetime.now().strftime("%Y-%m-%d %H:%M"))

            self.character_card = f"\nDescription: {self.description}\n"\
                                  f"\nPersonality: {self.personality}\n"\
                                  f"\nWorld Scenario: {self.scenario}\n"#\
                                  #f"\nExample Dialog:\n{self.example_dialogue}\n"

            queue_message(f"LOAD: Character loaded: {self.char_name}")
        except FileNotFoundError:
            queue_message(f"ERROR: Character file '{self.character_card_path}' not found.")
        except Exception as e:
            queue_message(f"ERROR: Error while loading character attributes: {e}")

    def load_persona_traits(self):
        """
        Load persona traits from the persona.ini file.
        """
        persona_path =  os.path.join("..", 'character', self.char_name, 'persona.ini')
        config = configparser.ConfigParser()

        try:
            config.read(persona_path)
            if 'PERSONA' not in config:
                queue_message("ERROR: [PERSONA] section not found in persona.ini.")
                return

            self.traits = {key: int(value) for key, value in config['PERSONA'].items()}
            #queue_message("Traits loaded:", self.traits)
        except Exception as e:
            queue_message(f"ERROR: Error while loading persona traits: {e}")