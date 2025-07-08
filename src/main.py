import sys
from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel

from results.Results import Results

from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type=str, 
    default="HumanEval", 
    choices=[
        "HumanEval", 
        "MBPP", 
        "APPS",
        "xCodeEval", 
        "CC", 
    ]
)
parser.add_argument(
    "--strategy", 
    type=str, 
    default="MapCoder", 
    choices=[
        "Direct",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
    ]
)
parser.add_argument(
    "--model", 
    type=str, 
    default="ChatGPT", 
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
    ]
)
parser.add_argument(
    "--temperature", 
    type=float, 
    default=0
)
parser.add_argument(
    "--pass_at_k", 
    type=int, 
    default=1
)
parser.add_argument(
    "--language", 
    type=str, 
    default="Python3", 
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)

args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
TEMPERATURE = args.temperature
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language

RUN_NAME = f"{MODEL_NAME}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE}-{PASS_AT_K}"
RESULTS_PATH = f"./outputs/{RUN_NAME}.jsonl"

print(f"#########################\nRunning start {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

# your any instruction
snake_game_instruction = {
    "task_id": "Custom/snake_game",
    "prompt": """
Please write a Python program to implement the classic Snake game.
The game should run in the console (terminal) and use a library like `curses` for screen drawing and keyboard input.

Game Requirements:
1.  The game area should be a rectangular box.
2.  The snake starts as a small segment and moves in one direction.
3.  The player can control the snake's direction using the arrow keys. The snake cannot immediately reverse its direction.
4.  Food appears at a random location on the screen.
5.  When the snake eats the food, it grows longer, and the score increases.
6.  The game ends if the snake collides with the screen boundaries or with itself.
7.  Display the current score on the screen.
""",
    "entry_point": "main",
    "canonical_solution": "",
    "test": "",
    "sample_io": []
}



original_dataset = DatasetFactory.get_dataset_class("HumanEval")()


original_dataset.data = [snake_game_instruction]



strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_NAME)(temperature=TEMPERATURE),
    data=original_dataset, # changed
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
)


strategy.run()

print(f"#########################\nRunning end {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

