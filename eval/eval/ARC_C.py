import re
import pandas as pd
from gsm8k import GSM8KDataset
from datasets import Dataset as HFDataset
import os
from parsers import Parser
import json

ARC_SYSTEM_PROMPT = """Solve the ABCD multiple-choice question step by step.
Output your final answer only in \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.
"""

class ARC_CDataset(GSM8KDataset):

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=False,
        system_prompt=ARC_SYSTEM_PROMPT,
        subsample=256,
    ):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = f"{cur_path}/../dataset/arc_challenge.jsonl" 
        super().__init__(tokenizer, num_examples, add_reasoning, ARC_SYSTEM_PROMPT, subsample)

    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        self.dataset = []
        with open(self.data_path, 'r') as f:
            for line in f:
                self.dataset.append(json.loads(line)) 


    def create_prompt(self, input_text):

        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\n##Question: {input_text}\n##Answer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return user_input

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        question = str(self.dataset[self.subsample[idx].item()]["problem"])
        solution = str(self.dataset[self.subsample[idx].item()]["solution"])

        prompt = self.create_prompt(ARC_SYSTEM_PROMPT+"\n\n"+question+"\n\n##Answer:")
        return prompt, question, solution