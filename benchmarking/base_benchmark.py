from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from utils.custom_logging import Logger
from utils import is_debug_mode
from evaluation.parsing import ParsingHelper
import json
from rich.progress import track
import os

if TYPE_CHECKING:
    from modeling.base_model import BaseModel

class BaseBenchmark(ABC):
    def __init__(self, 
                 benchmark_name: str,
                 additional_args: Optional[str] = None
                 ):
        self.benchmark_name = benchmark_name
        
        if additional_args:
            self.parse_additional_args(additional_args)
        self.load_benchmark()
    
    @classmethod                      
    def get_url(cls) -> str:
        if getattr(cls, "get_hf_name", None) is not None:
            return cls.get_hf_name()
        else:
            return "NotApplicable"
    
    @abstractmethod
    def preprocess_samples(self, samples: list[dict]) -> list[dict]:
        """Convert a benchmark document to text input for the model.
        
        Here the sample has to contain at least the following fields:
        - "query": The main question or prompt to be answered.
        - "decoded_image": The image data associated with the query.
        - "pid": The unique identifier for the sample.
        - "answer": The ground truth answer for evaluation.
        - "choices": (Optional) A list of multiple-choice options if applicable.
        """
        pass

    def parse_additional_args(self, additional_args: str):
        """Parse additional arguments specific to the benchmark."""
        parsed_args = {v[0]: v[1] for v in (arg.split('=') for arg in additional_args.split(';') if additional_args != "")}
        
        for key, value in parsed_args.items():
            setattr(self, key, value)

    def load_benchmark(self):
        """Load the benchmark dataset here."""
        pass
        
    def generate_outputs(self, model: "BaseModel", output_file: Optional[str]) -> list[dict[str, str]]:

        with open(output_file, "w") as f:
            for i, sample in enumerate(self.dataset):
                Logger.info(f"Processing example {i+1}/{len(self.dataset)}")

                processed_sample = self.preprocess_samples([sample])[0]
                images = [] if processed_sample["decoded_image"] is None else [processed_sample["decoded_image"]]

                # TODO: Batch not supported yet
                output_texts, messages = model.generate_batch([{"question": processed_sample["query"], 
                                                                "images": images}], 
                                                                return_inputs=True)

                # ---- SAVE OUTPUT ----
                record = {
                    "idx": i,
                    "pid": processed_sample["pid"],
                    "question": processed_sample["query"],
                    "choices": processed_sample.get("choices", []),
                    "actual_query": messages[0],
                    "model_output": output_texts[0],
                    "ground_truth": processed_sample["answer"],
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                if is_debug_mode():
                    if i >= 30:
                        break
    
    def evaluate(self, model: "BaseModel" = None, output_file: str = None, parsed_responses_filename: str = "parsed_responses_only.jsonl", include_difficulty: bool = False, show_progress: bool = False) -> dict[str, float]:
        """Evaluate the model on the benchmark and return metrics."""
        correct = 0
        total = 0

        choices_map = ["a", "b", "c", "d", "e", "f", "g", "h", "i", 
                        "j", "k", "l", "m", "n", "o", "p", "q", "r", 
                        "s", "t", "u", "v", "w", "x", "y", "z"]

        response_list = []

        with open(output_file, "r") as f:

            if show_progress:
                total_lines = sum(1 for _ in f)
                f.seek(0)
                records_iter = track(f, total=total_lines)
            else:
                records_iter = f
            
            for line in records_iter:

                record = json.loads(line)

                out = record["model_output"]
                pred = record["model_parsed_answer"] if "model_parsed_answer" in record else None
                gt = record["ground_truth"]

                if pred is None and model is not None:
                    pred = model.normalize_answer(out, clean_only=False)
                    parsed_gt = model.normalize_answer(gt, clean_only=True)

                pred = ParsingHelper.clean(pred)
                parsed_gt = ParsingHelper.clean(gt)
                
                print(record["choices"])

                if record["choices"] and pred != parsed_gt:
                    choices = record["choices"]
                    ground_truth = record["ground_truth"]
                    ground_truth = ParsingHelper.clean(ground_truth)
                    choices = [ParsingHelper.clean(choice) for choice in choices]
                    correct_idx = choices.index(ground_truth)
                    correct_choice = choices_map[correct_idx]
                    correct_choice = ParsingHelper.clean(correct_choice)
                    parsed_gt = correct_choice
                
                if pred == parsed_gt or pred in parsed_gt: 
                    correct += 1

                total += 1

                if include_difficulty:
                    response_list.append({"idx": record["idx"],
                                          "prediction": pred,
                                          "ground_truth": parsed_gt,
                                          "difficulty_idx": record.get("difficulty_idx"),
                                          "granularity": record.get("granularity"),
                                          "budget": record.get("difficulty_idx") * record.get("granularity")})
                else:
                    response_list.append({"idx": record["idx"],
                                          "prediction": pred,
                                          "ground_truth": parsed_gt})

                

        accuracy = correct / total * 100

        with open(os.path.dirname(output_file) + f"/{parsed_responses_filename}", "w") as f:
            for response in response_list:
                f.write(json.dumps(response) + "\n")

        return {"accuracy": accuracy}
