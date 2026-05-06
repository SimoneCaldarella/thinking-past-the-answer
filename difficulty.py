import copy
import json
import os
import re

from utils import (
    Logger,
    build_run_path_components,
    difficulty_parse_args,
    is_debug_mode,
    setup_experiment_folder,
    setup_reproducibility_environment,
)
from modeling import load_model
from modeling.prompt_builder import build_prompt
from benchmarking import load_benchmark
from utils.ddp import setup_ddp_environment


def load_jsonl_outputs(file_path):
    """
    Load generated outputs from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing generated outputs
    Returns:
        List of records with questions, model answers, and metadata
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def split_with_granularity(text, tokenizer, difficulty_level="utterance", granularity=10):
    """
    Split the input text into progressively larger segments.

    Args:
        text: The input text to split
        tokenizer: The tokenizer to use for tokenization
        difficulty_level: The strategy to use for splitting ("utterance" or "token")
        granularity: The number of utterances or tokens per segment
    Returns:
        List of progressively larger text prefixes
    """
    if difficulty_level == "utterance":
        segments = text.split("\n")
        segments = ["\n".join(segments[i:i + granularity]) for i in range(0, len(segments), granularity)]
    elif difficulty_level == "token":
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        segments = [tokenizer.decode(token_ids[i:i + granularity]) for i in range(0, len(token_ids), granularity)]
    else:
        raise ValueError(
            f"Invalid difficulty level: {difficulty_level}. Must be 'utterance' or 'token'."
        )

    result = []
    for i in range(1, len(segments) + 1):
        if difficulty_level == "utterance":
            result.append("\n".join(segments[:i]))
        else:
            result.append(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:i * granularity]))

    return result


def resolve_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return processor


def build_base_messages(sample, model):
    images = [] if sample["decoded_image"] is None else [sample["decoded_image"]]
    prompt_sample = {
        "question": sample["query"],
        "images": images,
    }
    return build_prompt(
        sample=prompt_sample,
        system_prompt=model.system_prompt,
        reasoning_prompt=model.reasoning_prompt,
        prepend_reasoning_prompt=model.prepend_reasoning_prompt,
    )


def build_continuation_prompt(model, base_messages, assistant_prefix):
    continuation_messages = base_messages + [
        {
            "role": "assistant",
            "content": assistant_prefix,
        }
    ]

    try:
        return model.processor.apply_chat_template(
            continuation_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except ValueError as exc:
        if "continue_final_message is set but the final message does not appear" not in str(exc):
            raise

        prompt = model.processor.apply_chat_template(
            continuation_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=False,
        )
        tokenizer = resolve_tokenizer(model.processor)
        eos_token = getattr(tokenizer, "eos_token", None)

        if eos_token:
            prompt = re.sub(rf"{re.escape(eos_token)}\s*$", "", prompt)

        return prompt


def generate_from_prompt(model, prompt, images):
    sampling_params = copy.deepcopy(model.sampling_params)
    model_input = {"prompt": prompt}
    if images:
        model_input["multi_modal_data"] = {"image": images}
    outputs = model.model.generate(
        [model_input],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text


def build_assistant_prefix(sub_output, budget_forcing_prompt):
    if not sub_output:
        return budget_forcing_prompt
    return f"{sub_output.rstrip()} {budget_forcing_prompt}".lstrip()


def main(args):
    if is_debug_mode():
        Logger.info("Running in DEBUG MODE")

    setup_reproducibility_environment(args.seed)

    if args.backend == "ddp":
        setup_ddp_environment()

    if args.backend != "vllm":
        raise ValueError("difficulty.py currently supports only the vllm backend.")

    source_path_components = build_run_path_components(
        seed=args.seed,
        answer_extraction_model=args.answer_extraction_model,
        include_answer_extraction_model=args.path_include_answer_extraction_model or args.llm_endpoint != "unset",
        budget_forcing_prompt=args.path_budget_forcing_prompt or args.budget_forcing_prompt,
    )
    source_experiment_folder = setup_experiment_folder(
        output=args.output,
        model_name=args.model,
        benchmark_name=args.benchmark,
        experiment_name=args.experiment_name,
        path_components=source_path_components,
    )
    experiment_folder = setup_experiment_folder(
        output=args.output,
        model_name=args.model,
        benchmark_name=args.benchmark,
        experiment_name=args.experiment_name,
        path_components=build_run_path_components(
            seed=args.seed,
            answer_extraction_model=args.answer_extraction_model,
            include_answer_extraction_model=args.path_include_answer_extraction_model or args.llm_endpoint != "unset",
            budget_forcing_prompt=args.budget_forcing_prompt,
        ),
    )

    logger = Logger(
        experiment_folder=experiment_folder,
        configs=vars(args),
        use_wandb=args.use_wandb,
        wandb_name=args.wandb_name,
        wandb_project=args.wandb_project,
    )

    logger.info(f"Experiment folder set up at: {experiment_folder}")

    generations_file = f"{source_experiment_folder}/generations.jsonl"
    difficulty_file = f"{experiment_folder}/difficulty_generations.jsonl"

    if not os.path.exists(generations_file):
        logger.info(f"Generate outputs for the benchmark first before assessing difficulty. Missing: {generations_file}")
        exit(0)

    logger.info("Output file found. Assessing difficulty...")

    generation_records = load_jsonl_outputs(generations_file)
    benchmark = load_benchmark(args.benchmark, additional_args=args.additional_benchmark_args)

    model = load_model(
        model_name=args.model,
        backend=args.backend,
        max_tokens=args.max_tokens,
        max_num_images=args.max_num_images,
        seed=args.seed,
        additional_args=args.additional_model_args,
        override_system_prompt=args.override_system_prompt,
        override_reasoning_prompt=args.override_reasoning_prompt,
        override_prepend_reasoning_prompt=args.override_prepend_reasoning_prompt,
    )
    model.eval()

    tokenizer = resolve_tokenizer(model.processor)

    from evaluation.answer_extractor import AnswerExtractionPipeline

    assert args.llm_endpoint is not None, "LLM endpoint must be provided when using LLM-based answer extraction."

    if args.llm_endpoint == 'unset':
        logger.info("LLM endpoint not set. Skipping LLM-based answer extraction.")
        answer_extractor = None
    else:
        answer_extractor = AnswerExtractionPipeline(
            model_name=args.answer_extraction_model,
            llm_endpoint=args.llm_endpoint,
        )

    with open(difficulty_file, "w", encoding="utf-8") as f:
        for i, record in enumerate(generation_records):
            idx = record["idx"]
            raw_sample = benchmark.dataset[idx]
            processed_sample = benchmark.preprocess_samples([raw_sample])[0]
            question = record["question"]
            full_model_output = record["model_output"]
            base_messages = build_base_messages(processed_sample, model)
            split_outputs = [""] + split_with_granularity(
                full_model_output,
                tokenizer,
                difficulty_level=args.difficulty_level,
                granularity=args.granularity,
            )

            logger.info(
                f"Processing example {i + 1}/{len(generation_records)} "
                f"with {len(split_outputs)} difficulty levels"
            )

            for j, sub_output in enumerate(split_outputs):
                assistant_prefix = build_assistant_prefix(sub_output, args.budget_forcing_prompt)
                chat_template = build_continuation_prompt(model, base_messages, assistant_prefix)
                continuation = generate_from_prompt(
                    model=model,
                    prompt=chat_template,
                    images=[] if processed_sample["decoded_image"] is None else [processed_sample["decoded_image"]],
                )
                final_model_output = f"{assistant_prefix}{continuation}"
                
                if answer_extractor is not None:
                    parsed_answer = answer_extractor.extract_answer(
                        model_answer=final_model_output,
                    )
                else:
                    parsed_answer = None

                difficulty_record = {
                    "idx": idx,
                    "pid": processed_sample.get("pid", record.get("pid")),
                    "difficulty_idx": j,
                    "difficulty_level": args.difficulty_level,
                    "granularity": args.granularity,
                    "question": question,
                    "choices": record.get("choices", []),
                    "source_actual_query": record.get("actual_query"),
                    "actual_query": chat_template,
                    "sub_output": sub_output,
                    "budget_forcing_prompt": args.budget_forcing_prompt,
                    "model_output": final_model_output,
                    "model_continuation": continuation,
                    "model_parsed_answer": parsed_answer,
                    "ground_truth": processed_sample["answer"],
                    "source_model_output": full_model_output,
                }

                f.write(json.dumps(difficulty_record) + "\n")
                f.flush()

            if is_debug_mode() and i >= 30:
                break

    logger.info(f"Saved difficulty generations to: {difficulty_file}")

    logger.terminate_experiment()


if __name__ == "__main__":
    args = difficulty_parse_args()

    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    if args.show_models_and_benchmarks:
        from utils import show_available_models_and_benchmarks

        show_available_models_and_benchmarks(args.results_folder)
    else:
        main(args)
