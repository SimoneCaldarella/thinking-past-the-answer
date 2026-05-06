from typing import Any, List, Union, Optional
from abc import abstractmethod
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
from PIL import Image
from modeling.prompt_builder import build_prompt
from vllm import LLM, SamplingParams
from utils.custom_logging import Logger
import torch
import os

from modeling.base_model import BaseModel

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28



class HFModel(BaseModel):
    text_only = False

    def eval(self):
        if self.backend != "vllm":
            self.model.eval()

    def load_model(self):
        self.hf_name = self.get_hf_name()

        print(f"Loading HF model {self.hf_name}...")

        if self.backend == "vllm":
            if torch.are_deterministic_algorithms_enabled():
                Logger.info(
                    "Disabling torch deterministic algorithms before vLLM "
                    "initialization to avoid Inductor compile-time benchmarking "
                    "failures. Generation seeding remains enabled."
                )
                torch.use_deterministic_algorithms(False)

            # Adapt the parallel to the number of gpus available
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                num_gpus = len(cuda_visible.split(","))
            else:
                num_gpus = 4

            Logger.info(f"Using tensor_parallel={num_gpus}")

            if "qwen3" in self.hf_name.lower():
                gpu_uti = 0.85
            else:
                gpu_uti = 0.90
            gpu_uti = float(getattr(self, "gpu_memory_utilization", gpu_uti))

            llm_kwargs = {
                "model": self.hf_name,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "tensor_parallel_size": num_gpus,
                "gpu_memory_utilization": gpu_uti,
                "seed": self.seed,
            }
            if "qwen3-vl" in self.hf_name.lower():
                qwen3vl_cudagraph_mode = getattr(
                    self, "qwen3vl_cudagraph_mode", "FULL_DECODE_ONLY"
                ).upper()
                Logger.info(
                    f"Using Qwen3-VL cudagraph_mode={qwen3vl_cudagraph_mode}."
                )
                llm_kwargs["compilation_config"] = {
                    "cudagraph_mode": qwen3vl_cudagraph_mode
                }

            if not self.text_only:
                llm_kwargs.update(
                    {
                        "max_model_len": 16392,
                        "limit_mm_per_prompt": {"image": 1, "video": 0},
                        "mm_processor_cache_gb": 0,
                        "mm_encoder_tp_mode": "data",
                    }
                )

            self.model = LLM(**llm_kwargs)

            self.sampling_params = SamplingParams(
                **self.sampling_params,
                max_tokens=self.max_tokens,
                seed=self.seed
            )

        elif self.backend == "ddp":
            raise NotImplementedError("DDP backend is not implemented yet.")
        
        elif self.backend == "hf":

            if self.text_only:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.hf_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
        
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported.")

        if self.text_only:
            self.processor = AutoTokenizer.from_pretrained(
                self.hf_name,
                trust_remote_code=True
            )
        elif "qwen2" in self.hf_name.lower():
            self.processor = AutoProcessor.from_pretrained(
                self.hf_name,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                trust_remote_code=True
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                self.hf_name,
                trust_remote_code=True
            )

    def preprocess_inputs(self, samples: List[dict[str, Any]] ) -> List[dict[str, Any]]:
        """Preprocess input samples and build prompts."""
        return [self.processor.apply_chat_template(build_prompt(sample=sample, 
                                                                system_prompt=self.system_prompt,
                                                                reasoning_prompt=self.reasoning_prompt,
                                                                prepend_reasoning_prompt=self.prepend_reasoning_prompt),
                                                   tokenize=False, 
                                                   add_generation_prompt=True) 
                for sample in samples]

    def vllm_generate(self, samples: List[dict[str, Any]], return_inputs: bool = False) -> List[str]:
        """Generate outputs using vLLM backend."""
        # 1. Build prompts for each sample
        messages = self.preprocess_inputs(samples)

        # 2. Prepare inputs for vLLM
        vllm_inputs = []
        for message, sample in zip(messages, samples):
            vllm_input = {
                "prompt": message,
            }
            if sample["images"]:
                vllm_input["multi_modal_data"] = {"image": sample["images"]}
            vllm_inputs.append(vllm_input)
        
        # 3. Generate using vLLM
        outputs = self.model.generate(vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False)
        
        # 4. Extract generated text from outputs
        if return_inputs:
            return [output.outputs[0].text.strip() for output in outputs], messages
        else:
            return [output.outputs[0].text.strip() for output in outputs]

    def generate_batch(self, samples: List[dict[str, Any]], return_inputs: bool = False) -> Union[List[str], tuple[List[str], List[dict[str, Any]]]]:
        """Generate outputs based on the backend."""
        if self.backend == "vllm":
            return self.vllm_generate(samples=samples, return_inputs=return_inputs)
        
        elif self.backend == "ddp":
            raise NotImplementedError("DDP backend is not implemented yet.")
        
        else:
            raise NotImplementedError(f"{self.backend} generate_batch method is not implemented yet.")
