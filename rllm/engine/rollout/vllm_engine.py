import asyncio
from uuid import uuid4

import ray
from transformers import AutoTokenizer, AutoProcessor

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.workers.config.rollout import RolloutConfig
from verl.workers.config.model import HFModelConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool
from rllm.workflows import TerminationEvent, TerminationReason


class vLLMEngine(RolloutEngine):
    """Standalone vLLM server and rollout engine..

    Usage:
        engine = await vLLMEngine.create("Qwen/Qwen3-VL-30B-A3B", tp=8)
        output = await engine.get_model_response(messages, temperature=0.7)
    """

    def __init__(
        self,
        server_handles: list,
        tokenizer,
        processor=None,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        sampling_params: dict | None = None,
        tools: list[Tool | dict] | None = None,
        accumulate_reasoning: bool = False,
        disable_thinking: bool = False,
        chat_parser=None,
    ):
        self.server_manager = AsyncLLMServerManager(config=None, server_handles=server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.sampling_params = sampling_params or {}
        self.tools = tools or []
        self.accumulate_reasoning = accumulate_reasoning
        self.chat_parser = chat_parser or ChatTemplateParser.get_parser(
            tokenizer, processor=processor, disable_thinking=disable_thinking
        )

    @classmethod
    async def create(
        cls,
        model_path: str,
        tp: int = 8,
        num_replicas: int = 1,
        gpus_per_node: int = 8,
        tokenizer=None,
        processor=None,
        chat_parser=None,
        **kwargs,
    ) -> "vLLMEngine":
        """Create a vLLMEngine, starting the vLLM server.

        Args:
            model_path: HuggingFace model path
            tp: Tensor parallel size (GPUs per replica)
            num_replicas: Number of independent model replicas
            gpus_per_node: Total GPUs available
            tokenizer: Optional tokenizer (auto-loaded from model_path if None)
            processor: Optional processor (auto-loaded from model_path if None)
            chat_parser: Optional chat parser (auto-created from tokenizer if None)
            **kwargs: Additional arguments passed to __init__
        """
        server_handles = await start_server(model_path, tp, num_replicas, gpus_per_node)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                processor = None

        return cls(server_handles=server_handles, tokenizer=tokenizer, processor=processor, chat_parser=chat_parser, **kwargs)

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        request_id = kwargs.pop("request_id", uuid4().hex)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        # Merge sampling params
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        # Convert tools to JSON format
        if tools:
            tools = [tool.json if isinstance(tool, Tool) else tool for tool in tools]

        # Parse messages to prompt
        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Handle images
        image_data = None
        if hasattr(self.chat_parser, "process_image_data"):
            try:
                image_data = self.chat_parser.process_image_data(messages)
                if image_data:
                    model_inputs = self.processor(text=[prompt], images=image_data)
                    prompt_ids = model_inputs["input_ids"][0]
            except Exception as e:
                print(f"Image processing error: {e}")                                        

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # Generate
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params={
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "top_k": sampling_params.get("top_k", -1),
                "max_tokens": max_tokens,
                "logprobs": sampling_params.get("logprobs", 1),
            },
            image_data=image_data,
        )

        completion_ids = output.token_ids
        logprobs = output.log_probs

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]
            logprobs = logprobs[:max_tokens] if logprobs else None

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        parsed = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed["content"],
            reasoning=parsed["reasoning"],
            tool_calls=parsed["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )


async def start_server(model_path: str, tp: int = 8, num_replicas: int = 1, gpus_per_node: int = 8) -> list:
    """Start vLLM server(s)."""
    if not ray.is_initialized():
        ray.init()

    assert tp * num_replicas <= gpus_per_node, f"tp={tp} * num_replicas={num_replicas} exceeds gpus_per_node={gpus_per_node}"

    rollout_config = RolloutConfig(
        name="vllm",
        load_format="auto",
        tensor_model_parallel_size=tp,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        enable_sleep_mode=False,
        free_cache_engine=False,
    )

    model_config = HFModelConfig(path=model_path)

    replicas = []
    for i in range(num_replicas):
        replica = vLLMReplica(
            replica_rank=i,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=tp,
        )
        replicas.append(replica)

    await asyncio.gather(*[r.init_standalone() for r in replicas])

    for r in replicas:
        print(f"Server ready: {r._server_address}")

    return [r._server_handle for r in replicas]
