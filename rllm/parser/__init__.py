from rllm.parser.tool_parser import Qwen3p5ToolParser, QwenToolParser, R1ToolParser, ToolParser

__all__ = [
    "ChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "Qwen3p5ChatTemplateParser",
    "QwenChatTemplateParser",
    "LlamaChatTemplateParser",
    "ToolParser",
    "R1ToolParser",
    "Qwen3p5ToolParser",
    "QwenToolParser",
]


def __getattr__(name):
    _chat_template_classes = {
        "ChatTemplateParser",
        "DeepseekQwenChatTemplateParser",
        "LlamaChatTemplateParser",
        "Qwen3p5ChatTemplateParser",
        "QwenChatTemplateParser",
    }
    if name in _chat_template_classes:
        import importlib

        mod = importlib.import_module("rllm.parser.chat_template_parser")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
    "qwen3p5": Qwen3p5ToolParser,
    "qwen3p6": Qwen3p5ToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {PARSER_REGISTRY}"
    return PARSER_REGISTRY[parser_name]
