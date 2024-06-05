from challenge import llm
from challenge.utils import extract_code_from_result


def get_code_from_llm(prompt: str, data: dict, temperature: float) -> str:
    return extract_code_from_result(
        llm.call_llm(prompt, data, temperature=temperature)
    )
