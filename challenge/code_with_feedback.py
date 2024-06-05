from challenge import params, llm
from challenge.prompts import get_feedback_prompt
from challenge.utils import extract_code_from_result
import subprocess


def execute_code_with_feedback(code: str) -> str:

    i = -1
    while i < params['feedback-length']:

        with open("temp_code.py", "w") as file:
            file.write(code)

        output = subprocess.run([params['python-command'], "temp_code.py"], capture_output=True, text=True)

        if not output.stderr:
            break

        print(f"feedback iteration {i + 1}")

        prompt, data = get_feedback_prompt(code, output.stderr)
        code = extract_code_from_result(
            llm.call_llm(prompt, data)
        )

        i += 1

    if not output.stdout:
        return "Error."

    return output.stdout
