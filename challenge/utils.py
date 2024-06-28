import json
import sys

import yaml


def read_params() -> dict:
    with open("data/params/params.yaml", "r") as file:
        return yaml.safe_load(file)


def read_args() -> dict:

    if len(sys.argv) <= 3:
        print("Usage: python3 main.py input_file_path output_majority_file_path output_llm_file_path")
        exit(1)

    return {
        "input_file_path": sys.argv[1],
        "output_majority_file_path": sys.argv[2],
        "output_llm_file_path": sys.argv[3],
    }


def read_json(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)


def write_json(data: list[dict], file_path: str):
    with open(file_path, "w") as file:
        json.dump(data, file)


def extract_code_from_result(result, language: str = "python") -> str:
    if not f"```{language}" in result:
        return result
    start_index = result.find(f"```{language}") + len(f"```{language}")
    end_index = result.find("```", start_index)
    return result[start_index:end_index]


def get_empty_result(result_keys: list) -> dict:
    return {key: "0" for key in result_keys}