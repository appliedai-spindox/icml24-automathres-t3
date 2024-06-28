import json
from collections import Counter

from challenge import llm
from challenge.prompts import get_get_optimal_solution_prompt, get_check_optimal_solution_prompt
from challenge.utils import extract_code_from_result


def from_solution_to_tuple(solution: dict, keys: list):

    mytuple = []

    for key in keys:

        if not key in solution:
            return []

        value = solution[key]

        if value is None:

            mytuple.append("0")

        elif isinstance(value, str):

            if value == "" or value == "None":
                mytuple.append("0")

            value = value.replace("$", "")

            try:
                value = float(value)
                mytuple.append(f"{value:.2f}")
            except ValueError:
                value = str.lower(value)
                if value in ["true", "yes"]:
                    mytuple.append("1")
                elif value in ["false", "not", "no"]:
                    mytuple.append("0")

        else:

            (mytuple.append(f"{float(value):.2f}"))

        mytuple = [val.replace("1.00", "1") for val in mytuple]
        mytuple = [val.replace("0.00", "0") for val in mytuple]

    return tuple(mytuple)


def get_solution_count_dict(solutions: dict, keys: list) -> dict:

    tuples = {}
    for solution in solutions.keys():
        tupl3 = from_solution_to_tuple(solutions[solution], keys)
        if tupl3:
            tuples[solution] = tupl3

    return dict(Counter(tuples.values()))


def from_tuple_to_result(tupl3: tuple, keys: list):

    result = {}
    for i in range(len(tupl3)):
        result[keys[i]] = tupl3[i]

    return result


def get_best_solution_by_llm(question: str, solutions: list[dict]) -> dict:

    prompt, data = get_check_optimal_solution_prompt(question, solutions)
    gpt_answer = llm.call_llm(prompt, data)

    prompt, data = get_get_optimal_solution_prompt(solutions, gpt_answer)
    results_raw = llm.call_llm(prompt, data)

    json_raw = extract_code_from_result(
        results_raw,
        language="json"
    )

    print(f"Solution found by LLM voting: {json_raw}")

    return json.loads(
        json_raw
    )


def get_best_solution_as_result(solution_freq: dict, keys: list, question: str) -> (dict, dict):

    best_majority_result = from_tuple_to_result(
        max(solution_freq, key=solution_freq.get),
        keys
    )

    if len(solution_freq) == 1:
        return best_majority_result, best_majority_result

    best_llm_result = get_best_solution_by_llm(
        question,
        [from_tuple_to_result(tupl3, keys) for tupl3 in solution_freq.keys()]
    )

    for key in keys:
        if key not in best_llm_result:
            return best_majority_result, best_majority_result

    return best_majority_result, best_llm_result
