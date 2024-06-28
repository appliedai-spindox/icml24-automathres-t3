import json
import math

from langchain_core.prompts import PromptTemplate

from challenge import params
from challenge.code_from_llm import get_code_from_llm
from challenge.code_with_feedback import execute_code_with_feedback
from challenge.prompts import get_code_prompt, get_code_prompt_similarity, get_code_prompt_static_example
from challenge.solution_manager import get_best_solution_as_result, get_solution_count_dict
from challenge.text_similarity import TextSimilarity
from challenge.utils import read_args, read_json, write_json, get_empty_result


def get_prompt(item: dict, text_similarity: TextSimilarity) -> (PromptTemplate, dict):

    prompt, data = get_code_prompt(item['question'], item['results'])

    if params['static-example']:
        prompt, data = get_code_prompt_static_example(item['question'], item['results'])

    if text_similarity:
        most_similar_problem, similarity = text_similarity.get_most_similar_problem(item)
        if similarity >= params['similarity-threshold']:
            prompt, data = get_code_prompt_similarity(
                most_similar_problem['question'],
                most_similar_problem['results'],
                most_similar_problem['code'],
                item['question'],
                item['results']
            )

    return prompt, data


def process_item(item: dict, text_similarity: TextSimilarity) -> (dict, dict):

    prompt, data = get_prompt(item, text_similarity)

    print(f"Processing item {item['id']}")

    result_keys = list(item['results'].keys())

    solutions = {}
    for temperature in params['temperatures']:
        for it in range(1 + math.floor(temperature * params['temp-coeff'])):

            print(f"\tRun with temp. {temperature}, iteration {it}")

            code = get_code_from_llm(prompt, data, temperature=temperature)
            results_raw = execute_code_with_feedback(code)

            result = get_empty_result(result_keys)
            try:
                 result = json.loads(results_raw)
            except Exception as e:
                print(f"\tRun with temp. {temperature}, iteration {it} discarded due to errors: ", e)

            solutions[f"{temperature}-{it}"] = result



    solution_freq = get_solution_count_dict(solutions, result_keys)

    print(f"solutions and frequency: {solution_freq}")

    best_result_majority, best_result_llm = get_best_solution_as_result(solution_freq, result_keys, item['question'])

    return best_result_majority, best_result_llm


def process_data(test_data: list[dict], args: dict) -> (list[dict], list[dict]):

    text_similarity = None
    if params['similarity']:
        train_data = read_json(params['train-data-path'])
        text_similarity = TextSimilarity(train_data, test_data)

    output_majority = []
    output_llm = []

    for item in test_data:

        best_result_majority, best_result_llm = process_item(item, text_similarity)

        item_majority = {
            "id": item["id"],
            "question": item["question"],
            "results": best_result_majority
        }

        item_llm = {
            "id": item["id"],
            "question": item["question"],
            "results": best_result_llm
        }

        output_majority.append(item_majority)
        output_llm.append(item_llm)

        write_json(output_majority, args['output_majority_file_path'])
        write_json(output_llm, args['output_llm_file_path'])

    return output_majority, output_llm


def main():
    args = read_args()
    test_data = read_json(args['input_file_path'])
    output_majority, output_llm = process_data(test_data, args)
    write_json(output_majority, args['output_majority_file_path'])
    write_json(output_llm, args['output_llm_file_path'])


if __name__ == '__main__':
    main()
