# GitHub Repository Used for the "ICML 2024 Challenges on Automated Math Reasoning - Track 3: Automated Optimization Problem-Solving with Code" by AppliedAI (Spindox)

## Introduction

This project was developed for the "ICML 2024 Challenges on Automated Math Reasoning - Track 3: Automated Optimization Problem-Solving with Code." The objective is to enable large language models (LLMs) to handle complex mathematical optimization, encompassing problem formulation and solution.

The code is a pipeline designed to interpret a prompt, model a mathematical optimization problem, generate the related executable code, execute it, and collect the results. The pipeline is also capable of self-correcting errors in the execution phase.

It leverages Large Language Models (LLMs) to address mathematical reasoning tasks, encompassing the entire process from problem formulation to solution. By experimenting with various prompting techniques, including few-shot and self-consistency, the aim is to evaluate the current advanced capabilities of LLMs in mathematical problem-solving.

The technologies employed include several advanced techniques:

- **LLM API call**: Commercial and free Large Language Models have been tested to generate and correct code, leveraging their powerful natural language processing capabilities to understand and solve optimization problems. The current setting is OpenAI GPT-4-turbo.
- **Few-shot Learning** with **TF-IDF and Cosine Similarity Algorithm**: Employed to find the most similar example problems within the training dataset to include in the prompts to improve model performance.
- **Prompt Engineering**: Advanced configuration of prompts to obtain the best results from the model.
- **Feedback and Automatic Correction**: Implementation of a feedback loop to automatically correct the generated code.

For each prompt, several LLM calls are performed to generate candidate solutions. These calls are performed using different temperature values, and for each value, multiple calls are made. The number of calls increases according to the temperature value to exploit the creativity and randomness of LLMs, using the following formula:

$$n = 1 + \alpha \cdot T$$

where $\alpha$ is a custom parameter.

Among all the candidate solutions, two are selected using:

- **Majority Voting**: The final answer is chosen based on majority voting, where the solution that appears most frequently among the candidates is selected.
- **LLM Voting**: The LLM is prompted with the problem and each candidate solution and is asked to check their feasibility and select the one that optimizes the objective function.

## Project Usage

### Prerequisites

Before starting, ensure you have the following tools and libraries installed:

- Python 3.8 or higher
- The following Python libraries:
  - langchain
  - openai
  - PuLP
  - scikit-learn
  - PyYAML

For further information about the libraries, please refer to the `requirements.txt` file. To install the libraries, you can use the following command at the root of the project:

```bash
pip install -r requirements.txt
```

### Configuration

All functionalities of the module are configurable via the `params.yaml` file, located in the `params` folder found at the root of the project. Below is a description of the configurable parameters:

- `similarity`: Flag that activates or deactivates the text similarity function to perform few-shot learning.
- `similarity-threshold`: Threshold to accept a similar problem.
- `train-data-path`: Path to the training dataset for text similarity.
- `static-example`: Flag to activate few-shot learning with a fixed example.
- `temperatures`: List of temperature values to be tested in the run.
- `temp-coeff`: The $\alpha$ coefficient that determines the number of runs for each temperature value.
- `python-command`: The path of the Python interpreter to execute auto-generated code.
- `feedback-length`: Maximum number of iterations in the automatic correction process.

The default values for the parameters are the following:

```yaml
similarity: false
similarity-threshold: 0.7
train-data-path: "data/input/task3_train_public.json"
static-example: false
temperatures:
 - 0.0
 - 0.8
 - 2.0
temp-coeff: 2.0
python-command: "./.venv/bin/python"
feedback-length: 5
```

The `secret.yaml` file contains the OpenAI API key to call LLMs, which must be present to run the pipeline (located in the same directory as `params.yaml`).

```yaml
api-key: insert-key-here
```

### Running the Project

To run the project from the command line:

1.  Configure the `params.yaml` file with the desired parameters.
2.  Insert the API key in the `secret.yaml` file.
3.  Execute the `main.py` file with the following parameters:
    1.  The input file path containing the problems to solve.
    2.  The output file path, which will contain the solutions found using majority voting.
    3.  The output file path, which will contain the solutions found using LLM voting.

```bash
python3 main.py input_file_path output_majority_file_path output_llm_file_path
```

## Project Structure

-   `main.py`: Entry point of the pipeline, selects the prompt based on the parameters and invokes the `process_item` method for each problem, generating solutions and creating prediction files.
-   `code_from_llm.py`: Obtains Python code by invoking OpenAI models.
-   `code_with_feedback.py`: Executes and corrects the obtained code.
-   `llm_caller.py`: Contains the class to call OpenAI models with a generic prompt.
-   `prompts.py`: Contains all possible prompts used in the pipeline:
    -   `code_prompt` - Prompt without few-shot to obtain the Python code.
    -   `code_prompt_static_example` - Prompt with few-shot via a fixed example to obtain the Python code.
    -   `code_prompt_similarity` - Prompt with few-shot via similarity to obtain the Python code.
    -   `feedback_prompt` - Prompt to autocorrect wrong Python code.
    -   `check_optimal_solution_prompt` - Prompt to check which is the optimal solution (LLM voting).
    -   `get_optimal_solution_prompt` - Prompt to parse the LLM answer to retrieve the optimal solution in JSON format (LLM voting).
-   `solution_manager.py`: Manages the solutions found by LLMs, performing majority and LLM voting.
-   `text_similarity.py`: Computes the similarity between a given problem and the problems inside the train dataset.
-   `utils.py`: Utility functions for file management and output parsing.

## Fine-Tuning

Fine-tuning of GPT-3.5-turbo did not yield significant results. However, the notebook to perform fine-tuning is included as a separate `.ipynb` file, allowing for the creation of an additional model on the platform. This can be invoked by simply modifying the `model_name` within the code in the `llm_caller.py` class.