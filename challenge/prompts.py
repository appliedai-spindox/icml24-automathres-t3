from langchain_core.prompts import PromptTemplate

prompts = {

    "code_prompt": '''You are an expert in Operations Research, Python programming language, and the PuLP library. 
You are also proficient in JSON file format. 
I will provide you a description of an optimization problem in natural language and I need you to write a Python script using PuLP to solve that specific problem instance and find the optimal solution. 
Don't write anything before and after the code, it will be simple for me to parse your answer.
The code must only print a single string to stdout, formatted in JSON, which contains the answers to the question I will give you in input.
Here is the problem description:
{test_question}

Here is the question to answer:
{test_results}

Please ensure your solution follows these guidelines and addresses all specified points from both the problem description and the question to answer.
Guidelines:
1. Define the decision variables clearly based on the problem description, paying attention to whether the variables should be integer (e.g., when variables represent finite items), binary, or continuous.
2. Formulate the objective function and constraints accurately.
3. Use 'pulp.LpMaximize' or 'pulp.LpMinimize' as appropriate for the objective.
4. Implement constraints correctly, considering any special conditions:
   - If strict inequalities are present, such as \( x > y \), transform them into \( x \geq y + 1 \) for integer and binary variables, or \( x \geq y + 0.0001 \) for continuous variables.
5. Ensure all logical and physical constraints from the problem description are incorporated:
   - For tiered constraints (e.g., different values or costs at different levels), ensure correct implementation.
   - Pay attention to special conditions like non-overlapping categories, specific setups, or unique constraints.
6. Solve the problem with `problem.solve(PULP_CBC_CMD(msg=0))` to suppress the solver's log.
7. Format the output as a JSON string with the required answers, ensuring all specified outputs in the test_results are addressed.''',

    "code_prompt_static_example": '''You are an expert in Operations Research, Python programming language, PuLP library and JSON file format. 
I will give you a description in natural language of an optimization problem and you just have to write a Python script using PuLP to solve that specific instance 
of the problem. Don't write anything before and after the code, it will be simple for me to parse your answer. The code must only print a single string to stdout, formatted in JSON, which contains the 
answers to the question I will give you in input (be sure to insert all the results into the JSON output).
Now I will give you an example (it is just an example) and then I will prompt you with the problem I want to solve and the question I want to answer.
Please be extremely accurate in formulating the problem.

Example: 
Problem (input 1): A company is organizing a team-building event and needs to assign participants to different activities. They have a total of 100 employees. Activity A requires 5 employees a team, activity B requires 3 employees a team, and activity C requires 7 employees a team. The company has a total of 100 employees available for the event. The company has a limitation on the number of teams in activity B, which cannot exceed 20. The company wants to maximize participation teams and decides to allocate different weights to each activity: activity A has a weight of 3, activity B has a weight of 2, and activity C has a weight of 4. The objective is to maximize the total participation weighted by the assigned weights.

Question to answer (input 2):
{{
    "The number of teams in activity A": "",
    "The number of teams in activity B": "",
    "The number of teams in activity C": "",
    "The total weighted participation": ""
}}

Code to retrieve the answers (output):
# Import PuLP library
from pulp import *

# Define the decision variables
num_participants_A = LpVariable("NumParticipantsA", lowBound=0, cat='Integer') # number of participants in activity A
num_participants_B = LpVariable("NumParticipantsB", lowBound=0, upBound=20, cat='Integer') # number of participants in activity B
num_participants_C = LpVariable("NumParticipantsC", lowBound=0, cat='Integer') # number of participants in activity C

# Define the question as a maximum or minimum problem
problem = LpProblem("TeamBuildingEvent", LpMaximize)

# Define the objective function
objective = 3 * num_participants_A + 2 * num_participants_B + 4 * num_participants_C
problem += objective # maximize the total participation weighted by the assigned weights

# Define the constraints
problem += 5 * num_participants_A + 3 * num_participants_B + 7 * num_participants_C <= 100 # the total number of employees is 100

# Solve the problem
status = problem.solve()

# Output the answer
print("## start solving")
print("The number of participants in activity A:", num_participants_A.value())
print("The number of participants in activity B:", num_participants_B.value())
print("The number of participants in activity C:", num_participants_C.value())
print("The total weighted participation:", objective.value())
print("## end solving")


My question:
Problem:
{test_question}
Question to answer:
{test_results}

Be careful: if there are strict inequality constraints such as x>y then use x>= y +1 for integer and binary variables, use x>= 1e-4 for continuous variables instead.
Please remember to solve the problem with status = problem.solve(PULP_CBC_CMD(msg=0)) in order to suppress solver's log.
If your problem needs you to decide whether to activate some resources that has fixed costs, evaluate the opportunity to add some other variable to model this condition.
In the output all the results value must be printed as strings and all binary variables must be set to True or False (not 1 and 0 or true and false lower case).''',

    "code_prompt_similarity": '''You are an expert in Operations Research, Python programming language, PuLP library and JSON file format. 
I will give you a description in natural language of an optimization problem and you just have to write a Python script using PuLP to solve that specific instance 
of the problem. Don't write anything before and after the code, it will be simple for me to parse your answer. The code must only print a single string to stdout, formatted in JSON, which contains the 
answers to the question I will give you in input (be sure to insert all the results into the JSON output).
Now I will give you an example (it is just an example) and then I will prompt you with the problem I want to solve and the question I want to answer.
Please be extremely accurate in formulating the problem.

Example: 
Problem (input 1): {train_question}

Question to answer (input 2):
{train_results}

Code to retrieve the answers (output):
{train_code}

My question:
Problem:
{test_question}
Question to answer:
{test_results}

Be careful: if there are strict inequality constraints such as x>y then use x>= y +1 for integer and binary variables, use x>= 1e-4 for continuous variables instead.
Please remember to solve the problem with status = problem.solve(PULP_CBC_CMD(msg=0)) in order to suppress solver's log.
If your problem needs you to decide whether to activate some resources that has fixed costs, evaluate the opportunity to add some other variable to model this condition.
In the output all the results value must be printed as strings and all binary variables must be set to True or False (not 1 and 0 or true and false lower case).''',

    "feedback_prompt": """
I have a code that gives me the following error:

Error (Input 1): {stderr}

(end of Input 1).

The code is the following:

Code (Input 2): {code}

(end of Input 2).

I need you to correct it without writing anything else but the code.
Don't write anything before and after the code, it will be simpler for me to parse your answer. 
Don't change neither a word in the final prints as they are the results I'm interested in.""",

    "check_optimal_solution_prompt": '''You are an expert in Operations Research, Python programming language and you are also proficient in JSON file format. 
I have the description of an optimization problem and a set of different solutions, and I need you to establish which solution is the best one.
Please also verify the feasibility of every constraint taking into account also the nature of the variables (continuous, integer or binary) and discard the infeasible solutions.

Here is the problem description:

{test_question}

Here is the set of different solutions:

{solutions}

Note: For constraints involving multiplicative relationships with zero, assume any non-negative number satisfies the constraint when the multiplier is zero.''',

    "get_optimal_solution_prompt": """Hi, I have an optimization problem and a set of different solutions and I asked chat gpt which is the best one, provided the problem description.
Now, I will provide you gpt's answer (that already estabished which solution is the best one) and I need you to provide to me only the best solution without writing anything else.

Here is the set of different solutions:

{solutions}

Here is the gpt's answer: (beginning of gpt's answer)

{gpt_answer}
(end of gpt's answer)

Please write only the optimal solution determined by gpt (in the original format) in JSON format inside a JSON Markdown block. Remember that the JSON format requires double quotes to enclose keys and values"""

}

def get_code_prompt(test_question: str, test_results: str) -> (PromptTemplate, dict):

    data = {
        "test_question": test_question,
        "test_results": test_results
    }

    return PromptTemplate(template=prompts['code_prompt'], input_variables=list(data.keys())), data


def get_code_prompt_static_example(test_question: str, test_results: str) -> (PromptTemplate, dict):

    data = {
        "test_question": test_question,
        "test_results": test_results
    }

    return PromptTemplate(template=prompts['code_prompt_static_example'], input_variables=list(data.keys())), data


def get_code_prompt_similarity(train_question: str, train_results: str, train_code: str, test_question: str, test_results: str) -> (PromptTemplate, dict):

    data = {
        "train_question": train_question,
        "train_results": train_results,
        "train_code": train_code,
        "test_question": test_question,
        "test_results": test_results
    }

    return PromptTemplate(template=prompts['code_prompt_similarity'], input_variables=list(data.keys())), data


def get_feedback_prompt(code: str, stderr: str) -> (PromptTemplate, dict):

    data = {
        "code": code,
        "stderr": stderr
    }

    return PromptTemplate(template=prompts['feedback_prompt'], input_variables=list(data.keys())), data


def get_check_optimal_solution_prompt(test_question: str, solutions: list[dict]) -> (PromptTemplate, dict):

    data = {
        "test_question": test_question,
        "solutions": solutions
    }

    return PromptTemplate(template=prompts['check_optimal_solution_prompt'], input_variables=list(data.keys())), data


def get_get_optimal_solution_prompt(solutions: list[dict], gpt_answer: str) -> (PromptTemplate, dict):

    data = {
        "solutions": solutions,
        "gpt_answer": gpt_answer
    }

    return PromptTemplate(template=prompts['get_optimal_solution_prompt'], input_variables=list(data.keys())), data
