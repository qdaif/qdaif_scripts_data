import math
import traceback
from typing import Sequence, List
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticRepresentation, AsyncClient, CompletionRequest, EvaluationRequest
import asyncio
import os
import json
import random
from tqdm import tqdm
import numpy as np

CLIENT = Client(token="TOKEN", nice=False)

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

class AIFeedback:
    def __init__(
        self, 
        classifier_model: str, 
        label_options: List[str],
        feedback_template: str, 
    ):
        """
        Class which defines pipeline setup for general completion of instructions, 
        and evaluating classifications for AI feedback

        Args:
            classifier_model: checkpoint used to evaluate likelihoods of AI feedback attributes from LM predicting feedback label options
            label_options: multiple choice answers for feedback classification
            feedback_template: prompt template to feed to classifier model
        """
        self.classifier_model = classifier_model
        self.feedback_template = feedback_template
        self.label_options = label_options
        self.aa_client = CLIENT

    def evaluate(self, generation_text):
        eval_requests = []
        client_responses = []
        response_logprobs = []
        probability_fields = {}
        prompt = self.feedback_template.format(**generation_text)
        try:
            while True:
                try:
                    for eval_answer in self.label_options:
                        request = EvaluationRequest(
                            prompt=Prompt.from_text(prompt),
                            completion_expected=eval_answer,
                        )
                        eval_requests.append(request)
                    for eval_request in eval_requests:
                        response = self.aa_client.evaluate(eval_request, model=self.classifier_model)
                        client_responses.append(response)

                    for client_response in client_responses:
                        log_prob = client_response[2]['log_probability']
                        response_logprobs.append(log_prob)

                    response_logprobs = np.array(response_logprobs)

                    normalized_probs = lognormalize(response_logprobs)

                    for i, score in enumerate(normalized_probs):
                        probability_fields[self.label_options[i]] = score

                    return probability_fields
                except Exception: 
                    print("Error with AA API, retry")
                    traceback.print_exc()
        except KeyboardInterrupt:
            print("killed")
            raise


def safe_open_w(path, *args, **kwargs):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, *args, **kwargs)

def construct_prompt(init_prompt_template, list_of_fewshot_examples: list[str]) -> str:
    prompt = init_prompt_template + list_of_fewshot_examples[0]
    for i in range(1, len(list_of_fewshot_examples)):
        prompt += "\n###\n" + init_prompt_template + list_of_fewshot_examples[i]
    prompt += "\n###\n" + init_prompt_template
    return prompt

def measure_fitness(completion, quality_ai_feedback):
    if completion == "":
        return None
    feedback_scores = quality_ai_feedback.evaluate({"genotype": completion})
    return float(feedback_scores[quality_ai_feedback.label_options[0]])


# random-search and lmx, quality-only baselines
def main():
    NUM_COMPLETIONS = 2000
    MAX_TOKENS = 50
    POOL_SIZE = 100 # only for quality-only baseline
    QUALITY_OPT = True # set to false for random-search, true for quality-only baseline
    ZERO_SHOT = True # initialize prompt pool from zero-shot completions, mainly for quality-only baseline extension

    domain = "opinions" # "opinions" or "stories"

    stop_sequences = ["\n#", "\n##", "\n###", "###", "\n####", "\n#####", "####", "#####", "\n", "\n\n", "\n\n\n", "\n\n\n\n", "@@@", "#", "##", "\nHere", "\n\nHere"]

    if domain == "opinions":
        quality_feedback_prompt = """Determine whether or not the input text is closely related to the following topic: "someone talking about whether or not they like to eat vegetables and plant-based foods as well as an explanation for their preferences". Answer "yes" if it is about the topic, or "no" if it is not about the topic."""
        prompt_template = "Here is a random opinion piece about eating vegetables and plant-based foods:"
        prompt_pool = [
            "Plant-based foods are a great source of healthy micronutrients, and can play a significant role in providing you with macronutrients also. I would highly recommend including many different foods such as vegetables and pulses in your regular diet.",
            "Vegetables taste quite bad, and I don't like eating them. I would much prefer eating meat and ice cream.",
            "I do not have an opinion on eating vegetables and other plant-based foods. I know that some people prefer a vegetarian or vegan diet, and others prefer eating meaty diets.",
        ]
    elif domain == "stories":
        quality_feedback_prompt = """Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Answer "yes" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer "no"."""
        prompt_template = "Here is a random example of a fantasy story about a suspicious spy and a rich politician:"
        prompt_pool = [
            "A spy named Joanne wants to infiltrate the premises of Karl Johnson, a highly-influential figure in the city. Karl was a wealthy mayor, and would do anything in his power to suppress any opposing voices. Joanne wanted to figure out what Karl was hiding, but she took a turn for the worse, as she was highly suspicious in her presence outside his home.",
            "The wealthy entrepreneur and member of parliament, Susan, hosted a party at her mansion. She invited all of the residents, as well as an unusual looking man. The man, Dave, was wearing a tacky shirt, and star-shaped glasses, and was actually a spy. He made the whole room laugh with his jokes, and had a secret agenda - to find what Susan does in her private fun room!",
            "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        ]
    
    # for quality-only baseline
    quality_ai_feedback = AIFeedback(
        classifier_model="luminous-supreme-qdaif",
        label_options=[f" \nyes", f" \nno",],
        feedback_template=f"### Instruction:\n{quality_feedback_prompt}\n\n### Input:{{genotype}}\n\n### Response:",
    )

    if ZERO_SHOT:
        prompt_pool = [] # initialize prompt pool from scratch
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt_template),
            maximum_tokens=MAX_TOKENS,
            temperature=0.8,
            stop_sequences=stop_sequences,
        )
        # initialize 10 items for pool
        for i in range(10):
            while True:
                try:
                    response = CLIENT.complete(request, model="luminous-base")
                except Exception:
                    print("Error with AA API, retry")
                completion = response.completions[0].completion
                if completion != "": # non-empty completion
                    if completion[-1] != ".":
                        completion += "." # add extra full stop to ensure few-shot examples always end with stop
                    prompt_pool.append(completion)
                    break # once client successfully returns non-empty completion, do next item for generation of pool items
                else:
                    pass

    results_dict_lines = []
    if QUALITY_OPT:
        pool = [(item, measure_fitness(item, quality_ai_feedback)) for item in prompt_pool]
    else:
        pool = prompt_pool
    
    for i in tqdm(range(NUM_COMPLETIONS)):
        if type(pool[0]) == str:
            chosen_prompts = random.sample(pool, 3)
        else:
            pool_strings = [pool[i][0] for i in range(len(pool))]
            chosen_prompts = random.sample(pool_strings, 3)
        prompt_string = construct_prompt(prompt_template, chosen_prompts)
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt_string),
            maximum_tokens=MAX_TOKENS,
            temperature=0.8,
            stop_sequences=stop_sequences,
        )
        while True:
            try:
                response = CLIENT.complete(request, model="luminous-base")
                break
            except Exception:
                print("Error with AA API, retry")

        completion = response.completions[0].completion
        if completion != "" and QUALITY_OPT:
            fitness = measure_fitness(completion, quality_ai_feedback)
        else:
            fitness = None
        
        if QUALITY_OPT: # maintain pool so that least fit items are discarded
            if fitness is not None:
                pool.append((completion, fitness))
                pool.sort(key=lambda x: x[1]) # if new solution is less fit than existing lowest solution, it's always discarded
                if len(pool) > POOL_SIZE:
                    pool.pop(0)
        else:
            pool.append(completion)
        
        results_dict_lines.append({
            "genotype": completion,
            "fitness": fitness
        })

    with safe_open_w("history.jsonl", "a+", encoding="UTF-8") as f:
        for i in range(len(results_dict_lines)):
            f.write(json.dumps(results_dict_lines[i]))
            f.write("\n")

if __name__ == "__main__":
    main()