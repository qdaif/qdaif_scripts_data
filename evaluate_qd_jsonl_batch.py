import os
import json
import math
import traceback
from typing import List, Optional, Sequence
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import asyncio
from pathlib import Path

from transformers import pipeline
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.evaluation import EvaluationRequest
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticRepresentation, AsyncClient

TOKEN = "TOKEN" # replace with your AA Client Token

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
        self.client = Client(token=TOKEN)

    async def evaluate_batch(self, generation_batch) -> list:
        eval_requests = []
        client_responses = []
        response_logprobs = []
        probability_fields = {}
        CONC_REQ = 50
        total_requests = None
        prompts = [self.feedback_template.format(**{"genotype": generation_text}) for generation_text in generation_batch]
        async with AsyncClient(token=TOKEN) as client:
            for prompt in prompts:
                for eval_answer in self.label_options:
                    request = EvaluationRequest(
                        prompt=Prompt.from_text(prompt),
                        completion_expected=eval_answer,
                    )
                    eval_requests.append(request)
            total_requests = len(eval_requests)
            for i in tqdm(range(0, total_requests, CONC_REQ)):
                start_idx = i
                end_idx = i + CONC_REQ
                requests = eval_requests[start_idx:end_idx]

                responses = await gather_with_concurrency(
                    CONC_REQ,
                    *(retry_request(client, req, model="luminous-supreme-qdaif") for req in requests),
                )
                client_responses.extend(responses)
            
        prob_results_list = []
        # given that logprobs are computed for both labels, two at a time per evaluation step in async batch, compute normalized prob scores for each sample
        for i in range(0, total_requests, 2):
            start_idx = i
            end_idx = i + 2
            response_logprobs = []
            for client_response in client_responses[start_idx:end_idx]:
                log_prob = client_response[2]['log_probability']
                response_logprobs.append(log_prob)

            response_logprobs = np.array(response_logprobs)

            normalized_probs = lognormalize(response_logprobs)

            for i, score in enumerate(normalized_probs):
                probability_fields[self.label_options[i]] = score
            prob_results_list.append(probability_fields[self.label_options[0]])

        return prob_results_list

# Helper for limiting number of requests at once
# Based on: https://blog.jonlu.ca/posts/async-python-http
async def gather_with_concurrency(n, *tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))

# Helper function for retrying requests in case of exceptions
async def retry_request(client, request, model):
    while True:
        try:
            response = await client.evaluate(request, model=model)
            return response
        except Exception as e:
            print(f"Exception caught: {e}. Retrying request...")
            await asyncio.sleep(1) # wait for a second before retrying

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

async def get_ai_feedback_client_results(ai_feedback, batch):
    prob_list = await ai_feedback.evaluate_batch(batch)
    return prob_list

def get_ai_feedback(answer_space: list[str], feedback_template: str):
    classifier_model = "luminous-supreme-qdaif"

    ai_feedback_pipeline = AIFeedback(
        classifier_model=classifier_model,
        label_options=answer_space,
        feedback_template=feedback_template,
    )
    return ai_feedback_pipeline


if __name__ == "__main__":
    history_paths = []
    
    # load path containing history.jsonl which require extended phenotype evals
    new_base_path = Path("data/histories_opinions_stories/baselines/stories_genre_ending/fixed_few_shot")
    history_paths_seeded = list(new_base_path.rglob("*.jsonl"))
    history_paths.extend(history_paths_seeded)

    for base_path_run in history_paths:
        # example: genre and ending evaluation for stories, quality feedback if needed too
        div_1_answer_space = [
            " \nhorror",
            " \nromance",
        ]
        div_2_answer_space = [
            " \nhappy ending",
            " \ntragedy",
        ]
        quality_answer_space = [
            " \nyes",
            " \nno",
        ]
        div_1_feedback_template = """### Instruction:\nWhat is the genre of this story? Reply with 'horror' or 'romance'\n\n### Input:{genotype}\n\n### Response:"""
        div_2_feedback_template = """### Instruction:\nYou are given an input text of a short story. Determine if the story has a happy ending or ends in a tragedy. Write 'happy ending' if the protagonist succeeds in his mission and lives a happy life, answer 'tragedy' if the protagonist fails to resolve the conflict and the world or characters in the story are doomed.\n\n### Input:{genotype}\n\n### Response:"""
        quality_feedback_template = """### Instruction:\nDetermine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Answer "yes" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer "no".\n\n### Input:{genotype}\n\n### Response:"""
        div_1_ai_feedback = get_ai_feedback(div_1_answer_space, div_1_feedback_template)
        div_2_ai_feedback = get_ai_feedback(div_2_answer_space, div_2_feedback_template)
        quality_ai_feedback = get_ai_feedback(quality_answer_space, quality_feedback_template)
        
        base_path_run_parent = Path(base_path_run.parent)
        with open(base_path_run, 'r') as f:
            data = [json.loads(line) for line in f]
        list_of_completions = []
        for line in data:
            list_of_completions.append(line['genotype'])
        non_empty_strings = [string for string in list_of_completions if string != ""]

        # async ai feedback client eval
        div_1_probability_list = asyncio.run(get_ai_feedback_client_results(div_1_ai_feedback, non_empty_strings))
        div_2_probability_list = asyncio.run(get_ai_feedback_client_results(div_2_ai_feedback, non_empty_strings))
        quality_probability_list = asyncio.run(get_ai_feedback_client_results(quality_ai_feedback, non_empty_strings))

        count = 0
        client_div_1_score_list = []
        client_div_2_score_list = []
        client_quality_score_list = []
        for i in range(len(list_of_completions)):
            if list_of_completions[i] != "":
                client_div_1_score_list.append(div_1_probability_list[count])
                client_div_2_score_list.append(div_2_probability_list[count])
                client_quality_score_list.append(quality_probability_list[count])
                count += 1
            else:
                client_div_1_score_list.append(None)
                client_div_2_score_list.append(None)
                client_quality_score_list.append(None)

        with open(base_path_run_parent / f"{base_path_run.stem}.jsonl", "w+", encoding="UTF-8") as f:
            for i, completion in enumerate(tqdm(list_of_completions)):
                dict_line_old = data[i]
                quality_sup_cont = client_quality_score_list[i]
                dict_line_old["fitness"] = quality_sup_cont
                dict_line_old["phenotype"] = [client_div_1_score_list[i], client_div_2_score_list[i]]

                f.write(json.dumps(dict_line_old))
                f.write("\n")
