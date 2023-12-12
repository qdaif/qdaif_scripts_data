from functools import partial
import math
import traceback
from typing import Sequence, List
import asyncio
import os
import json
import random
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from copy import deepcopy

from rouge_score import rouge_scorer
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticRepresentation, AsyncClient, CompletionRequest, EvaluationRequest

CLIENT = Client(token="TOKEN") # replace with your AA Client Token

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

def construct_prompt(init_prompt_template, list_of_fewshot_reviews: list[str]) -> str:
    prompt = init_prompt_template + list_of_fewshot_reviews[0]
    for i in range(1, len(list_of_fewshot_reviews)):
        prompt += "\n###\n" + init_prompt_template + list_of_fewshot_reviews[i]
    prompt += "\n###\n" + init_prompt_template
    return prompt

def measure_aif(str_review, quality_ai_feedback):
    if str_review == "":
        return None
    feedback_scores = quality_ai_feedback.evaluate({"genotype": str_review})
    return float(feedback_scores[quality_ai_feedback.label_options[0]])

def map_to_even_bins(value, uneven_bins):
    # Generate evenly spaced bins
    even_bins = np.linspace(0, 1, len(uneven_bins))

    # Find the position of the input value in the uneven bins
    for i in range(len(uneven_bins) - 1):
        if uneven_bins[i] <= value <= uneven_bins[i + 1]:
            # Compute the relative position within the uneven bin
            proportion = (value - uneven_bins[i]) / (uneven_bins[i + 1] - uneven_bins[i])
            # Map this proportion to the even bin
            return even_bins[i] + proportion * (even_bins[i + 1] - even_bins[i])
    
    # For values outside the range, return an appropriate limit
    return 0 if value <= uneven_bins[0] else 1

def average_distance_knn(value_coord, coordinates_list, k):
    # Convert to numpy arrays for efficient computation
    value_coord = np.array(value_coord)
    coordinates_list = np.array(coordinates_list)

    # Compute distances using numpy broadcasting
    distances = np.sqrt(np.sum((coordinates_list - value_coord) ** 2, axis=1))

    # Adjust k if it's greater than the length of the list
    k = min(k, len(coordinates_list))

    # If k equals the length of the list, use all distances
    if k == len(coordinates_list):
        return np.mean(distances)
    else:
        # Find the k smallest distances
        k_nearest_distances = np.partition(distances, k)[:k]
        # Calculate the average of these distances
        return np.mean(k_nearest_distances)

# nsaif
def main(seed: str="0", domain: str="story_2d", init_div_threshold: float=0.05, quality_filter: bool=False, quality_threshold: float=0.8):
    NUM_COMPLETIONS = 2000
    POOL_SIZE = 100
    DOMAIN = domain
    QUALITY_FILTER = quality_filter
    div_2_ai_feedback = None # only for 2d domain
    div_threshold = init_div_threshold
    max_nn_k = 15 # from the paper
    num_rejects_before_adjust_threshold = 20 # roughly 1% of total steps, as in NS paper
    num_accepts_before_adjust_threshold = 2 # no clear number here
    
    if DOMAIN == "opinion":
        bins = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1]
        div_1_answer_space = [
            " \npositive",
            " \nnegative",
        ]
        div_1_feedback_template = """Determine the sentiment of the given opinion on eating vegetables and plant-based foods (from the input text) by writing "positive" or "negative" in the output."""
        div_1_ai_feedback = AIFeedback(
            classifier_model="luminous-supreme-qdaif",
            label_options=div_1_answer_space,
            feedback_template=f"### Instruction:\n{div_1_feedback_template}\n\n### Input:{{genotype}}\n\n### Response:",
        )
        quality_feedback_prompt = """Determine whether or not the input text is closely related to the following topic: "someone talking about whether or not they like to eat vegetables and plant-based foods as well as an explanation for their preferences". Answer "yes" if it is about the topic, or "no" if it is not about the topic."""
        prompt_template = "Here is a random opinion piece about eating vegetables and plant-based foods:"
        prompt_pool = [
            "Plant-based foods are a great source of healthy micronutrients, and can play a significant role in providing you with macronutrients also. I would highly recommend including many different foods such as vegetables and pulses in your regular diet.",
            "Vegetables taste quite bad, and I don't like eating them. I would much prefer eating meat and ice cream.",
            "I do not have an opinion on eating vegetables and other plant-based foods. I know that some people prefer a vegetarian or vegan diet, and others prefer eating meaty diets.",
        ]
        MAX_TOKENS = 50
    elif DOMAIN == "story_genre":
        bins = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1]
        div_1_answer_space = [
            " \nhorror",
            " \nromance",
        ]
        div_1_feedback_template = "What is the genre of this story? Reply with 'horror' or 'romance'"
        div_1_ai_feedback = AIFeedback(
            classifier_model="luminous-supreme-qdaif",
            label_options=div_1_answer_space,
            feedback_template=f"### Instruction:\n{div_1_feedback_template}\n\n### Input:{{genotype}}\n\n### Response:",
        )
        quality_feedback_prompt = """Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Answer "yes" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer "no"."""
        prompt_template = "Here is a random example of a fantasy story about a suspicious spy and a rich politician:"
        prompt_pool = [
            "A spy named Joanne wants to infiltrate the premises of Karl Johnson, a highly-influential figure in the city. Karl was a wealthy mayor, and would do anything in his power to suppress any opposing voices. Joanne wanted to figure out what Karl was hiding, but she took a turn for the worse, as she was highly suspicious in her presence outside his home.",
            "The wealthy entrepreneur and member of parliament, Susan, hosted a party at her mansion. She invited all of the residents, as well as an unusual looking man. The man, Dave, was wearing a tacky shirt, and star-shaped glasses, and was actually a spy. He made the whole room laugh with his jokes, and had a secret agenda - to find what Susan does in her private fun room!",
            "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        ]
        MAX_TOKENS = 100
    elif DOMAIN == "story_ending":
        bins = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1]
        div_1_answer_space = [
            " \nhappy ending",
            " \ntragedy",
        ]
        div_1_feedback_template = "You are given an input text of a short story. Determine if the story has a happy ending or ends in a tragedy. Write 'happy ending' if the protagonist succeeds in his mission and lives a happy life, answer 'tragedy' if the protagonist fails to resolve the conflict and the world or characters in the story are doomed."
        div_1_ai_feedback = AIFeedback(
            classifier_model="luminous-supreme-qdaif",
            label_options=div_1_answer_space,
            feedback_template=f"### Instruction:\n{div_1_feedback_template}\n\n### Input:{{genotype}}\n\n### Response:",
        )
        quality_feedback_prompt = """Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Answer "yes" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer "no"."""
        prompt_template = "Here is a random example of a fantasy story about a suspicious spy and a rich politician:"
        prompt_pool = [
            "A spy named Joanne wants to infiltrate the premises of Karl Johnson, a highly-influential figure in the city. Karl was a wealthy mayor, and would do anything in his power to suppress any opposing voices. Joanne wanted to figure out what Karl was hiding, but she took a turn for the worse, as she was highly suspicious in her presence outside his home.",
            "The wealthy entrepreneur and member of parliament, Susan, hosted a party at her mansion. She invited all of the residents, as well as an unusual looking man. The man, Dave, was wearing a tacky shirt, and star-shaped glasses, and was actually a spy. He made the whole room laugh with his jokes, and had a secret agenda - to find what Susan does in her private fun room!",
            "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        ]
        MAX_TOKENS = 100
    elif DOMAIN == "story_2d":
        bins = [0, 0.005, 0.02, 0.05, 0.20, 0.50, 0.80, 0.95, 0.98, 0.995, 1]
        div_1_answer_space = [
            " \nhorror",
            " \nromance",
        ]
        div_2_answer_space = [
            " \nhappy ending",
            " \ntragedy",
        ]
        div_1_feedback_template = "What is the genre of this story? Reply with 'horror' or 'romance'"
        div_2_feedback_template = "You are given an input text of a short story. Determine if the story has a happy ending or ends in a tragedy. Write 'happy ending' if the protagonist succeeds in his mission and lives a happy life, answer 'tragedy' if the protagonist fails to resolve the conflict and the world or characters in the story are doomed."
        div_1_ai_feedback = AIFeedback(
            classifier_model="luminous-supreme-qdaif",
            label_options=div_1_answer_space,
            feedback_template=f"### Instruction:\n{div_1_feedback_template}\n\n### Input:{{genotype}}\n\n### Response:",
        )
        div_2_ai_feedback = AIFeedback(
            classifier_model="luminous-supreme-qdaif",
            label_options=div_2_answer_space,
            feedback_template=f"### Instruction:\n{div_2_feedback_template}\n\n### Input:{{genotype}}\n\n### Response:",
        )
        quality_feedback_prompt = """Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Answer "yes" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer "no"."""
        prompt_template = "Here is a random example of a fantasy story about a suspicious spy and a rich politician:"
        prompt_pool = [
            "A spy named Joanne wants to infiltrate the premises of Karl Johnson, a highly-influential figure in the city. Karl was a wealthy mayor, and would do anything in his power to suppress any opposing voices. Joanne wanted to figure out what Karl was hiding, but she took a turn for the worse, as she was highly suspicious in her presence outside his home.",
            "The wealthy entrepreneur and member of parliament, Susan, hosted a party at her mansion. She invited all of the residents, as well as an unusual looking man. The man, Dave, was wearing a tacky shirt, and star-shaped glasses, and was actually a spy. He made the whole room laugh with his jokes, and had a secret agenda - to find what Susan does in her private fun room!",
            "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        ]
        MAX_TOKENS = 100

    # for getting eval done, and also as an optional quality filter approaching a QD algorithm not using MAP-Elites
    quality_ai_feedback = AIFeedback(
        classifier_model="luminous-supreme-qdaif",
        label_options=[f" \nyes", f" \nno",],
        feedback_template=f"### Instruction:\n{quality_feedback_prompt}\n\n### Input:{{genotype}}\n\n### Response:",
    )
    
    results_dict_lines = []
    pool = prompt_pool
    if DOMAIN == "story_2d":
        diversity_pool = [[item, map_to_even_bins(measure_aif(item, div_1_ai_feedback), bins), map_to_even_bins(measure_aif(item, div_2_ai_feedback), bins)] for item in prompt_pool]
        history_of_coords = [item[1:] for item in diversity_pool]
    else:
        diversity_pool = [[item, map_to_even_bins(measure_aif(item, div_1_ai_feedback), bins)] for item in prompt_pool]
        history_of_coords = [item[1:] for item in diversity_pool]
    
    # counters for accepts/rejects
    num_accept_in_row = 0
    num_reject_in_row = 0
    
    for i in tqdm(range(NUM_COMPLETIONS)):
        chosen_prompts = random.sample(pool, 3)
        prompt_string = construct_prompt(prompt_template, chosen_prompts)
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt_string),
            maximum_tokens=MAX_TOKENS,
            temperature=0.8,
            stop_sequences=["\n#", "\n##", "\n###", "###", "\n####", "\n#####", "####", "#####", "\n", "\n\n", "\n\n\n", "\n\n\n\n", "@@@", "#", "##", "\nHere", "\n\nHere"],
        )
        while True:
            try:
                response = CLIENT.complete(request, model="luminous-base")
                break
            except Exception:
                print("Error with AA API, retry")
        
        # get completion and eval diversity and map to equivalent uniform bins phenotype space
        completion = response.completions[0].completion
        if completion == "":
            fitness = None
            div_coord = None
        else:
            div_1 = measure_aif(completion, div_1_ai_feedback)
            div_1_transformed = map_to_even_bins(div_1, bins)
            if DOMAIN == "story_2d":
                div_2 = measure_aif(completion, div_2_ai_feedback)
                div_2_transformed = map_to_even_bins(div_2, bins)
                div_coord = [div_1, div_2]
                div_coord_transformed = [div_1_transformed, div_2_transformed]
            else:
                div_coord = [div_1]
                div_coord_transformed = [div_1_transformed]

            # measure fitness
            fitness = measure_aif(completion, quality_ai_feedback)
            
            # check the average distance to nearest neighbors
            mean_nn_dist = average_distance_knn(div_coord_transformed, history_of_coords, max_nn_k)

            # handle rejection/acceptance criteria
            if mean_nn_dist > div_threshold:
                # adjust threshold for novelty if needed
                num_accept_in_row += 1
                num_reject_in_row = 0
                if num_accept_in_row > num_accepts_before_adjust_threshold:
                    num_accept_in_row = 0
                    div_threshold *= 1.05 # increase level of novelty required
                
                # add to pools - novelty threshold is still adapted if completions deemed novel appear frequently, regardless of QAIF filter outcome
                if not QUALITY_FILTER or (QUALITY_FILTER and fitness > quality_threshold):
                    pool.append(completion)
                    if len(pool) > POOL_SIZE:
                        pool.pop(0)
                    if DOMAIN == "story_2d":
                        diversity_pool.append([completion, div_1_transformed, div_2_transformed])
                    else:
                        diversity_pool.append([completion, div_1_transformed])
                    history_of_coords.append(div_coord_transformed)
            else:
                # adjust threshold for novelty if needed
                num_reject_in_row += 1
                num_accept_in_row = 0
                if num_reject_in_row > num_rejects_before_adjust_threshold:
                    num_reject_in_row = 0
                    div_threshold *= 0.95 # decrease level of novelty required
        
        results_dict_lines.append({
            "genotype": completion,
            "fitness": fitness,
            "phenotype": div_coord,
        })

    threshold_str = str(init_div_threshold).replace(".", "_")
    quality_threshold_str = str(quality_threshold).replace(".", "_")
    with safe_open_w(f"baselines/nsaif/{DOMAIN}_nsaif_t_{threshold_str}/{seed}.jsonl", "w+", encoding="UTF-8") as f:
        for i in range(len(results_dict_lines)):
            f.write(json.dumps(results_dict_lines[i]))
            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed for rerun tracking", default="0")
    parser.add_argument("--domain", help="opinion or story", default="opinion")
    parser.add_argument("--threshold", type=float, help="novelty threshold", default=0.05)
    parser.add_argument("--q_threshold", type=float, help="quality threshold", default=0.8)
    args = parser.parse_args()
    print(args)

    quality_filter = False

    main(str(args.seed), str(args.domain), args.threshold, quality_filter, args.q_threshold)
