import asyncio
import os
import json
import random
from tqdm import tqdm

from aleph_alpha_client import AsyncClient, Client, CompletionRequest, Prompt

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

# fixed-few-shot and shuffling-few-shot baselines
async def main():
    token = "TOKEN" # replace with your AA Client Token
    domain = "opinions" # "opinions" or "stories"

    NUM_COMPLETIONS = 10000
    CONC_REQ = 50
    MAX_TOKENS = 100 # 50 (tokens) for "opinions" experiments
    TEMP = 0.8
    FIXED_PROMPT = True # set to True to run fixed-few-shot baseline, False to run shuffling-few-shot baseline

    if domain == "opinions":
        prompt_template = "Here is a random opinion piece about eating vegetables and plant-based foods:"
        prompt_pool = [
            "Plant-based foods are a great source of healthy micronutrients, and can play a significant role in providing you with macronutrients also. I would highly recommend including many different foods such as vegetables and pulses in your regular diet.",
            "Vegetables taste quite bad, and I don't like eating them. I would much prefer eating meat and ice cream.",
            "I do not have an opinion on eating vegetables and other plant-based foods. I know that some people prefer a vegetarian or vegan diet, and others prefer eating meaty diets.",
        ]
    elif domain == "stories":
        prompt_template = "Here is a random example of a fantasy story about a suspicious spy and a rich politician:"
        prompt_pool = [
            "A spy named Joanne wants to infiltrate the premises of Karl Johnson, a highly-influential figure in the city. Karl was a wealthy mayor, and would do anything in his power to suppress any opposing voices. Joanne wanted to figure out what Karl was hiding, but she took a turn for the worse, as she was highly suspicious in her presence outside his home.",
            "The wealthy entrepreneur and member of parliament, Susan, hosted a party at her mansion. She invited all of the residents, as well as an unusual looking man. The man, Dave, was wearing a tacky shirt, and star-shaped glasses, and was actually a spy. He made the whole room laugh with his jokes, and had a secret agenda - to find what Susan does in her private fun room!",
            "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        ]
    
    # same order of in-context examples as order of the prompt pool list
    fixed_prompt_string = construct_prompt(prompt_template, prompt_pool)
    
    async with AsyncClient(token=token, nice=False) as client:
        for i in tqdm(range(int(NUM_COMPLETIONS / CONC_REQ))):
            if FIXED_PROMPT:
                stop_sequences = ["\n#", "\n##", "\n###", "###", "\n####", "\n#####", "####", "#####", "\n", "\n\n", "\n\n\n", "\n\n\n\n", "@@@", "#", "##", "\nHere", "\n\nHere"]
                # Lots of requests to execute
                requests = (
                    CompletionRequest(
                        prompt=Prompt.from_text(fixed_prompt_string),
                        maximum_tokens=MAX_TOKENS,
                        temperature=TEMP,
                        stop_sequences=stop_sequences,
                    )
                    for _ in range(CONC_REQ)
                )
            else:
                batch_prompts = []
                for _ in range(CONC_REQ):
                    chosen_prompts = random.sample(prompt_pool, 3)
                    prompt_string = construct_prompt(prompt_template, chosen_prompts)
                    batch_prompts.append(prompt_string)
                requests = (
                    CompletionRequest(
                        prompt=Prompt.from_text(batch_item_string),
                        maximum_tokens=MAX_TOKENS,
                        temperature=TEMP,
                        stop_sequences=stop_sequences,
                    )
                    for batch_item_string in batch_prompts
                )
            responses = await gather_with_concurrency(
                CONC_REQ,
                *(retry_request(client, req, model="luminous-base") for req in requests),
            )

            completions = []
            for response in responses:
                completions.append({"genotype": response[1][0].completion})

            with safe_open_w("history.jsonl", "a+", encoding="UTF-8") as f:
                for j in range(len(completions)):
                    f.write(json.dumps(completions[j]))
                    f.write("\n")


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
            response = await client.complete(request, model=model)
            return response
        except Exception as e:
            print(f"Exception caught: {e}. Retrying request...")
            await asyncio.sleep(1) # wait for a second before retrying

if __name__ == "__main__":
    asyncio.run(main())