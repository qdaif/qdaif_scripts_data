import json

def trace_chain(poem, poem_to_prev_mapping):
    chain = [poem]
    while poem_to_prev_mapping[poem] is not None:
        poem = poem_to_prev_mapping[poem]
        chain.append(poem)
    return chain

def extract_poems_from_file(file_path):
    with open(file_path, 'r') as file:
        records = [json.loads(line) for line in file.readlines()]
        return [record['poem'] for record in records]

if __name__ == "__main__":
    with open('data/histories_poetry/gpt_3_5/qdaif_lmx_guided/1/history.jsonl', 'r') as file:
        data = [json.loads(line) for line in file.readlines()]

    # create mapping of poem links
    poem_to_prev_mapping = {record['poem']: record['prev_poem'] for record in data}

    # determine longest rewriting chain
    longest_chain = []
    chains = []
    for poem in poem_to_prev_mapping:
        chain = trace_chain(poem, poem_to_prev_mapping)
        if len(chain) >= 5:
            chains.append(chain)
        if len(chain) > len(longest_chain):
            longest_chain = chain

    longest_chain_data = [record for record in data if record['poem'] in longest_chain]
    output_file_path = "poetry_longest_chain.jsonl"

    with open(output_file_path, 'w') as file:
        for record in longest_chain_data:
            file.write(json.dumps(record) + '\n')
