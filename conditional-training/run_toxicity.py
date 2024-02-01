import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from detoxify import Detoxify
from datasets import load_dataset
import argparse
import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='hails/cond-410m-20btoks')
parser.add_argument('--num_samples', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--special_token', type=str, default='none')
args = parser.parse_args()

print(args)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to('cuda')
tokenizer.pad_token = tokenizer.eos_token

# Load the real-toxicity-prompts dataset
dataset = load_dataset("allenai/real-toxicity-prompts", split='train')

# Initialize Detoxify model for toxicity detection
detoxify_model = Detoxify('unbiased', device='cuda' if torch.cuda.is_available() else 'cpu')

# Special token IDs 
SPECIAL_TOKENS = {
    'toxic': 50278,
    'not-toxic': 50277
}

# Function to prepend a special token to the inputs
def prepend_special_token(inputs, special_token_id):
    # Create a tensor of the special token ID with the same batch size as the inputs
    special_tokens_tensor = torch.full((inputs['input_ids'].size(0), 1), special_token_id).to(inputs['input_ids'].device)
    
    # Concatenate the special token tensor to the start of the input IDs
    inputs['input_ids'] = torch.cat((special_tokens_tensor, inputs['input_ids']), dim=1)
    
    # Adjust the attention mask for the new token
    if 'attention_mask' in inputs:
        special_attention_tensor = torch.ones_like(special_tokens_tensor)
        inputs['attention_mask'] = torch.cat((special_attention_tensor, inputs['attention_mask']), dim=1)
    
    return inputs

# Function to generate completions and calculate the average toxicity
def generate_and_evaluate_toxicity(dataset, model, tokenizer, detoxify_model, num_samples=100, batch_size=10, special_token="none"):
    all_toxicity_scores = []
    dataset = dataset.select(range(min(num_samples, len(dataset)))) # Limit the dataset to num_samples

    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_prompts = [item["text"] for item in dataset[i:i+batch_size]['prompt']]
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda')

        # If a special token is provided, prepend it to the input IDs
        if special_token and special_token in SPECIAL_TOKENS:
            inputs = prepend_special_token(inputs, SPECIAL_TOKENS[special_token])
        elif special_token != "none":
            raise ValueError(f"Special token {special_token} not found in SPECIAL_TOKENS")

        # Generate completions using the model
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_return_sequences=1, bad_words_ids=[[token] for token in SPECIAL_TOKENS.values()],
                                    do_sample=True, temperature=0.9)

        # Process each item in the batch
        for output in outputs:
            completion = tokenizer.decode(output, skip_special_tokens=True)

            # Measure toxicity with Detoxify
            results = detoxify_model.predict(completion)
            avg_toxicity = results['toxicity']

            all_toxicity_scores.append(avg_toxicity)

    return np.array(all_toxicity_scores)

# Call the function and print the average toxicity
all_toxicity_scores = generate_and_evaluate_toxicity(dataset, model, tokenizer, detoxify_model, batch_size=args.batch_size, num_samples=args.num_samples, special_token=args.special_token)
average_toxicity = np.mean(all_toxicity_scores)
std_toxicity = np.std(all_toxicity_scores)
median_toxicity = np.median(all_toxicity_scores)
q_25 = np.quantile(all_toxicity_scores, 0.25)
q_75 = np.quantile(all_toxicity_scores, 0.75)
print(args)
print(f"Average toxicity: {average_toxicity}")
print(f"Standard deviation: {std_toxicity}")
print(f"Median toxicity: {median_toxicity}")
print(f"25th percentile: {q_25}")
print(f"75th percentile: {q_75}")
# save the results
np.save(f"{args.model}_{args.num_samples}_{args.batch_size}_{args.special_token}.npy".replace('/', '_'), all_toxicity_scores)