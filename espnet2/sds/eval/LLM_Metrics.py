from multiprocessing import Pool
from espnet2.sds.eval.vert import get_self_bleu2_geometric, get_auto_bleu2_geometric, run_f
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gmean
def perplexity(LLM_Output):
    try:
        import evaluate
    except Exception as e:
        print("Error: evaluate is not properly installed.")
        raise e
    # import pdb;pdb.set_trace()
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(model_id='gpt2',predictions=[LLM_Output])
    return f"Perplexity: {results['mean_perplexity']:.2f}\n"

def vert(LLM_response_arr):
    # import pdb;pdb.set_trace()
    terms = [x.strip().split() for x in LLM_response_arr]


    tasks = [
        ('Self-BLEU2-geometric', get_self_bleu2_geometric),
        ('Auto-BLEU2-geometric', get_auto_bleu2_geometric),
    ]
    n_processes = min(16, len(tasks))
    with Pool(n_processes) as pool:
        metrics = pool.map(run_f, [(t[1], terms) for t in tasks])
    metric_arr=[]
    str1=""
    for (metric_name, _), metric in zip(tasks, metrics):
        metric, sem = np.mean(metric), np.std(metric) / np.sqrt(len(metric))

        metric, sem = [
            round(100 * x, 2) for x in [metric, sem]
        ]
        metric_arr.append(metric)

        str1+=(f'{metric_name}: {metric}\n')
    str1+=(f'VERT: {round(100*gmean(metric), 2)}\n')
    return str1

def bert_score(total_response_arr):
    # import pdb;pdb.set_trace()
    def cosine_similarity_context_response(context, response, model, tokenizer):
        # Tokenize and encode both context and response
        context_inputs = tokenizer(context, return_tensors="pt", truncation=True)
        response_inputs = tokenizer(response, return_tensors="pt", truncation=True)
        for k in context_inputs:
            context_inputs[k]=context_inputs[k].cuda()
        for k in response_inputs:
            response_inputs[k]=response_inputs[k].cuda()

        # Get embeddings from the model
        with torch.no_grad():
            context_embedding = model(**context_inputs).last_hidden_state.mean(dim=1)
            response_embedding = model(**response_inputs).last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        similarity = cosine_similarity(context_embedding.cpu().numpy(), response_embedding.cpu().numpy())
        return similarity[0][0]

    bert_model_name = "bert-base-uncased"
    bert_model = AutoModel.from_pretrained(bert_model_name).cuda()
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    similarity = cosine_similarity_context_response(" ".join(total_response_arr[:-1]), total_response_arr[-1], bert_model, bert_tokenizer)
    return (f"Cosine Similarity: {similarity*100:.2f}"+"\n")

def DialoGPT_perplexity(user_utterance, response):
    # import pdb;pdb.set_trace()
    def evaluate_response_with_dialoGPT(context, response, model, tokenizer):
        """
        Evaluate the appropriateness of a response based on the given context using DialoGPT.

        Args:
            context (str): The dialogue context (previous conversation).
            response (str): The generated response to evaluate.
            model: Pre-trained DialoGPT model.
            tokenizer: Corresponding tokenizer for the DialoGPT model.

        Returns:
            float: Perplexity score of the response given the context.
        """
        model.eval()
        
        # Combine context and response as input
        input_text = context + tokenizer.eos_token + response + tokenizer.eos_token
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs['input_ids']=inputs['input_ids'].cuda()
        inputs['attention_mask']=inputs['attention_mask'].cuda()
        # import pdb;pdb.set_trace()
        
        # Compute model outputs and loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"].cuda())
            loss = outputs.loss

        # Calculate perplexity
        perplexity = torch.exp(loss)
        return perplexity.cpu().item()

    # Load DialoGPT model and tokenizer
    model_name = "microsoft/DialoGPT-medium"  # Choose small/medium/large based on your resources
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    perplexity = evaluate_response_with_dialoGPT(user_utterance, response, model, tokenizer)
    return (f"DialoGPT Perplexity: {perplexity:.2f}"+"\n")

