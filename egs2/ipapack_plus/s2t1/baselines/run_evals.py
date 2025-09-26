"""Run evaluation with baselines.
Usage:
    python baselines/run_evals.py \
        --dataset buckeye \
        --model allosaurus \
        [--reuse]
"""

import os
import panphon.distance
from tqdm import tqdm
import json
import torch

from egs2.ipapack_plus.s2t1.baselines.baselines_inference import get_inference_model
from egs2.ipapack_plus.s2t1.baselines.espnet_basic_dataset import get_inference_dataset

def run_inference(dataset, model, device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = get_inference_dataset(dataset)
    model = get_inference_model(model, device=device)
    print(f"Running inference on {len(dataset)} utterances using {model.__class__.__name__} on {device}")

    test_data = {}
    for i,item in enumerate(tqdm(dataset)):
        pred = model.infer(item)
        test_data[item['key']] = {
            'key': item['key'], 
            'transcription': item['transcription'], 
            'prediction': pred,
            'wavpath': item['wavpath']
        }
        # if i>5: break
    return test_data

def load_json(results_file):
    """Load saved results from file"""
    with open(results_file, 'r') as f:
        json_data = json.load(f)
    return json_data

def save_json(data, out_file):
    """Save data to a json file"""
    with open(out_file, 'w') as f:
        json.dump(data, f , indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")
    return

def get_metrics(hyps, refs):
    ###############################################################
    # formatting: make them phone sequence
    cleaner = {"ẽ": "ẽ", "ĩ": "ĩ", "õ": "õ", "ũ": "ũ", # nasal unicode
                "ç": "ç", "g": "ɡ", # common unicode
                "-": "", "'": ""} # noise
    def clean(phones):
        return "".join([cleaner.get(p, p) for p in phones])
    ###############################################################
    hyps = [clean(x) for x in hyps]
    refs = [clean(x) for x in refs]
    print(hyps, refs)
    dst = panphon.distance.Distance()
    FED = dst.feature_edit_distance(hyps, refs)
    FER = dst.feature_error_rate(hyps, refs)
    PER = dst.phoneme_error_rate(hyps, refs)
    return {"FER": FER*100, "FED": FED*100, "PER": PER*100}

def compute_metrics(test_data):
    """Compute PFER metrics"""
    hyps, refs =[], []
    for _, value in test_data.items():
        hyps.append(value['prediction'])
        refs.append(value['transcription'])
    metrics = get_metrics(hyps, refs)
    return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Phoneme recognition evaluation')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--model', help='Model name for inference')
    parser.add_argument('--device', default='auto', help='Device to run inference on (e.g., cpu, cuda, auto)')
    parser.add_argument('--output_dir', default='./preds', help='Directory to save results')
    parser.add_argument('--reuse_predictions', action='store_true', help='Use saved results instead of running inference')
    parser.add_argument('--skip_metric_computation', action='store_true', help='Skip metric computation step')

    args = parser.parse_args()
    prediction_file = f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}/preds.json"
    result_file = f"{args.output_dir}/{args.dataset}.{args.model.replace('/','.')}/metrics.json"
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    if args.reuse_predictions:
        print(f"Loading: {prediction_file}")
        test_data = load_json(prediction_file)
    else:
        if os.path.exists(prediction_file):
            raise RuntimeError(f"Warning: {prediction_file} already exists!")
        print(f"Running: {args.model}")
        test_data = run_inference(args.dataset, args.model, args.device)
        save_json(test_data, prediction_file)

    if args.skip_metric_computation:
        return
    
    # Compute and display results
    metrics = compute_metrics(test_data)
    print(f"\n{args.model} on {args.dataset}")
    print(f"PFER: {metrics['PFER mean']:.3f} ± {metrics['PFER std']:.3f}")
    save_json({**metrics, 'model': args.model, 'dataset': args.dataset}, result_file)

if __name__ == "__main__":
    main()