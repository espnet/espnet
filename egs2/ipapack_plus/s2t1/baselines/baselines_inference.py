import torch
from espnet_basic_dataset import ESPnetBasicDataset

from allosaurus.app import read_recognizer
from allophant.dataset_processing import Batch
from allophant.estimator import Estimator
from allophant import predictions
import soundfile as sf
import tempfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class AllophantInference:
  def __init__(self, device='cpu', **kwargs):
    self.device= device
    self.model,self.attribute_indexer = Estimator.restore("kgnlp/allophant", device=self.device)
    self.inventory = self.attribute_indexer.phoneme_inventory(languages=kwargs.get('language'))
    self.feature_matrix=self.attribute_indexer.composition_feature_matrix(self.inventory).to(self.device)
    self.inventory_indexer = self.attribute_indexer.attributes.subset(self.inventory)
    self.feature_name = 'phoneme'
    self.decoder = predictions.feature_decoders(self.inventory_indexer, feature_names=[self.feature_name])[self.feature_name]

  def infer(self, input_batch):
    with torch.no_grad():
      audio_input = input_batch['wav']
      assert audio_input.shape[0] == 1, "Batch size > 1 not supported for inference!"
      batch = Batch(audio_input, torch.tensor([audio_input.shape[1]]), torch.zeros(1)).to(self.device)
      # .outputs['phoneme'] contains the logits over inventory
      model_outputs = self.model.predict(batch, self.feature_matrix)
      decoded = self.decoder(model_outputs.outputs[self.feature_name].transpose(1, 0), model_outputs.lengths)
      for [hypothesis] in decoded:
          recognized = self.inventory_indexer.feature_values(self.feature_name, hypothesis.tokens - 1)
          recognized = ''.join(recognized)
    return recognized


class AllosaurusInference:
    def __init__(self, device='cpu', **kwargs):
        self.device = device
        self.model = read_recognizer()        
        if device == 'cuda' and torch.cuda.is_available():
            self.model.config.device_id = 0
            self.model.am.to('cuda')
        else:
            self.model.config.device_id = -1  # CPU

    def infer(self, input_batch):
        with torch.no_grad():
            audiopath = input_batch['wavpath']
            if not audiopath.endswith('.wav'):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                    sf.write(temp_wav.name, input_batch['wav'].squeeze(0).numpy(), 16000)
                    transcription = self.model.recognize(temp_wav.name)
            else:
                transcription = self.model.recognize(audiopath)
        recognized = "".join(transcription).replace("͡", '').replace(" ", '')
        return recognized


class Wav2Vec2PhonemeInference:
    def __init__(self, device='cpu', **kwargs):
        self.device = device
        self.model_path = kwargs.get('model_path','facebook/wav2vec2-lv-60-espeak-cv-ft')
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        # Load model with appropriate dtype based on device
        if device == 'cuda' and torch.cuda.is_available():
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )
            self.dtype = torch.float16
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
            self.dtype = torch.float32
        
        self.model.eval().to(device)

    def infer(self, input_batch):
        audio = input_batch['wav'].squeeze(0).numpy()
        inputs = self.processor(
            audio,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            input_values = inputs.input_values.to(self.device)
            input_values = input_values.to(self.dtype)
            if hasattr(inputs, 'attention_mask'):
                attention_mask = inputs.attention_mask.to(self.device)
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        recognized = "".join(transcription)
        recognized = recognized.replace("͡", '').replace(" ", '')
        return recognized


def get_inference_model(model_name, device='cpu', **kwargs):
    if model_name == 'allophant':
        return AllophantInference(device=device, **kwargs)
    elif model_name == 'allosaurus':
        return AllosaurusInference(device=device, **kwargs)
    elif model_name in ['facebook/wav2vec2-lv-60-espeak-cv-ft',
                        'facebook/wav2vec2-xlsr-53-espeak-cv-ft',
                        'ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns']:
        return Wav2Vec2PhonemeInference(model_path=model_name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
  