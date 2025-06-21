# GTZAN Dataset Music Genre Classification Recipe

This recipe implements the audio classification task with a BEATs encoder and linear layer decoder model on the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset. 

	@misc{tzanetakis_essl_cook_2001,
	author    = "Tzanetakis, George and Essl, Georg and Cook, Perry",
	title     = "Automatic Musical Genre Classification Of Audio Signals",
	url       = "http://ismir2001.ismir.net/pdf/tzanetakis.pdf",
	publisher = "The International Society for Music Information Retrieval",
	year      = "2001"
	}

We reuse part of the code from the [BEATs repository](https://github.com/microsoft/unilm/tree/master/beats) for this implementation.

# Trained checkpoints
Fine-tuned checkpoint is available at:
* https://huggingface.co/espnet/BEATs-AS20K , mAP: 37.5




