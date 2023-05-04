#!/usr/bine/env python3
import sys

from espnet_model_zoo.downloader import ModelDownloader

model_name = sys.argv[1]
d = ModelDownloader()
model_path = d.download(sys.argv[1])
print(model_path)
