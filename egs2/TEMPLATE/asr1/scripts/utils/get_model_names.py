from espnet_model_zoo.downloader import ModelDownloader
import sys

model_name = sys.argv[1]
d = ModelDownloader()
model_path = d.download(sys.argv[1])
print(model_path)
