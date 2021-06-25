from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader()
for model_name in d.query("name"):
	model_path = d.download(model_name)
	print(model_path+"||"+model_name.replace(" ","\_"))