from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader()
for model_name in d.query("name")[10:]:
	model_path = d.download(model_name)
	print(model_path.replace("%2B","_")+"||"+model_name.replace(" ","\_"))