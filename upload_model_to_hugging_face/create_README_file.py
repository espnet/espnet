import pandas as pd
from espnet_model_zoo.downloader import ModelDownloader
import sys

def create_Readme_file(repo_name,model_name):
	# Fill in the blanks in the template Readme eg. add task tags, model name etc.
	df = pd.read_csv("table.csv")
	d = ModelDownloader()
	corpus_dict={}
	task_dict={}
	url_dict={}
	name_dict={}
	for row in df.values[1:]:
		corpus_dict[row[2]]=row[0]
		task_dict[row[2]]=row[1]
		url_dict[row[2]]=row[3].split("files/")[0]
		name_dict[row[2]]=row[2].split("/")[0]
	template_Readme=open("TEMPLATE_Readme.md")
	new_Readme=open(repo_name+"/README.md","w")
	lines_arr=[line for line in template_Readme]
	line_final_arr=[]
	for line in lines_arr:
		if "<add_more_tags>" in line:
			if task_dict[model_name]=="asr":
				line=line.replace("<add_more_tags>","automatic-speech-recognition")
			elif task_dict[model_name]=="tts":
				line=line.replace("<add_more_tags>","text-to-speech")
			elif task_dict[model_name]=="enh":
				line=line.replace("<add_more_tags>","speech-enhancement\n- audio-to-audio")
		line=line.replace("<add_model_name>",model_name)
		line=line.replace("<add_url>",url_dict[model_name])
		line=line.replace("<add_name>",name_dict[model_name])
		line=line.replace("<add_corpus>",corpus_dict[model_name])
		line=line.replace("<add_task_name>",task_dict[model_name].upper())
		new_Readme.write(line)

if __name__ == "__main__":
	repo_name=sys.argv[1]
	model_name=sys.argv[2].replace("\_"," ")
	create_Readme_file(repo_name,model_name)