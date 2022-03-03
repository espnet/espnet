def combine_type(input_file, output_file):
	for line in input_file:
		file_name=line.split()[0]
		str2=" ".join(line.split()[1:]).split("▁SEP")
		str1=str2[:-1]
		type_arr=[]
		name_arr=[]
		for k in str1:
			print(k)
			ent_type=k.split("▁fill")[0]
			name=k.split("▁fill")[1]
			ent_type=ent_type.replace(" ","").replace("▁","")
			type_arr.append(ent_type)
			name_arr.append(name)
		final_str=file_name+" "
		for k in range(len(type_arr)):
			final_str+=type_arr[k]+" ▁FILL"+name_arr[k]+"▁SEP "
		final_str=final_str+str2[-1]+"\n"
		output_file.write(final_str)

combine_type(open("data/train/text","r"),open("data/train/text_new","w"))
combine_type(open("data/devel/text","r"),open("data/devel/text_new","w"))
combine_type(open("data/test/text","r"),open("data/test/text_new","w"))
