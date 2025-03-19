import json
import kaldiio

class DialogueJsonReader:
    def __init__(self, path):
        self.data = dict()
        
        uttids, json_files = dict(), dict()
        for line in open(path):
            uttid, json_file = line.strip().split()
            uttids[uttid] = None
            json_files[json_file] = None
        
        for json_file in json_files:
            json_dict = json.load(open(json_file, 'r', encoding='utf-8'))
            for uttid, content in json_dict.items():
                if uttid in uttids:
                    self.data[uttid] = content

    def __getitem__(self, name):
        # TODO: load kaldiio data
        retval = list()
        for role, modality, target, content in self.data[name]:
            if modality in ["codec", "ssl", "codec_ssl", "image"]:
                content = kaldiio.load_mat(content)
            retval.append((role, modality, target, content))
        
        return tuple(retval)

    def __len__(self):
        return len(self.data)