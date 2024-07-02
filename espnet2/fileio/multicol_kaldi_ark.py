import kaldiio

class MultiColKaldiArkReader:
    def __init__(self, path):
        self.utt2example = {}
        for line in open(path):
            elems = line.strip().split()
            name = elems[0]
            contents = elems[1:]
            self.utt2example[name] = contents
    
    def __getitem__(self, name):
        contents = self.utt2example[name]
        retval = []

        for content in contents:
            retval.append(kaldiio.load_mat(content))
        
        return retval
    
    def __len__(self):
        return len(self.utt2example)