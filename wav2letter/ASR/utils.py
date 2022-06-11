import os

class Text_Processor():
    def __init__(self, config):
        self.vocab_path = os.path.join(config["config_path"],config["vocab_file"])
        with open(self.vocab_path, 'r') as f:
            chars = f.read().split('\n')
        self.char_to_idx = {c:i for i, c in enumerate(chars)}
        self.idx_to_char = {i:c for i, c in enumerate(chars)}
    
    def text_to_indices(self, text):
        indices = []
        for c in text.lower():
            if c == ' ':
                c = '<space>'
            # if c=='.':
            #     c = '<eos>'
            # if c not in self.char_to_idx.keys():
            #     c = '<unk>'
            indices.append(self.char_to_idx[c])

        return indices
    
    def indices_to_text(self, indices):
        text = []
        for i in indices:
            c = self.idx_to_char[i]
            if c == '<space>':
                c = ' '
            text.append(c)
        return text
    
    def label_to_text(self, label):
        batch_size = label.shape[0]
        strings = []
        for n in range(batch_size):
            char = ''
            
            for i in label[n]:
                c = self.idx_to_char[i.item()]
                if c == '<space>':
                    c = ' '
                char = char+c
            strings.append(char)
        return strings
        
    def c_to_text(self, chars):
        text = ''
        for c in chars:
            text = text+c
        return text