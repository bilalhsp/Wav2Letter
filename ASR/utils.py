class Text_Processor():
    def __init__(self, vocab_path):

        with open(vocab_path, 'r') as f:
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