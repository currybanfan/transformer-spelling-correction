import string
from torch import Tensor

class Tokenizer:
    def __init__(self):
        characters = list(string.ascii_lowercase) + ['[unk]', '[sos]', '[eos]', '[pad]', '[mask]']

        self.char_2_index = {char: idx for idx, char in enumerate(characters)}
        self.index_2_char = {idx: char for idx, char in enumerate(characters)}
        
        self.mask_idx = self.char_2_index['[mask]']
        self.pad_idx = self.char_2_index['[pad]']
        self.eos_idx = self.char_2_index['[eos]']
        self.sos_idx = self.char_2_index['[sos]']
        self.unk_idx = self.char_2_index['[unk]']


    def char_to_index(self, word: str) -> list:
        """Convert a string to a list of indices based on char_2_index mapping."""
        return [self.char_2_index.get(char, self.unk_idx) for char in word]

    def index_to_char(self, indices: list, without_token=True) -> str:
        """Convert a list of indices back to a string, removing special tokens if needed."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        
        result = ''.join(self.index_2_char.get(i, '[unk]') for i in indices)
        
        if without_token:
            result = result.split('[eos]')[0]
            result = result.replace('[sos]', '').replace('[eos]', '').replace('[pad]', '')
        
        return result

