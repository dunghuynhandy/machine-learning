import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
tokenizers = {
    "de": 'de_core_news_sm',
    "en": 'en_core_web_sm'
}

class TranslationDataLoader:
    def __init__(self, 
                 src_language='en', 
                 trg_language='de', 
                 batch_size=32, 
                 min_freq=2, 
                 special_symbols=['<unk>', '<pad>', '<bos>', '<eos>']):
        self.src_language = src_language
        self.trg_language = trg_language
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.special_symbols = special_symbols
        
        self.token_transform = {}
        self.vocab_transform = {}
        
        # Initialize tokenizers
        self.token_transform[self.src_language] = get_tokenizer('spacy', language=tokenizers[self.src_language])
        self.token_transform[self.trg_language] = get_tokenizer('spacy', language=tokenizers[self.trg_language])
        
        # Build vocabularies
        self.build_vocab()

        # Special tokens indices
        self.BOS_IDX = self.vocab_transform[self.src_language]['<bos>']
        self.EOS_IDX = self.vocab_transform[self.src_language]['<eos>']
        self.PAD_IDX = self.vocab_transform[self.src_language]['<pad>']

    def yield_tokens(self, data_iter, language):
        language_index = {self.src_language: 0, self.trg_language: 1}

        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])

    def build_vocab(self):
        for ln in [self.src_language, self.trg_language]:
            train_iter = Multi30k(split='train')
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=self.min_freq,
                                                                 specials=self.special_symbols,
                                                                 special_first=True)
        
        for ln in [self.src_language, self.trg_language]:
            self.vocab_transform[ln].set_default_index(self.vocab_transform[ln]['<unk>'])

    def tensor_transform(self, token_ids):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS_IDX])))

    def collate_fn(self, batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_tokens = self.token_transform[self.src_language](src_sample)
            trg_tokens = self.token_transform[self.trg_language](trg_sample)
            src_tensor = self.tensor_transform(self.vocab_transform[self.src_language](src_tokens))
            trg_tensor = self.tensor_transform(self.vocab_transform[self.trg_language](trg_tokens))
            
            src_batch.append(src_tensor)
            trg_batch.append(trg_tensor)

        # Pad sequences and transpose to get (batch_size, seq_len)
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX).transpose(0, 1)
        trg_batch = pad_sequence(trg_batch, padding_value=self.PAD_IDX).transpose(0, 1)
        
        return src_batch, trg_batch

    def get_collate_fn(self):
        return self.collate_fn

    def get_special_token_indices(self):
        return {
            '<bos>': self.BOS_IDX,
            '<eos>': self.EOS_IDX,
            '<pad>': self.PAD_IDX,
        }

    def get_vocab_size(self):
        src_vocab_size = len(self.vocab_transform[self.src_language])
        trg_vocab_size = len(self.vocab_transform[self.trg_language])
        return src_vocab_size, trg_vocab_size