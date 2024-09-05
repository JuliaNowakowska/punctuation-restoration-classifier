import pandas as pd
from torch.utils.data import Dataset
import torch

class PunctuationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load sentences and labels
        df = pd.read_excel(dataset)
        self.sentences = df['input'].apply(lambda x: x.split())
        self.labels = df['labels'].apply(lambda x: x.split())

        # Label to index mapping
        self.label2idx = {'O': 0, 'D': 1, 'A': 2, 'Q': 3, 'X': 4, 'N': 5, 'H': 6}
        self.idx2label = {0: ' ', 1: '.', 2: ',', 3: '?', 4: '!', 5: ':', 6: '-'}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_tensors='pt',
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length)

        word_ids = encoding.word_ids()
        label_ids = [self.label2idx['O']] * self.max_length

        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(labels):
                label_ids[i] = self.label2idx[labels[word_id]]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids)
        }
