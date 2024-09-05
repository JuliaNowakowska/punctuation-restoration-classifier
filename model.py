import torch.nn as nn
from transformers import AutoModelForMaskedLM

class PunctuationRestorationModel(nn.Module):
    def __init__(self, num_labels):
        super(PunctuationRestorationModel, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits
