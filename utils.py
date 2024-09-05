import torch

def predict(sentence, model, tokenizer, idx2label, device='cuda'):
    encoding = tokenizer(sentence.split(), is_split_into_words=True, return_tensors='pt',
                         padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = logits[0].cpu().numpy()
    predictions = logits.argmax(axis=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_labels = [idx2label[pred] for pred in predictions]

    result = []
    for token, label in zip(tokens, predicted_labels):
        if token != '[PAD]':
            result.append((token, label))

    return result
