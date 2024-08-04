import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast
import re
import emoji

class MBTIPredictor(nn.Module):
    def __init__(self, hidden_layer_size, sequence_length):
        super(MBTIPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn = nn.RNN(768, hidden_layer_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_layer_size * sequence_length, 16)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        rnn_out, _ = self.rnn(outputs.last_hidden_state)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)  # Flatten the output
        relu_out = self.relu(rnn_out)
        logits = self.linear(relu_out)

        return logits
    


def clean_text(tweets):
    tweets = emoji.replace_emoji(tweets, replace='')
    tweets = re.sub(r'http\S+|www\S+|https\S+', '', tweets, flags=re.MULTILINE)
    tweets = tweets.lower()
    tweets = re.sub(r'@\w+', '', tweets)
    tweets = re.sub(r'\s+', ' ', tweets).strip()
    
    return tweets

# model/tokenizer creation parameters
hidden_size = 64
sequence_length = 512

# creating model/tokenizer
model = MBTIPredictor(hidden_size, sequence_length)
state_dict = torch.load('../model_state.pth')
model.load_state_dict(state_dict)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# needed to make predictions
label_mapping = {
 '0': 'ENFJ',
 '1': 'ENFP',
 '2': 'ENTJ',
 '3': 'ENTP',
 '4': 'ESFJ',
 '5': 'ESFP',
 '6': 'ESTJ',
 '7': 'ESTP',
 '8': 'INFJ',
 '9': 'INFP',
 '10': 'INTJ',
 '11': 'INTP',
 '12': 'ISFJ',
 '13': 'ISFP',
 '14': 'ISTJ',
 '15': 'ISTP'
}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def predict(tweets):
    assertTrue(isinstance(tweets, str), "Input to function must be one string")

    cleaned_text = clean_text(tweets)
    tokenized = tokenizer(text=cleaned_text, truncation=True, padding=True, max_length=sequence_length)

    tokenized['input_ids'] = torch.tensor(tokenized['input_ids'])
    tokenized['attention_mask'] = torch.tensor(tokenized['attention_mask'])

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    pred = model(input_ids, attention_mask)
    return label_mapping[torch.argmax(pred, dim=-1)]