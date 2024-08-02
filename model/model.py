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
        self.linear = nn.Linear(32768, 16)
        # confused hidden_layer_size * sequence_length (= 32000) doesn't work instead of 32768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        rnn_out, _ = self.rnn(outputs.last_hidden_state)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)  # Flatten the output
        relu_out = self.relu(rnn_out)
        logits = self.linear(relu_out)

        return logits
    


def clean_text(tweets):
    
    # remove emojis
    tweets = emoji.replace_emoji(tweets, replace='')
    
    # remove links 
    tweets = re.sub(r'http\S+|www\S+|https\S+', '', tweets, flags=re.MULTILINE)
    
    # make lowercase
    tweets = tweets.lower()

    # remove twitter handles
    tweets = re.sub(r'@\w+', '', tweets)

    # remove extra whitespace
    tweets = re.sub(r'\s+', ' ', tweets).strip()
    
    return tweets



model = MBTIPredictor()
state_dict = torch.load('../model_state.pth')
model.load_state_dict(state_dict)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

reverse_mapping = {
 'enfj': 0,
 'enfp': 1,
 'entj': 2,
 'entp': 3,
 'esfj': 4,
 'esfp': 5,
 'estj': 6,
 'estp': 7,
 'infj': 8,
 'infp': 9,
 'intj': 10,
 'intp': 11,
 'isfj': 12,
 'isfp': 13,
 'istj': 14,
 'istp': 15
 }
label_mapping = {v: k.upper() for k, v in reverse_mapping.items()}


def predict(tweets):
    clean_text = clean_text(tweets)
    tokenized = tokenizer(text=clean_text, truncation=True, padding=True)
    prediction_as_number = model(tokenized)
    return label_mapping[prediction_as_number]
    

