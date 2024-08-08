import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast
import re
import emoji


# model class
class MBTIPredictor(nn.Module):
    def __init__(self, hidden_layer_size, sequence_length):
        super(MBTIPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn = nn.RNN(768, hidden_layer_size, batch_first = True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_layer_size * sequence_length, 16)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        rnn_out, _ = self.rnn(outputs.last_hidden_state)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)  # Flatten the output
        relu_out = self.relu(rnn_out)
        logits = self.linear(relu_out)

        return logits


# function to clean text, make it ready to tokenize
def clean_text(tweets):
    tweets = emoji.replace_emoji(tweets, replace = '')
    tweets = re.sub(r'http\S+|www\S+|https\S+', '', tweets, flags = re.MULTILINE)
    tweets = tweets.lower()
    tweets = re.sub(r'@\w+', '', tweets)
    tweets = re.sub(r'\s+', ' ', tweets).strip()

    return tweets


# model/tokenizer creation parameters
hidden_size = 64
sequence_length = 512

# creating model/tokenizer
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = MBTIPredictor(hidden_size, sequence_length)
<<<<<<< HEAD
state_dict = torch.load('./model_state.pth', map_location=torch.device('cpu')) # if device is cpu
=======
state_dict = torch.load('../model/model_state.pth', map_location = torch.device('cpu'))  # if device is cpu
>>>>>>> 8f5b376 (webscraper and predict work together.)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

<<<<<<< HEAD

# needed to make predictions
label_mapping = {
 0: 'ENFJ',
 1: 'ENFP',
 2: 'ENTJ',
 3: 'ENTP',
 4: 'ESFJ',
 5: 'ESFP',
 6: 'ESTJ',
 7: 'ESTP',
 8: 'INFJ',
 9: 'INFP',
 10: 'INTJ',
 11: 'INTP',
 12: 'ISFJ',
 13: 'ISFP',
 14: 'ISTJ',
 15: 'ISTP'
=======
# needed to make predictions
label_mapping = {
    0: 'ENFJ',
    1: 'ENFP',
    2: 'ENTJ',
    3: 'ENTP',
    4: 'ESFJ',
    5: 'ESFP',
    6: 'ESTJ',
    7: 'ESTP',
    8: 'INFJ',
    9: 'INFP',
    10: 'INTJ',
    11: 'INTP',
    12: 'ISFJ',
    13: 'ISFP',
    14: 'ISTJ',
    15: 'ISTP'
>>>>>>> 8f5b376 (webscraper and predict work together.)
}


# function to make prediction from a given text
def predict(tweets):
<<<<<<< HEAD

    cleaned_text = clean_text(tweets)
    tokenized = tokenizer(text=cleaned_text, truncation=True, padding=True, max_length=sequence_length, return_tensors='pt')
=======
    cleaned_text = clean_text(tweets)
    tokenized = tokenizer(text = cleaned_text, truncation = True, padding = True,
                          max_length = sequence_length, return_tensors = 'pt')
>>>>>>> 8f5b376 (webscraper and predict work together.)

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        pred = model(input_ids, attention_mask)
<<<<<<< HEAD
        
    label_index = torch.argmax(pred, dim=-1).item()
=======

    label_index = torch.argmax(pred, dim = -1).item()
>>>>>>> 8f5b376 (webscraper and predict work together.)
    return label_mapping[label_index]