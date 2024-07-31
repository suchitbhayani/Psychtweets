import torch
from torch import nn
from transformers import BertModel

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