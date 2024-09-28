import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, emb_init,vocab_size=500000, emb_size=300, hidden_size=256, label_size=7,  emb_trainable=True, dropout=0.5):
        super(BiLSTM, self).__init__()

        if emb_init is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_init, dtype=torch.float32), freeze=not emb_trainable)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, label_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id):
        emb_out = self.dropout(self.embedding(input_id))
        lstm_out, _ = self.lstm(emb_out)
        final_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)
        logit = self.fc(final_output)
        return logit