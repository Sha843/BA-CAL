import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRNN(nn.Module):
    def __init__(self,num_classes,embeddings,embed=300,hidden_size = 128,num_layers = 2,dropout = 0.5,hidden_size2 = 64,vocab_size=None,emb_size=300):
        super(TextRNN, self).__init__()
        
        self.lstm = nn.LSTM(embed, hidden_size,num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2,hidden_size2)
        self.fc = nn.Linear(hidden_size2,num_classes)
        self.embeddings = self._load_embeddings(embeddings, emb_size, vocab_size)

    def forward(self, x):
        if isinstance(x, tuple):
            x, _ = x
        emb = self.embeddings(x)
        H, _ = self.lstm(emb)
        M = self.tanh1(H)
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        return out
    def _load_embeddings(self, embeddings, emb_size, vocab_size):
        if embeddings is not None:
            if vocab_size is not None:
                assert vocab_size == embeddings.shape[0]
            if emb_size is not None:
                assert emb_size == embeddings.shape[1]
            # vocab_size, emb_size = embeddings.shape
        word_embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        word_embeddings.weight = nn.Parameter(torch.from_numpy(embeddings).float())
        return word_embeddings