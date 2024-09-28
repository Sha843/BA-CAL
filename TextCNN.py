import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):
    def __init__(self,num_classes,embeddings,embed=300,num_filters=256,dropout=0.5,filter_sizes = (2, 3, 4),
                 vocab_size=None,emb_size=300):
        super(TextCNN,self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k,embed)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.embeddings = self._load_embeddings(embeddings, emb_size, vocab_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embeddings(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
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
