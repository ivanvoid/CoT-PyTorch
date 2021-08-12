import torch
import torch.nn as nn

class Mediator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, 
                 vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Mediator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = torch.zeros((1, batch_size, self.hidden_dim), requires_grad=True)
        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):           
        # batch_size x seq_len x embedding_dim
        emb = self.embeddings(input)
        # seq_len x batch_size x embedding_dim
        emb = emb.permute(1, 0, 2)
        # 4 x batch_size x hidden_dim
        _, hidden = self.gru(emb, hidden)
        # batch_size x 1 x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()
        # batch_size x hidden_dim
        out = hidden.view(-1, self.hidden_dim)
        out = self.dropout_linear(out)
        # batch_size x 1
        out = self.hidden2out(out)
        out = torch.nn.functional.log_softmax(out)
        
        return out, hidden

    def get_reward(self, x):
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        out, hidden = self.forward(x, hidden)
        return out
        # return log_predictions 
    
        # pad with zeros
        # run forward 
        # return log_predictions