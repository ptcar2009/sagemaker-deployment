import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_filters=3, filter_sizes=[3,4,5], dropout=0.5):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.ModuleList([
            nn.Conv1d(
                embedding_dim, 
                n_filters, 
                fs, 
            ) for fs in filter_sizes]
        )
        
        self.dense = nn.Linear(in_features=len(filter_sizes)*n_filters, out_features=1)
        self.drop = nn.Dropout(dropout)
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        embeds = embeds.permute(1,2,0)
        
        conveds = [F.relu(conv(embeds)) for conv in self.conv]  
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conveds]
        
        cat = self.drop(torch.cat(pooled,dim=1))
        #print(out.squeeze().size())
        return self.dense(cat)