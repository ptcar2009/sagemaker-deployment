import torch.nn as nn
import torch
import torchtext

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, kernel_size = 3, conv_channels = 2, batch_size = 20, padding = 1):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, conv_channels, kernel_size, padding=padding) 
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=conv_channels*embedding_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        output = None
        embeds = self.embedding(reviews)
        #for embed in embeds.permute(1,2,0):
         #   conved = self.conv(embed.unsqueeze(0))
         #   
         #   if output == None:
         ##       output = [conved]
          #  else:
           #     output.append(conved)
        #output = torch.cat(output, dim=0)
        #output = nn.ReLU()(output.permute(2,0,1))
        #print(output.size())
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(output.squeeze())
        out = out[lengths - 1, range(len(lengths))]
        #print(out.squeeze().size())
        
        return self.sig(out.squeeze())