import torch.nn as nn
import torch
from torch.nn.utils.rnn import unpack_sequence
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    

    def save(self,model_path):
        torch.save(self.cpu().state_dict(), model_path)

    def load(self,model_path):
        self.load_state_dict(torch.load(model_path))


class BaseLSTMNetwork(NeuralNetwork):
    def __init__(self,lstm_input_size,lstm_hidden_size,layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size= lstm_input_size,hidden_size= lstm_hidden_size,batch_first= True,num_layers = layers)

    def _unpackAndGetFeatureFromLSTMOutput(self,lstm_out : nn.utils.rnn.PackedSequence):
        lstm_out = unpack_sequence(lstm_out)
        lstm_out = list(map(lambda x : x[-1],lstm_out))
        lstm_out = torch.stack(lstm_out,dim= 0)
        return lstm_out

class LSTMNetwork(BaseLSTMNetwork):
    def __init__(self,lstm_input_size,lstm_hidden_size,output_dim,layers = 1) -> None:
        super().__init__(lstm_hidden_size=lstm_hidden_size,lstm_input_size=lstm_input_size,layers= layers)
        self.output_dim = output_dim
        self.linear = nn.Linear(lstm_hidden_size,output_dim)
    
    def forward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        OR
        packed_sequence

        The output is of shape (BS,num_classes)
        """
        lstm_out, _ = self.lstm(X)
        if isinstance(X,torch.Tensor):
            lstm_out = lstm_out[:,-1,:]
        else:
            lstm_out = self._unpackAndGetFeatureFromLSTMOutput(lstm_out= lstm_out)

        lstm_out = lstm_out#/torch.linalg.norm(lstm_out,dim = -1,keepdims = True)
        out = self.linear(lstm_out)
        return out,lstm_out
    

    def earlyClassificationForward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        outputs (BS,seq_len,feature_len)
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(X)
            return self.linear(lstm_out),lstm_out


     
class LSTMDuelingNetwork(BaseLSTMNetwork):
    def __init__(self,lstm_input_size,lstm_hidden_size,output_dim,layers = 1) -> None:
        super().__init__(lstm_hidden_size=lstm_hidden_size,lstm_input_size=lstm_input_size,layers= layers)
        self.output_dim = output_dim
        self.value_linear = nn.Sequential(nn.Linear(lstm_hidden_size,lstm_hidden_size//2), nn.ReLU(),
                                           nn.Linear(lstm_hidden_size//2, lstm_hidden_size//2), nn.ReLU(),
                                           nn.Linear(lstm_hidden_size//2, 1)
                                           )
        self.advantage_linear = nn.Sequential(nn.Linear(lstm_hidden_size,lstm_hidden_size//2), nn.ReLU(),
                                    nn.Linear(lstm_hidden_size//2, lstm_hidden_size//2), nn.ReLU(),
                                    nn.Linear(lstm_hidden_size//2, output_dim)
                                    )

    def forward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        OR
        packed_sequence

        The output is of shape (BS,num_classes)
        """
        lstm_out, _ = self.lstm(X)
        if isinstance(X,torch.Tensor):
            lstm_out = lstm_out[:,-1,:]
        else:
            lstm_out = self._unpackAndGetFeatureFromLSTMOutput(lstm_out= lstm_out)


        #lstm_out = lstm_out#/torch.linalg.norm(lstm_out,dim = -1,keepdims = True)
        values = self.value_linear(lstm_out)
        advantage = self.advantage_linear(lstm_out)
        out = advantage - torch.mean(advantage,dim= -1, keepdim= True) + values
        return out,lstm_out
    

    def earlyClassificationForward(self,X):
        """
        X is the timeseries input of shape 
        (BS,Seq len, lstm_input_size)
        outputs (BS,seq_len,feature_len)
        """
        with torch.no_grad():
            if isinstance(X,torch.Tensor):
                lstm_out, _ = self.lstm(X)
                advantage = self.advantage_linear(lstm_out)
                values = self.value_linear(lstm_out)
                out = advantage - torch.mean(advantage,dim= -1, keepdim= True) + values
                return out,lstm_out
            else:
                # in the case where X is a packed sequencce
                lstm_out, _ = self.lstm(X)
                lstm_out = unpack_sequence(lstm_out) # this is a list of tensors of shape (seq_len,hidden_dim)

                outs = []
                for seq in lstm_out:
                    advantage = self.advantage_linear(seq) # (seq_len,num_classes + 1)
                    values = self.value_linear(seq) # (seq_len,1)
                    outs.append( advantage - torch.mean(advantage,dim= -1, keepdim= True) + values)  # appended (seq_len,num_classes + 1)
                return outs, lstm_out





class TransformerGenerator(NeuralNetwork):
    def __init__(self,output_dim,random_dim,seq_len,embedding_dim,num_layers,num_heads,out_layer,cond_dim = None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        
        if self.cond_dim != None:
            self.first_linear=  nn.Sequential(nn.Linear(random_dim + self.cond_dim,random_dim),nn.LeakyReLU(),nn.Linear(random_dim,random_dim))
        else:
            self.first_linear = nn.Identity()
        self.random_to_embedding_dim_linear = nn.Linear(random_dim,embedding_dim*seq_len)
        self.positional_encodings = nn.Embedding(self.seq_len,embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding_dim_to_output_linear = nn.Sequential(nn.Linear(embedding_dim,output_dim),out_layer)

    

    def forward(self,z,cond = None):
        """
        Z is (BS,random_dim)

        we want it to be (BS,seq_len,embedding_dim)
        """

        batch_size = z.size(0)

        
        if self.cond_dim != None:
            z = torch.concatenate([z,cond], dim= -1)
        first_layer_out = self.first_linear(z)


        u = self.random_to_embedding_dim_linear(first_layer_out)
        u = u.view(batch_size,self.seq_len,self.embedding_dim)
        positions = torch.arange(0, self.seq_len).unsqueeze(0).expand([batch_size,-1]).to(u.device)
        x = u + self.positional_encodings(positions)

        x = self.transformer_encoder(x)
        x = self.embedding_dim_to_output_linear(x)
        return x,first_layer_out
    

    @staticmethod
    def generateRandomZ(batch_size,random_dim):
        return torch.randn(batch_size,random_dim)
    

    @staticmethod
    def makeImageFromGeneratorOutput(gen_out):
        """
        gen_out is (BS,seq_len,patch_size)
        """
        batch_size,seq_len = gen_out.size(0),gen_out.size(1)
        square_size = np.sqrt(gen_out.shape[-1])
        assert square_size%1 == 0
        square_size = int(square_size)
        gen_out = gen_out.view(batch_size,seq_len,square_size,square_size)
        num_squares = np.sqrt(seq_len)
        assert num_squares%1 == 0
        num_squares = int(num_squares)
        image_dim = num_squares*square_size
        img = torch.empty(batch_size,image_dim,image_dim).to(gen_out.device)


        index = 0
        for i in range(0,image_dim,square_size):
            for j in range(0,image_dim,square_size):
                img[:,i:i+square_size,j:j+square_size] = gen_out[:,index]
                index += 1
        
        img = img.unsqueeze(1)
        return img
        


class CNNGenerator(nn.Module):

    @staticmethod
    def generateRandomZ(batch_size,seq_len,random_dim):
        return torch.randn(batch_size,random_dim)



    def __init__(self,random_dim = 16):
        super().__init__()
        self.t_conv = nn.Sequential(nn.ConvTranspose2d(in_channels= 128, out_channels= 64,kernel_size= 3,stride= 1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels= 64,out_channels= 32,kernel_size= 3, stride= 1,bias= False),nn.BatchNorm2d(32),nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels= 32, out_channels= 16, kernel_size= 3, stride= 1, bias= False),nn.BatchNorm2d(16),nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels= 16, out_channels= 8, kernel_size= 3, stride= 2, bias= False),nn.BatchNorm2d(8),nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels= 8, out_channels= 4, kernel_size= 3, stride= 1, bias= False),nn.BatchNorm2d(4),nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels= 4, out_channels= 2, kernel_size= 3, stride= 1, bias= False),nn.BatchNorm2d(2),nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels= 2, out_channels= 1, kernel_size= 4, stride= 1, bias= False), nn.Sigmoid()
        )


        self.linear = nn.Sequential(nn.Linear(random_dim,64),nn.LeakyReLU(),nn.Linear(64,512),nn.LeakyReLU(), nn.Linear(512,4*4*128))
    
    def forward(self,z):
        x = self.linear(z)
        batch_size = x.size(0)
        x = x.view(batch_size,128,4,4)
        return self.t_conv(x)

class CNNNetwork(NeuralNetwork):
    def __init__(self, ts_dim, num_filters, kernel_sizes,output_dim):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=ts_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                #nn.MaxPool1d(kernel_size=2)
            )
            self.conv_blocks.append(conv_block)
        
        
        self.fc = nn.Linear(4*num_filters * len(kernel_sizes), output_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, ts_dim) 
        # Apply convolutional blocks
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_output = conv_block(x.permute(0, 2, 1).contiguous())  # Conv1D expects (batch_size, in_channels, seq_length)
            conv_output = F.max_pool1d(conv_output, kernel_size=conv_output.size(2)).squeeze(2)  # Global Max Pooling
            conv_outputs.append(conv_output)
        
        # Concatenate convolutional outputs

        conv_output_concat = torch.cat(conv_outputs, dim=1)
        return self.fc(conv_output_concat)
    





class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride= stride,bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )


        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,stride= stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # (BS,out_channels,H,W)
    def forward(self, x):
        out = self.conv(x)
        out = out + self.downsample(x)
        return out


class CNNNetwork2D(nn.Module):
    def __init__(self, in_channels, num_filters,output_dim,num_layers = 3):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels= in_channels, out_channels= num_filters,kernel_size= 3,padding= 1,bias= False),
                                  nn.BatchNorm2d(num_filters),
                                  nn.LeakyReLU()
                                  )
        layers = []
        final_out_channels = num_layers
        for i in range(1,num_layers+1):
            in_channels = num_filters*(2**(i-1))
            out_channels = num_filters*(2**i)
            layers.append(ResNetBlock(in_channels= in_channels,out_channels= out_channels,stride= 2))
            final_out_channels = out_channels
        self.layers = nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(final_out_channels,output_dim)



    def forward(self, x):
        """
        x is  (BS,num_channels,H,W)
        """
        # Apply convolutional blocks
        x = self.conv(x)
        x = self.layers(x)[:,:,0,0]
        return self.fc(x)



class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0),stride= stride,bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.LeakyReLU(),
        )


        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,stride= stride,bias=False),
            )
        # (BS,out_channels,seq_len)
    def forward(self, x):
        out = self.conv(x)
        out = out + self.downsample(x)
        return out
    
class CNNNetwork1D(nn.Module):
    def __init__(self, in_channels, num_filters,output_dims,num_layers = 3,cond_dim = None):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels= in_channels, out_channels= num_filters,kernel_size= (3,1),padding= (1,0),bias= False,stride= 1),
                                  nn.LeakyReLU()
                                  )
        layers = []
        final_out_channels = num_layers
        for i in range(1,num_layers+1):
            in_channels = num_filters*(2**(i-1))
            out_channels = num_filters*(2**i)
            layers.append(ResNetBlock1D(in_channels= in_channels,out_channels= out_channels,stride= 2 if i >= 2 else 1))
            final_out_channels = out_channels
        self.layers = nn.Sequential(*layers,nn.AdaptiveAvgPool2d(output_size= 1))

        self.linears = nn.ModuleList()

        for i in range(len(output_dims)):
            self.linears.append(
                nn.Sequential(nn.Linear(final_out_channels + (cond_dim if cond_dim != None else 0),final_out_channels//2),nn.LeakyReLU(),
                              nn.Linear(final_out_channels//2,final_out_channels//2),nn.LeakyReLU(),
                            nn.Linear(final_out_channels//2,output_dims[i]))
            )


    def forward(self, x,condition = None):
        # x: (batch_size, seq_length, ts_dim) 
        # Apply convolutional blocks
        # if condition exists its (BS,cond_dim)
        x = x.permute(0, 2, 1).contiguous()  # Conv1D expects (batch_size, in_channels, seq_length)
        x = x.unsqueeze(-1) # (BS,in_channels,seq_len,1) 
        x = self.conv(x)
        x = self.layers(x)[:,:,0,0]
        outputs = []

        if condition != None:
            x = torch.concatenate([x,condition],dim= -1)

        for linear in self.linears:
            outputs.append(linear(x))

        return outputs


    

class LinearPredictor(nn.Module):
    def __init__(self,feature_dim,num_classes) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(feature_dim,feature_dim*2), nn.LeakyReLU(), nn.Linear(feature_dim*2,feature_dim*2), nn.LeakyReLU(),
                                    nn.Linear(feature_dim*2,feature_dim*2), nn.LeakyReLU(),nn.Linear(feature_dim*2,feature_dim), nn.LeakyReLU(),
                                    nn.Linear(feature_dim,num_classes)
                                    )
    
    def forward(self,X):
        return self.linear(X)
    

class Predictor(NeuralNetwork):
    def __init__(self,feature_dim,num_classes) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim,num_classes)
    def forward(self,X):
        return self.linear(X)
    




