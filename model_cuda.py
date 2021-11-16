import torch
import torch.nn as nn
from scipy.special import softmax as softmax_matrix
from torchvision import models
from torchinfo import summary
import config as cfg

n_input= (1,cfg.n_input, 128)
n_classes= cfg.n_classes

class PEN(nn.Module):
    def __init__(self, input=n_input, num_classes=n_classes):
        super().__init__()
        self.input_channels=input[1]
        self.N_LSTM_layers=1
        self.N_time=input[2]
        self.num_batches=input[0]
        self.linear_layer_features=0
        self.num_classes=num_classes
        # Feature Network
        self.features = nn.Sequential(
            nn.Conv1d(self.input_channels, 128, kernel_size=11),
            nn.BatchNorm1d(128) , #considered channel size
            nn.LeakyReLU(),
            nn.Conv1d(128, 128,kernel_size=7),
            nn.BatchNorm1d(128) ,  #considered channel size
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=5), 
            nn.BatchNorm1d(128) ,
            nn.LeakyReLU()
            )
        self.avg_Pool=nn.AvgPool1d(2)   #kernel size not given in paper
        #LSTM
        self.LSTM_Q=nn.LSTM(self.input_channels, 64, self.N_LSTM_layers,batch_first=True) # hidden size=64 (units=64 given in paper)
        self.LSTM_K=nn.LSTM(self.input_channels, 64, self.N_LSTM_layers,batch_first=True) # hidden size=64 (units=64 given in paper)
        self.LSTM_V=nn.LSTM(self.input_channels, 64, self.N_LSTM_layers,batch_first=True) # hidden size=64 (units=64 given in paper)

        self.LSTM_Q1=nn.LSTM(64, 64, self.N_LSTM_layers,batch_first=True) # hidden size=64 (units=64 given in paper)
        self.LSTM_K1=nn.LSTM(64, 64, self.N_LSTM_layers,batch_first=True) # hidden size=64 (units=64 given in paper)
        self.LSTM_V1=nn.LSTM(64, 64, self.N_LSTM_layers,batch_first=True) # hidden size=64 (units=64 given in paper)
        
        self.softmax=nn.Softmax()
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	#nn.Dropout(),
        	nn.Linear(64*236, num_classes),
            nn.Softmax(dim=1)
        	)
    def init_hidden(self,batch_size):
        # h0=torch.zeros(self.N_LSTM_layers, self.num_batches, 64)  #N_LSTM_out=64
        # c0=torch.zeros(self.N_LSTM_layers, self.num_batches, 64)
        # return h0,c0
        if(torch.cuda.is_available()):
          return (torch.zeros(self.N_LSTM_layers, batch_size, 64).cuda(),  #N_LSTM_out=64
                torch.zeros(self.N_LSTM_layers, batch_size, 64).cuda())
        else:
          return (torch.zeros(self.N_LSTM_layers, batch_size, 64),  #N_LSTM_out=64
                torch.zeros(self.N_LSTM_layers, batch_size, 64))
    def forward(self, x, hidden):
        x1 = self.features(x) # feature network
        x1= self.avg_Pool(x1.permute(0,2,1)).permute(0,2,1)
        #LSTM attention layer 1
        # h0,c0=self.init_hidden()
        h0=hidden[0]
        c0=hidden[1]
        x=x.permute(0,2,1)
        query_matrix,(ht_1,ct_1)=self.LSTM_Q(x,(h0,c0))
        key_matrix,(ht_2,ct_2)=self.LSTM_K(x,(h0,c0))
        value_matrix,(ht_3,ct_3)=self.LSTM_V(x,(h0,c0))

        if(torch.cuda.is_available()):
          temp= torch.from_numpy(softmax_matrix(torch.matmul(query_matrix,key_matrix.transpose(2,1)).cpu().detach().numpy())).cuda()
        else:
          temp= torch.from_numpy(softmax_matrix(torch.matmul(query_matrix,key_matrix.transpose(2,1)).detach().numpy()))
        # rn1_out=torch.matmul(softmax_matrix(torch.matmul(query_matrix,key_matrix.transpose(2,1))), value_matrix)
        rn1_out=torch.matmul(temp, value_matrix)
        
        if(torch.cuda.is_available()):
          temp= torch.from_numpy(softmax_matrix(torch.matmul(ht_1,ht_2.transpose(2,1)).cpu().detach().numpy())).cuda()
        else:
          temp= torch.from_numpy(softmax_matrix(torch.matmul(ht_1,ht_2.transpose(2,1)).detach().numpy()))
        # h_out=torch.matmul(softmax_matrix(torch.matmul(ht_1,ht_2.transpose(2,1))), ht_3)
        h_out=torch.matmul(temp, ht_3)

        if(torch.cuda.is_available()):
          temp= torch.from_numpy(softmax_matrix(torch.matmul(ct_1,ct_2.transpose(2,1)).cpu().detach().numpy())).cuda()
        else:
          temp= torch.from_numpy(softmax_matrix(torch.matmul(ct_1,ct_2.transpose(2,1)).detach().numpy()))
        # c_out=torch.matmul(softmax_matrix(torch.matmul(ct_1,ct_2.transpose(2,1))), ct_3)
        c_out=torch.matmul(temp, ct_3)

        #LSTM attention layer 2
        query_matrix,(ht,ct)=self.LSTM_Q1(rn1_out,(h_out,c_out))
        key_matrix,(ht,ct)=self.LSTM_K1(rn1_out,(h_out,c_out))
        value_matrix,(ht,ct)=self.LSTM_V1(rn1_out,(h_out,c_out))

        if(torch.cuda.is_available()):
          temp= torch.from_numpy(softmax_matrix(torch.matmul(query_matrix,key_matrix.transpose(2,1)).cpu().detach().numpy())).cuda()
        else:
          temp= torch.from_numpy(softmax_matrix(torch.matmul(query_matrix,key_matrix.transpose(2,1)).detach().numpy()))

        # rn2_out=torch.matmul(softmax_matrix(torch.matmul(query_matrix,key_matrix.transpose(2,1))), value_matrix)
        rn2_out=torch.matmul(temp, value_matrix)

        rn2_out=rn2_out.permute(0,2,1)
        #concat out and x1
        c=torch.cat((rn2_out,x1),2)
        #self.linear_layer_features=c.size(1)*c.size(2)
        #print(c.size(1)) #64
        #print(c.size(2)) #236
        c = c.view(c.size(0), c.size(1)*c.size(2)) 
        
        out = self.classifier(c)
        # out=nn.Linear(self.linear_layer_features, self.num_classes)(c)
        # out=nn.Softmax(dim=1)(out)
        return out

def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


# if __name__ == '__main__':
#     #input-> batch, num_features, time
#     data='UCI_HAR'
#     if data=='UCI_HAR':
#         num_channels=9
#         segment_size=128
#         num_classes=6
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model = PEN((1,num_channels,segment_size), num_classes).to(device)
#         print(model)
#         # model = PEN((1,9,30), 6)
#         # summary(model,[(1,9,128),((1,1,64),(1,1,64))])


