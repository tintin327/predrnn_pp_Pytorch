from lstm import cslstm
from GradientHighwayUnit import gradient_highway as ghu
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_shape, num_layers, num_hidden, seq_length,device , tln=True, loss_func=nn.MSELoss()):
        super(RNN,self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.seq_length = seq_length
        filter_size = 5 #??
        self.input_length = 10
        self.gradient_highway = ghu('highway', filter_size, num_hidden[0], input_shape[0], input_shape[3], input_shape[4], device, tln)
        #self.ghu_list = nn.ModuleList(ghu_list)
        self.conv = nn.Conv2d(self.num_hidden[-1], input_shape[2], 1, 1, 0) ###
        self.loss_func = loss_func
        self.lstm = []
        
        for i in range(num_layers):
            if i == 0:
                num_hidden_in = num_hidden[-1]
            else:
                num_hidden_in = num_hidden[i-1]
            
            input_channel = 16
            new_cell = cslstm('lstm_'+str(i+1),
                  filter_size,
                  num_hidden_in,
                  num_hidden[i],
                  input_shape,
                  device,
                  input_channel,
                  tln=tln)
            self.lstm.append(new_cell)
     
            
        self.lstm = nn.ModuleList(self.lstm)

    def forward(self, images, mask_true, test = False):
        # [batch, length, channel, width, height]
        gen_images = []
        cell = []
        hidden = []

        mem = None
        z_t = None
        for i in range(self.num_layers):
            cell.append(None)
            hidden.append(None)
            
        for t in range(self.seq_length-1):
            if t < self.input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-10]*images[:,t] + (1-mask_true[:,t-10])*x_gen
                #inputs = images[:,t]
            
            
            hidden[0], cell[0], mem = self.lstm[0](inputs, hidden[0], cell[0], mem) #?
  
            z_t = self.gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = self.lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, self.num_layers):
                hidden[i], cell[i], mem = self.lstm[i](hidden[i-1], hidden[i], cell[i], mem)
                
            x_gen = self.conv(hidden[self.num_layers-1])
            gen_images.append(x_gen)
        
       
        gen_images = torch.stack(gen_images,dim=1)
#         print(gen_images.shape)
#         print(images[:,1:].shape)
        #gen_images = gen_images.premute(1,0,2,3,4)
        loss = self.loss_func(gen_images,images[:,1:])

        return [gen_images*255, loss]

if __name__ == '__main__':
    a = torch.randn(3, 20, 1, 64, 64)
    shape = [3, 20, 1, 64, 64]
    numlayers = 4
    predrnn = RNN(shape, numlayers, [64,64,128,128], 20, True) #???
    predict, loss = predrnn(a)
    print(predict.shape)
    print(loss)
    
    
    