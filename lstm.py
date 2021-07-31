import torch
import torch.nn as nn
import tensorflow as tf

class cslstm(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape,device, input_channel, forget_bias=1.0, tln=False, initializer=0.001):
        super(cslstm,self).__init__()
        self.device = device
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.batch = seq_shape[0]
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self.layer_norm = tln
        self._forget_bias = forget_bias
        self.h_cc_conv = nn.Conv2d(num_hidden_out,num_hidden_out*4,filter_size,1,2) #?
        self.c_cc_conv = nn.Conv2d(num_hidden_out,num_hidden_out*3,filter_size,1,2) #?
        self.m_cc_conv = nn.Conv2d(num_hidden_in,num_hidden_out*3,filter_size,1,2) #?
        self.c2m_conv = nn.Conv2d(num_hidden_out,num_hidden_out*4,filter_size,1,2)
        if layer_name == "lstm_1":
            self.x_cc_conv = nn.Conv2d(input_channel,num_hidden_out*7,filter_size,1,2)
        else:
            self.x_cc_conv = nn.Conv2d(num_hidden_in,num_hidden_out*7,filter_size,1,2)
            
        self.o_m_conv = nn.Conv2d(num_hidden_out,num_hidden_out,filter_size,1,2) 
        self.cell_conv = nn.Conv2d(num_hidden_out*2,num_hidden_out,1,1,0) 
        
        self.h_norm = nn.BatchNorm2d(num_hidden_out*4)
        self.c_norm = nn.BatchNorm2d(num_hidden_out*3)
        self.m_norm = nn.BatchNorm2d(num_hidden_out*3)
        self.x_norm = nn.BatchNorm2d(num_hidden_out*7)
        self.c2m_norm = nn.BatchNorm2d(num_hidden_out*4)
        self.o_m_norm = nn.BatchNorm2d(num_hidden_out)
        
    def init_state(self):
        return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                        dtype=torch.float32)

    def __call__(self, x, h, c, m):
        if h is None:
            h = torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                        dtype=torch.float32).to(self.device)
        if c is None:
            c = torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                        dtype=torch.float32).to(self.device)
        if m is None:
            m = torch.zeros((self.batch, self.num_hidden_in, self.height, self.width),
                        dtype=torch.float32).to(self.device)
            
            
        
        h_cc = self.h_cc_conv(h)
        
        c_cc = self.c_cc_conv(c)

        m_cc = self.m_cc_conv(m)
        
        if self.layer_norm:
            h_cc = self.h_norm(h_cc)
            c_cc = self.c_norm(c_cc)
            m_cc = self.m_norm(m_cc)

        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.num_hidden, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.num_hidden, 1)

        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
            
        else:
#             print(x.shape)
            x_cc = self.x_cc_conv(x)
            
            if self.layer_norm:
                x_cc = self.x_norm(x_cc)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc, self.num_hidden, 1)

            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)

        c_new = f * c + i * g

        c2m = self.c2m_conv(c_new)
        
        if self.layer_norm:
            c2m = self.c2m_norm(c2m)

        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)

        m_new = ff * torch.tanh(m_m) + ii * gg
        
        o_m = self.o_m_conv(m_new)
        
        if self.layer_norm:
            o_m = self.o_m_norm(o_m)

        if x is None:
            o = torch.tanh(o_h + o_c + o_m)
        else:
            o = torch.tanh(o_x + o_h + o_c + o_m)

        cell = torch.cat([c_new, m_new],1)
        cell = self.cell_conv(cell)

        h_new = o * torch.tanh(cell)
         
        
        return h_new, c_new, m_new