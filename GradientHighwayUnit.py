
import torch
import torch.nn as nn


class gradient_highway(nn.Module):    
    def __init__(self, layer_name, filter_size, num_features, batch, height, width, device, tln=False):
        super(gradient_highway,self).__init__()
        """Initialize the Gradient Highway Unit.
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_features = num_features
        self.batch = batch
        self.height = height
        self.width = width
        self.device = device
        self.layer_norm = tln
        self.x_conv = nn.Conv2d(self.num_features,self.num_features*2,filter_size,1,2) ##??
        self.z_conv = nn.Conv2d(self.num_features,self.num_features*2,filter_size,1,2)
        
        self.z_norm = nn.BatchNorm2d(self.num_features*2)
        self.x_norm = nn.BatchNorm2d(self.num_features*2)


    def init_state(self):
        
        return torch.zeros((self.batch,self.num_features,self.width,self.height), dtype=torch.float32).to(self.device)

    def __call__(self, x, z):
        if z is None:
            z = self.init_state()
            
        z_concat = self.z_conv(z)
        
        if self.layer_norm:
            z_concat = self.z_norm(z_concat)

        x_concat = self.x_conv(x)
        if self.layer_norm:
            x_concat = self.x_norm(x_concat)
        
        gates = torch.add(x_concat, z_concat)
        p, u = torch.split(gates, self.num_features, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new
