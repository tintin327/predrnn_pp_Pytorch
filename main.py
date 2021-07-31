from model import RNN
import torch
import numpy as np
from train import train , eval_model
import time
import data
from sklearn.model_selection import train_test_split
from numpy import newaxis
from tensorboardX import SummaryWriter

start_time = time.time()

cuda = "cuda:1"
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
#device = "cpu"
moving_mnist = ((np.load("./mnist_test_seq.npy")).swapaxes(0,1))[:,:,newaxis,:,:]
lr = 0.001
delta = 0.00002
base = 0.99998
eta = 1
EPOCHS = 25
itr = 0
loss_function =  torch.nn.BCELoss().to(device) #nn.BCELoss()
batch_size = 8
patch_size = 4
img_channel = 1
height = 64
width = 64
shape = [batch_size, 20, patch_size*patch_size*img_channel, int(height/patch_size), int(width/patch_size)]
num_hidden = [128,64,64,64]
numlayers = len(num_hidden)


model = (RNN(shape, numlayers, num_hidden, 20, device, True, loss_function,)).to(device) #???
mnist_train, mnist_val = train_test_split(
    moving_mnist,
    test_size=0.1,
    random_state=6
)

# mnist_test, mnist_val = train_test_split(
#     mnist_val,
#     test_size=0.9,
#     random_state=6
# )

# mnist_train, mnist_val = train_test_split(
#     moving_mnist,
#     test_size=0.99,
#     random_state=6
# )
# mnist_test, mnist_val = train_test_split(
#     mnist_val,
#     test_size=0.01,
#     random_state=6
# )
# test_set = data.MnistDataset(mnist = mnist_test)
# test_dataloader = data.DataLoader(test_set, batch_size=batch_size, shuffle = True)

train_set = data.MnistDataset(mnist = mnist_train)
val_set = data.MnistDataset(mnist = mnist_val)

train_dataloader = data.DataLoader(train_set, batch_size=batch_size, shuffle = True)
val_dataloader = data.DataLoader(val_set, batch_size=batch_size ,shuffle = True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
writer = SummaryWriter("tensorboard-1")

best_loss = 100000
#train_dataloader = (torch.randn(30, 3 , 20, 1, 64, 64)).float()
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('----------')
    train_loss ,itr = train(model,train_dataloader,optimizer,itr,eta, delta,batch_size,patch_size,device)
    print("#Train")
    print(f'loss : {round(float(train_loss),7)} \n')
   
    
    val_loss = (eval_model(model,val_dataloader,eta, delta,batch_size,patch_size,best_loss,device))[0]
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch+1
        torch.save(model.state_dict(), 'best_model_state.bin')
    
    print("\n#Val")
    print(f'loss : {round(float(val_loss),7)} \n')
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    writer.add_scalars("loss ",{
    'train': np.asscalar(train_loss),
    'validation': np.asscalar(val_loss),
    }, epoch+1)

    print("\n")


    
print("Best loss : " + str(best_loss))
print("Best epoch : " + str(best_epoch))



# model.load_state_dict(torch.load('best_model_state.bin'))
# model = model.to(device)


# test_loss = (eval_model(model,test_dataloader,loss_function,eta, delta,batch_size,patch_size,best_loss,test = True))[0]


# test_msg = "TEST LOSS " + test_loss

# writer.add_text('RESULT', test_msg)
