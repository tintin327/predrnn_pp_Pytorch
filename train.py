from model import RNN
import torch
import numpy as np
from reshape_tensor import reshape_patch,reshape_patch_back


def train(model,data_loader,optimizer,itr,eta,delta,batch_size,patch_size,device):
    model = model.train()
    losses = []
    #device = "cpu
    loss_function = (torch.nn.MSELoss()).to(device)
    seq_length = 20
    input_length = 10
    img_width = 64
    img_channel = 1
        
    for data in data_loader:
        if(data.shape[0]!=batch_size):
            continue
            
        data = data.to(device)
        
        for flip in range(2):
        
            mnist_input = reshape_patch(data,patch_size)

            if itr < 50000:
                eta -= delta
            else:
                eta = 0.0

            random_flip = np.random.random_sample(
                (batch_size,seq_length-input_length-1))
            true_token = (random_flip < eta)
            #true_token = (random_flip < pow(base,itr))

            ones = np.ones((int(img_width/patch_size),
                            int(img_width/patch_size),
                            patch_size**2*img_channel))
            zeros = np.zeros((int(img_width/patch_size),
                            int(img_width/patch_size),
                            patch_size**2*img_channel))
            mask_true = []
            for i in range(batch_size):
                for j in range(seq_length-input_length-1):
                    if true_token[i,j]:
                        mask_true.append(ones)
                    else:
                        mask_true.append(zeros)
            mask_true = np.array(mask_true)
            mask_true = np.reshape(mask_true, (batch_size,
                                               seq_length-input_length-1,
                                                         int(img_width/patch_size),
                                     int(img_width/patch_size),
                                               patch_size**2*img_channel))
            mask_true = torch.from_numpy(mask_true)
            mask_true = (mask_true.permute(0,1,4,2,3)).to(device)

            outputs, loss = (model(mnist_input.float(),mask_true.float()))
            outputs = reshape_patch_back(outputs,patch_size)


            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if flip == 0:
                data = torch.flip(data, [1])
                
    np.save('train_p', np.uint8(((outputs).cpu()).detach().numpy()))
    np.save('train_t', np.uint8(((data*255).cpu()).detach().numpy()))

    return [np.mean(losses), itr] #itr###

def eval_model(model,data_loader,eta,delta,batch_size,patch_size,best_loss,device,test = False):

    model = model.eval()
    losses = []
    loss_function = (torch.nn.MSELoss()).to(device)
    seq_length = 20
    input_length = 10
    img_width = 64
    img_channel = 1
    save_num = 0
    with torch.no_grad():   
        for data in data_loader:
            if(data.shape[0]!=batch_size):
                continue
            data = data.to(device)
            
            mnist_input = reshape_patch(data,patch_size)

            mask_true = torch.from_numpy(np.zeros((batch_size,
                              seq_length-input_length-1,
                              int(patch_size**2*img_channel),
                              int(img_width/patch_size),
                              int(img_width/patch_size))))
            mask_true = mask_true.to(device)
            
            outputs, loss = (model(mnist_input.float(),mask_true.float(),test = test))
            outputs = reshape_patch_back(outputs,patch_size)


            losses.append(loss.item())
            
            if(save_num==0):
                save_num += 1
                Outputs = outputs
                Mnist_target = mnist_input
                
            if save_num <5 :
                save_num += 1
                Outputs = torch.cat((Outputs,outputs),0)
                Mnist_target = torch.cat((Mnist_target,mnist_input),0)
                
    if test is False :
        if best_loss > np.mean(losses):
            np.save('val_p', np.uint8(((Outputs).cpu()).detach().numpy()))
            np.save('val_t', np.uint8(((data*255).cpu()).detach().numpy()))
            print("current loss " + str(np.mean(losses)) + " < best loss " + str(best_loss))
    else:
        np.save('test_p', np.uint8(((Outputs).cpu()).detach().numpy()))
        np.save('test_t', np.uint8(((data*255).cpu()).detach().numpy()))



    return [np.mean(losses)] #itr###

