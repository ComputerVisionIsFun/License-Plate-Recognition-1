import torch
import torch.optim as optim
import time
from termcolor import colored

def train(model, train_dataloader, test_dataloader, optimizer, criterion, milestones,
          pretrained_path, n_epochs, model_save_folder, early_stop):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    schedler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=.5)

    # load pretrained model
    if pretrained_path!='None':
        model = torch.load(pretrained_path)

    # 
    model.to(device)

    # 
    for epoch in range(1, n_epochs + 1):
        start = time.time()

        for data in train_dataloader:
            # to device

            # zero the gradients of parameters of the model

            # forward

            # calculate the loss

            # loss backward


            # update optimizer


            # 
            end = time.time()
            time_consuming_epoch = round(end - start, 2)


            # save weights


            # print epoch-info
            print('Epoch({}/{})'.format(epoch,n_epochs))


            # early-stop


