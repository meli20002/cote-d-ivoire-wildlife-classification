import torch.nn as nn
import torch



def train_model(model,train_loader,criterion,optimizer,num_epochs):
    train_losses=[]
    train_accuracies=[]

    for epoch in  range(num_epochs):
        model.train()
        running_loss=0
        correct=0
        total=0
        for inputs,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        
        epoch_loss=running_loss/len(train_loader)
        epoch_acc=100 *correct/total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    return train_losses, train_accuracies

            
