import torch
from lossNacc import loss_fn_kd

def train_model(model, loss_function, optimizer, data_loader):
    # set model to train mode
    model.train()
    # Step

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1) #todo check loss
            loss = loss_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train loss : {:.4f}; Acuracy : {:.4f}'.format(total_loss, total_acc))
    return total_loss, total_acc

def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))
    return total_loss, total_acc

def train_kd(model, teacher_model, student_optimizer, data_loader, alpha, T):
    # set model to train mode
    model.train()

    loss_function = loss_fn_kd

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        student_optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            teacher_outputs = teacher_model(inputs)
            _, predictions = torch.max(outputs, 1)  # todo check loss
            loss = loss_function(outputs, labels, teacher_outputs, alpha, T)

            # backward
            loss.backward()
            student_optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train loss : {:.4f}; Acuracy : {:.4f}'.format(total_loss, total_acc))
    return total_loss, total_acc
