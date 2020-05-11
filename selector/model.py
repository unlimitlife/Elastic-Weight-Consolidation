import net
import torch.nn as nn


def model(model, device, num_classes, classes_per_task):
    model = getattr(net, model)(num_classes=num_classes, classes_per_task=classes_per_task)
    model.to(device)
    #model = nn.DataParallel(model)

    return model
