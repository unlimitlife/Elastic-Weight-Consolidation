import selector
import os
import copy
import torch.utils.data
import utils
from dataset import taskset
import torch.nn as nn
from .test import test, make_heatmap
from .loss import *
from copy import deepcopy
import torch.nn.functional as F


def train(config, method_config, data_config, logger):
    """
    Args:
        config (dict): config file dictionary.
            model (str): name of network. [selector.model(model, ...)]
            classes_per_task (int): classes per task.
            DA (bool): if True, apply data augment.
            memory_cap (int): sample memory size.
            num_workers (int): how many subprocesses to use for data loading.
                               0 means that the data will be loaded in the main process. (default: 0)
            batch_size (int): how many samples per batch to load. (default: 1)
            device (torch.device): gpu or cpu.
            data_path (str): root directory of dataset.
            save_path (str): directory for save. (not taskwise)
        data_config (dict): data config file dictionary.
            dataset (str): name of dataset.
            total_classes (int): total class number of dataset.
            transform (dict):
                train (transforms): transforms for train dataset.
                test (transforms): transforms for test dataset.
            curriculums (list): curriculum list.
            classes (list): class name list.
        method_config (dict): method config file dictionary.
            method (str): name of method.
            process_list (list): process list.
            package (string): current package name.
        logger (Logger): logger for the tensorboard.
    """
    model_ = config['model']
    classes_per_task = config['classes_per_task']
    memory_cap = config['memory_cap']
    device = config['device']
    data_path = config['data_path']
    save_path = config['save_path']

    total_classes = data_config['total_classes']
    train_transform = data_config['transform']['train']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']

    '''Split curriculum [[task0], [task1], [task2], ...]'''
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    '''Make sample memory'''
    sample_memory = taskset.SampleMemory(data_path, total_classes, len(curriculum), curriculum,
                                         transform=train_transform, capacity=memory_cap)

    model = selector.model(model_, device, total_classes, classes_per_task)
    fisher = None
    importance = 75000
    alpha=0.9
    normalize=True
    test_task = []
    train_task = []
    '''Taskwise iteration'''
    for task_idx, task in enumerate(curriculum):
        train_task = task
        train_taskset = taskset.Taskset(data_path, train_task, task_idx, train=True, transform=train_transform)
        test_task.extend(task)
        test_taskset = taskset.Taskset(data_path, test_task, 0, train=False, transform=test_transform)

        '''Make directory of current task'''
        if not os.path.exists(os.path.join(save_path, 'task%02d' % task_idx)):
            os.makedirs(os.path.join(save_path, 'task%02d' % task_idx))

        config['task_path'] = os.path.join(save_path, 'task%02d' % task_idx)
        config['cml_classes'] = len(test_task)
        

        ###
        """
        model = selector.model(model, device, len(test_task))
        if method_config["flag_ewc"] == True:
            for n, g in net.named_parameters():
                if 'fc' in n:
                    g[:len(method_config["old_param"][n])].data.copy_(method_config["old_param"][n].data.clone())
                else:
                    g.data.copy_(method_config["old_param"][n].data.clone())
        """
        ###
        model_old = deepcopy(model)
        for p in model_old.parameters():
            p.requires_grad = False
        ewc = EWCPPLoss(model, model_old, fisher=fisher, alpha=alpha, normalize=normalize)
        model, ewc = _train(task_idx, model, ewc, importance, train_taskset, sample_memory, test_taskset, config, method_config, data_config, logger)

        fisher = deepcopy(ewc.get_fisher())
        #sample_memory.update()
        torch.save(sample_memory, os.path.join(config['task_path'], 'sample_memory'))


def _train(task_idx, model, ewc, importance, train_taskset, sample_memory, test_taskset, config, method_config, data_config, logger):
    """
    Args:
        config (dict): config file dictionary.
            task_path (str): directory for save. (taskwise, save_path + task**)
            cml_classes (int): size of cumulative taskset
    """
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    task_path = config['task_path']
    num_workers = config['num_workers']
    device = config['device']
    cml_classes = config['cml_classes']

    process_list = method_config['process_list']

    log = utils.Log(task_path)
    epoch = 0
    single_best_accuracy, multi_best_accuracy = 0.0, 0.0

    for process in process_list:
        epochs = process['epochs']
        balance_finetune = process['balance_finetune']
        optimizer = process['optimizer'](model.parameters())
        scheduler = process['scheduler'](optimizer)

        if balance_finetune and cml_classes != classes_per_task:
            train_set = copy.deepcopy(sample_memory)
            train_set.update(BF=True)
        else:
            train_set = torch.utils.data.ConcatDataset((train_taskset, sample_memory))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
        
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)
        log.info("Start Training")
        for ep in range(epochs):
            log.info("%d Epoch Started" % epoch)
            model.train()
            epoch_loss = 0.0
            total = 0
            ce_loss = 0.0
            con_loss = 0.0

            for i, data in enumerate(train_loader):
                utils.printProgressBar(i + 1, len(train_loader), prefix='train')
                images, labels = data[0].to(device), data[1].to(device)
                cur_batch_size = images.size(0)
                optimizer.zero_grad()
                outputs = model(images, task_idx)
                loss = F.cross_entropy(outputs, labels)
                epoch_loss += loss#.item() * cur_batch_size
                """if type(loss) == tuple:
                    ce_loss += loss[0].item() * cur_batch_size
                    con_loss += loss[1].item() * cur_batch_size
                    loss = 0.1*loss[0] + loss[1]
                else:
                    ce_loss += loss.item() * cur_batch_size"""
                loss.backward()
                ewc.update()
                loss_ewc = importance * ewc.penalty()
                if loss_ewc != 0:
                    loss_ewc.backward()
                optimizer.step()
                total += cur_batch_size

            epoch_loss /= total
            """ce_loss /= total
            con_loss /= total"""
            selector.scheduler.step(scheduler, epoch_loss)
            """
            log.info("[CE_LOSS] : %.3lf" % ce_loss)
            if con_loss != 0:
                log.info("[Consolidate_LOSS] : %.3lf" % con_loss)"""
            log.info("epoch: %d  train_loss: %.3lf  train_sample: %d" % (epoch, epoch_loss, total))
            test_loss, single_total_accuracy, single_class_accuracy, multi_total_accuracy, multi_class_accuracy = \
                test(model, test_loader, config, data_config)
            single_best_accuracy, multi_best_accuracy = \
                utils.save_model(model, single_best_accuracy, multi_best_accuracy,
                                 single_total_accuracy, single_class_accuracy,
                                 multi_total_accuracy, multi_class_accuracy, task_path, ep, epochs)
            logger.printStatics(epoch_loss, test_loss, single_total_accuracy)
            logger.epoch_step()
            epoch += 1

        make_heatmap(model, test_loader, config)
        log.info("Finish Training")
        return model, ewc
