import os
import torch.cuda
import torchvision.transforms as transforms
import selector
import utils

config = {
    "num_workers": 8,
    "batch_size": 128,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "data_path": os.path.join(os.getcwd(), 'data'),
    "save_path": os.path.join(os.getcwd(), 'saved_results'),
}

method_config = {
    "ewc++": {
        "method": "ewc++",
        "process_list":
            [{"epochs": 120, "balance_finetune": False,
              "optimizer": selector.optimizer.SGD(lr=0.1, momentum=0.9, weight_decay=0),
              "scheduler": selector.scheduler.MultiStepLR([84,108], gamma=0.1)}],
        "package": "method.ewc++",
    }
}
"""
    "ewc++": {
        "method": "ewc++",
        "process_list":
            [{"epochs": 60, "balance_finetune": False,
              "optimizer": selector.optimizer.Adam(lr=1e-3, betas=(0.9, 0.999)),
              "scheduler": selector.scheduler.StepLR(step_size=70, gamma=0.1)}],
        "package": "method.ewc++",
    }
"""
data_config = {
    "mnist": {
        "dataset": "mnist",
        "total_classes": 10,
        "transform": {
            "train": transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "test": transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        },
        "curriculums": {
            "basic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "rand1": [8, 4, 3, 6, 9, 0, 1, 7, 5, 2],
            "rand2": [0, 2, 4, 9, 6, 8, 1, 7, 3, 5],
        },
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "stl10": {
        "dataset": "stl10",
        "total_classes": 10,
        "transform": {
            "train": transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "test": transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        },
        "curriculums": {
            "basic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "rand1": [8, 4, 3, 6, 9, 0, 1, 7, 5, 2],
            "rand2": [0, 2, 4, 9, 6, 8, 1, 7, 3, 5],
        },
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "cifar10": {
        "dataset": "cifar10",
        "total_classes": 10,
        "transform": {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        },
        "curriculums": {
            "basic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "rand1": [8, 4, 3, 6, 9, 0, 1, 7, 5, 2],
            "rand2": [0, 2, 4, 9, 6, 8, 1, 7, 3, 5],
        },
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "cifar100": {
        "dataset": "cifar100",
        "total_classes": 100,
        "transform": {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5070757, 0.48654833, 0.4409185), (0.26733443, 0.25643864, 0.2761505))
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070757, 0.48654833, 0.4409185), (0.26733443, 0.25643864, 0.2761505)),
            ])
        },
        "curriculums": {
            "basic": selector.curriculums.basic(100),
            "best": [68, 94, 48, 12, 53, 82, 9, 60, 76, 20, 87, 8, 56, 71, 0, 21, 28,
                     17, 23, 39, 66, 75, 85, 89, 90, 1, 24, 41, 49, 54, 69, 62, 43, 58,
                     97, 37, 57, 81, 91, 99, 6, 16, 36, 79, 15, 34, 88, 42, 86, 22, 51,
                     84, 5, 14, 52, 95, 70, 31, 29, 40, 47, 61, 63, 83, 92, 96, 18, 7,
                     32, 25, 26, 33, 45, 46, 59, 77, 13, 19, 78, 93, 30, 38, 80, 98, 2,
                     73, 4, 10, 27, 67, 50, 64, 74, 3, 44, 65, 11, 35, 55, 72],
            "best1": [60, 17, 0, 76, 56, 21, 87, 48, 68, 28, 82, 23, 94, 20, 39, 53, 8,
                      71, 12, 9, 58, 41, 69, 37, 24, 66, 90, 49, 89, 54, 99, 62, 97, 43,
                      57, 91, 75, 85, 1, 81, 79, 14, 86, 6, 31, 40, 88, 36, 51, 95, 34,
                      42, 70, 29, 84, 52, 16, 15, 5, 22, 92, 33, 47, 77, 93, 26, 13, 19,
                      25, 32, 46, 78, 63, 59, 45, 61, 96, 83, 7, 18, 4, 67, 30, 27, 55,
                      74, 73, 80, 3, 72, 50, 98, 11, 65, 35, 64, 10, 2, 38, 44],
            "best2": [9, 8, 23, 76, 68, 28, 82, 12, 21, 60, 48, 0, 20, 53, 39, 17, 94,
                      87, 71, 56, 62, 49, 75, 37, 91, 66, 85, 99, 43, 24, 57, 97, 69, 1,
                      58, 90, 89, 54, 81, 41, 52, 36, 29, 79, 34, 5, 86, 51, 84, 70, 14,
                      22, 95, 16, 6, 15, 31, 40, 88, 42, 61, 59, 63, 46, 18, 96, 83, 13,
                      45, 25, 47, 77, 93, 33, 92, 32, 78, 7, 26, 19, 10, 38, 74, 2, 50,
                      98, 64, 67, 11, 3, 27, 72, 35, 4, 65, 30, 44, 80, 73, 55],
            "best3": [28, 21, 9, 68, 53, 56, 82, 17, 8, 0, 20, 76, 94, 71, 12, 23, 39,
                      87, 48, 60, 66, 69, 37, 41, 81, 85, 49, 43, 57, 89, 97, 1, 54, 75,
                      58, 91, 90, 62, 99, 24, 70, 84, 95, 34, 16, 22, 42, 51, 15, 79, 86,
                      36, 31, 14, 29, 88, 52, 5, 40, 6, 78, 45, 77, 46, 47, 19, 33, 18,
                      61, 7, 96, 63, 83, 93, 92, 59, 32, 25, 26, 13, 80, 74, 30, 4, 3,
                      55, 64, 44, 27, 73, 65, 67, 11, 98, 35, 38, 72, 10, 2, 50],
            "best4": [48, 17, 87, 0, 12, 60, 23, 28, 20, 9, 56, 94, 82, 71, 21, 39, 76,
                      53, 8, 68, 85, 1, 37, 54, 66, 89, 75, 81, 57, 97, 62, 90, 91, 43,
                      49, 24, 41, 99, 58, 69, 84, 42, 6, 88, 52, 95, 36, 70, 14, 16, 79,
                      34, 15, 29, 86, 40, 51, 22, 5, 31, 47, 92, 18, 83, 78, 63, 25, 61,
                      7, 33, 46, 26, 45, 19, 32, 96, 77, 93, 13, 59, 67, 98, 73, 44, 72,
                      35, 30, 2, 4, 74, 3, 50, 11, 10, 38, 55, 27, 80, 64, 65],
            "best5": [20, 23, 28, 21, 71, 48, 39, 17, 82, 76, 60, 56, 94, 8, 53, 87, 0,
                      9, 12, 68, 24, 99, 75, 97, 49, 89, 91, 57, 1, 54, 90, 62, 81, 66,
                      37, 58, 69, 43, 41, 85, 6, 84, 16, 86, 15, 42, 52, 95, 34, 51, 88,
                      36, 29, 40, 22, 5, 79, 70, 14, 31, 83, 18, 47, 7, 32, 26, 13, 93,
                      63, 45, 96, 19, 77, 46, 59, 78, 25, 33, 92, 61, 2, 27, 30, 44, 73,
                      38, 98, 4, 3, 74, 10, 80, 11, 50, 72, 67, 65, 35, 55, 64],
            "worst": [50, 64, 74, 3, 44, 65, 11, 35, 55, 72, 30, 38, 80, 98, 2, 73, 4,
                      10, 27, 67, 26, 33, 45, 46, 59, 77, 13, 19, 78, 93, 47, 61, 63, 83,
                      92, 96, 18, 7, 32, 25, 51, 84, 5, 14, 52, 95, 70, 31, 29, 40, 6,
                      16, 36, 79, 15, 34, 88, 42, 86, 22, 69, 62, 43, 58, 97, 37, 57, 81,
                      91, 99, 66, 75, 85, 89, 90, 1, 24, 41, 49, 54, 87, 8, 56, 71, 0,
                      21, 28, 17, 23, 39, 68, 94, 48, 12, 53, 82, 9, 60, 76, 20],
            "rand1": [36, 35, 52, 91, 87, 48, 17, 31, 51, 40, 3, 18, 16, 74, 68, 23, 50,
                      75, 65, 5, 85, 73, 14, 2, 53, 64, 72, 61, 81, 41, 28, 59, 42, 60,
                      25, 63, 66, 76, 29, 24, 93, 82, 62, 69, 30, 47, 8, 96, 27, 88, 7,
                      34, 4, 6, 83, 1, 58, 67, 71, 92, 57, 49, 33, 95, 56, 9, 19, 80,
                      13, 46, 12, 43, 44, 99, 55, 10, 84, 89, 22, 79, 38, 21, 98, 0, 39,
                      86, 54, 97, 37, 26, 70, 15, 45, 78, 20, 32, 11, 90, 77, 94],
            "rand2": [4, 11, 92, 41, 42, 6, 29, 15, 47, 98, 53, 68, 8, 18, 25, 45, 59,
                      70, 65, 64, 90, 55, 88, 31, 9, 99, 7, 16, 51, 74, 73, 60, 5, 84,
                      97, 85, 1, 49, 40, 34, 14, 83, 48, 19, 17, 79, 50, 54, 2, 96, 36,
                      26, 43, 0, 58, 71, 22, 72, 82, 44, 21, 94, 62, 75, 52, 30, 33, 95,
                      89, 28, 76, 23, 63, 24, 81, 39, 93, 27, 37, 56, 86, 66, 67, 35, 77,
                      3, 32, 69, 61, 13, 80, 10, 78, 20, 12, 46, 87, 38, 57, 91],
            "base1": [30, 74, 77, 91, 67, 43, 81, 99, 45, 15, 16, 51, 49, 83, 39, 48, 29, 55, 47, 1, 23, 6, 8, 94, 56,
                      75, 0, 28, 50, 31, 36, 58, 60, 14, 11, 25, 97, 21, 26, 53, 70, 95, 13, 63, 52, 98, 89, 22, 3, 41,
                      32, 7, 93, 66, 17, 90, 78, 33, 5, 10, 37, 69, 86, 54, 64, 71, 96, 65, 20, 9, 85, 62, 34, 61, 27,
                      46, 2, 24, 57, 88, 79, 72, 84, 82, 38, 92, 59, 4, 73, 18, 80, 12, 19, 44, 68, 40, 87, 42, 76, 35],
            "base2": [25, 42, 11, 22, 88, 30, 34, 96, 44, 37, 4, 35, 62, 54, 48, 46, 26, 84, 28, 63, 17, 94, 74, 29, 40,
                      53, 57, 32, 16, 66, 6, 73, 12, 10, 20, 39, 72, 61, 27, 75, 33, 8, 69, 85, 98, 14, 49, 78, 64, 2,
                      13, 52, 50, 97, 80, 18, 67, 43, 89, 68, 81, 60, 24, 15, 51, 9, 3, 82, 76, 90, 77, 45, 1, 70, 91,
                      99, 55, 71, 7, 41, 65, 93, 92, 36, 83, 58, 31, 21, 79, 87, 38, 0, 86, 56, 59, 5, 19, 95, 23, 47],
            "super": [4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 54, 62, 70, 82, 92, 9, 10, 16, 28, 61, 0, 51, 53, 57, 83,
                      22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 6, 7, 14, 18, 24, 3, 42, 43, 88, 97, 12, 17, 37, 68, 76,
                      23, 33, 49, 60, 71, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 26, 45, 77, 79, 99, 2, 11, 35, 46, 98,
                      27, 29, 44, 78, 93, 36, 50, 65, 74, 80, 47, 52, 56, 59, 96, 8, 13, 48, 58, 90, 41, 69, 81, 85,
                      89],
            "super1": [4, 30, 55, 72, 95, 6, 7, 14, 18, 24, 34, 63, 64, 66, 75, 1, 32, 67, 73, 91, 3, 42, 43, 88, 97, 8,
                       13, 48, 58, 90, 15, 19, 21, 31, 38, 12, 17, 37, 68, 76, 22, 39, 40, 86, 87, 0, 51, 53, 57, 83,
                       27, 29, 44, 78, 93, 47, 52, 56, 59, 96, 41, 69, 81, 85, 89, 9, 10, 16, 28, 61, 26, 45, 77, 79,
                       99, 23, 33, 49, 60, 71, 54, 62, 70, 82, 92, 2, 11, 35, 46, 98, 5, 20, 25, 84, 94, 36, 50, 65, 74,
                       80],
            "super2": [5, 20, 25, 84, 94, 15, 19, 21, 31, 38, 1, 32, 67, 73, 91, 47, 52, 56, 59, 96, 23, 33, 49, 60, 71,
                       22, 39, 40, 86, 87, 26, 45, 77, 79, 99, 9, 10, 16, 28, 61, 4, 30, 55, 72, 95, 34, 63, 64, 66, 75,
                       2, 11, 35, 46, 98, 54, 62, 70, 82, 92, 41, 69, 81, 85, 89, 27, 29, 44, 78, 93, 36, 50, 65, 74,
                       80, 3, 42, 43, 88, 97, 0, 51, 53, 57, 83, 6, 7, 14, 18, 24, 8, 13, 48, 58, 90, 12, 17, 37, 68,
                       76],
            "another0": [30, 11, 64, 91, 27, 21, 89, 31, 28, 63, 1, 50, 85, 24, 37, 99, 73,
                         53, 3, 90, 45, 51, 47, 38, 10, 93, 56, 81, 67, 7, 72, 79, 15, 35,
                         82, 40, 76, 96, 23, 71, 32, 86, 46, 65, 88, 14, 98, 54, 44, 34, 62,
                         13, 36, 6, 18, 29, 60, 78, 42, 0, 20, 83, 41, 84, 39, 17, 5, 52,
                         57, 80, 94, 12, 8, 2, 26, 75, 19, 25, 61, 9, 49, 68, 16, 87, 92,
                         4, 59, 58, 70, 43, 77, 97, 55, 95, 48, 22, 74, 69, 33, 66],
            "another1": [16, 24, 70, 74, 26, 87, 73, 22, 4, 75, 27, 42, 21, 37, 9, 84, 90,
                         79, 40, 77, 57, 1, 86, 46, 3, 19, 45, 66, 83, 33, 88, 17, 95, 64,
                         55, 98, 76, 12, 7, 96, 52, 23, 43, 67, 39, 65, 44, 59, 48, 35, 58,
                         41, 93, 49, 6, 5, 68, 32, 18, 51, 20, 85, 2, 31, 80, 91, 82, 97,
                         94, 54, 72, 15, 71, 50, 13, 53, 47, 25, 62, 81, 99, 29, 10, 63, 11,
                         30, 36, 34, 14, 28, 8, 69, 89, 0, 92, 38, 56, 78, 60, 61],
            "another2": [68, 75, 55, 29, 24, 82, 32, 48, 85, 69, 60, 36, 77, 13, 35, 26, 39,
                         10, 81, 15, 45, 27, 65, 67, 30, 4, 51, 28, 18, 52, 59, 79, 86, 34,
                         70, 47, 9, 2, 44, 46, 7, 33, 11, 17, 40, 71, 19, 23, 62, 96, 73,
                         58, 63, 41, 6, 74, 20, 87, 94, 0, 91, 22, 42, 78, 72, 93, 31, 76,
                         56, 12, 37, 21, 61, 66, 84, 38, 89, 57, 25, 1, 16, 53, 88, 54, 64,
                         50, 49, 99, 5, 90, 3, 80, 95, 43, 14, 8, 98, 92, 97, 83],
            "another3": [16, 92, 35, 65, 4, 61, 38, 96, 68, 90, 45, 12, 1, 19, 37, 14, 34,
                         46, 39, 24, 55, 60, 71, 91, 87, 49, 62, 28, 6, 79, 7, 31, 70, 53,
                         72, 13, 88, 11, 95, 58, 73, 80, 75, 40, 27, 21, 67, 48, 99, 5, 52,
                         89, 63, 64, 36, 81, 56, 93, 97, 43, 74, 32, 20, 57, 85, 76, 9, 86,
                         10, 77, 98, 22, 42, 82, 59, 78, 94, 17, 8, 47, 3, 41, 69, 23, 50,
                         44, 18, 84, 30, 54, 66, 83, 0, 26, 2, 15, 25, 33, 29, 51],
            "another4": [37, 97, 15, 35, 81, 69, 41, 94, 7, 20, 60, 46, 55, 79, 14, 67, 34,
                         2, 43, 5, 26, 38, 61, 62, 58, 54, 51, 89, 47, 66, 80, 23, 63, 76,
                         93, 17, 1, 64, 91, 44, 73, 9, 92, 12, 90, 30, 70, 75, 49, 48, 59,
                         18, 86, 88, 83, 42, 33, 6, 24, 36, 68, 77, 11, 13, 98, 31, 56, 29,
                         50, 71, 78, 96, 28, 3, 25, 82, 53, 8, 32, 39, 57, 0, 10, 4, 22,
                         84, 40, 74, 65, 19, 99, 95, 16, 27, 45, 52, 72, 87, 85, 21],
            "another5": [86, 52, 32, 17, 11, 47, 24, 35, 46, 60, 16, 91, 9, 65, 72, 29, 6,
                         95, 66, 89, 85, 50, 14, 94, 68, 61, 27, 55, 2, 22, 74, 48, 13, 19,
                         12, 0, 81, 67, 88, 84, 98, 92, 64, 41, 37, 20, 43, 18, 87, 93, 39,
                         15, 62, 97, 44, 79, 1, 26, 31, 53, 4, 49, 28, 54, 38, 25, 69, 34,
                         77, 56, 63, 75, 3, 80, 23, 51, 96, 78, 21, 73, 71, 59, 45, 33, 42,
                         7, 5, 90, 99, 8, 40, 70, 36, 76, 57, 83, 30, 10, 58, 82],
            "another6": [49, 48, 78, 50, 15, 17, 58, 98, 96, 69, 25, 18, 14, 10, 56, 99, 28,
                         24, 76, 22, 39, 74, 40, 70, 2, 90, 68, 35, 6, 91, 12, 3, 66, 29,
                         87, 54, 92, 94, 34, 9, 97, 36, 81, 61, 45, 8, 37, 62, 79, 65, 57,
                         83, 21, 33, 82, 47, 59, 4, 67, 19, 95, 1, 31, 73, 88, 75, 27, 30,
                         5, 85, 84, 43, 46, 86, 23, 41, 13, 52, 38, 7, 63, 32, 77, 20, 80,
                         53, 11, 71, 0, 93, 16, 51, 72, 55, 44, 26, 64, 89, 42, 60],
            "another7": [73, 86, 18, 97, 24, 74, 88, 41, 7, 1, 65, 60, 49, 15, 26, 39, 17,
                         36, 20, 35, 61, 51, 91, 53, 13, 45, 55, 10, 27, 38, 11, 16, 85, 52,
                         14, 98, 33, 0, 90, 93, 28, 82, 58, 70, 87, 69, 8, 40, 92, 4, 78,
                         32, 25, 22, 96, 67, 21, 63, 19, 81, 71, 75, 47, 9, 6, 43, 37, 56,
                         3, 30, 66, 68, 79, 72, 99, 54, 34, 57, 59, 29, 46, 23, 5, 76, 42,
                         64, 77, 89, 31, 12, 44, 2, 80, 84, 94, 50, 48, 62, 83, 95],
            "another8": [4, 40, 69, 81, 1, 76, 80, 48, 35, 86, 63, 64, 78, 32, 96, 41, 74,
                         8, 20, 59, 55, 21, 9, 29, 28, 26, 17, 66, 6, 42, 77, 7, 16, 36,
                         5, 47, 30, 85, 43, 11, 73, 12, 50, 18, 82, 10, 38, 45, 79, 92, 58,
                         89, 15, 23, 68, 3, 49, 39, 71, 44, 95, 57, 90, 34, 83, 31, 0, 52,
                         53, 61, 98, 33, 25, 84, 65, 99, 67, 54, 88, 13, 97, 2, 94, 62, 27,
                         72, 75, 87, 24, 37, 22, 60, 14, 70, 91, 46, 19, 56, 93, 51],
            "another9": [35, 77, 62, 63, 88, 93, 73, 32, 16, 75, 27, 23, 48, 15, 56, 46, 0,
                         69, 59, 60, 26, 58, 64, 80, 72, 68, 97, 45, 81, 10, 11, 43, 20, 82,
                         49, 53, 83, 95, 61, 18, 84, 9, 89, 30, 40, 86, 44, 31, 25, 91, 52,
                         90, 8, 66, 7, 19, 65, 42, 96, 74, 51, 92, 5, 39, 4, 6, 78, 99,
                         38, 28, 55, 33, 29, 47, 36, 67, 87, 98, 71, 37, 94, 12, 14, 70, 24,
                         1, 13, 2, 57, 17, 34, 22, 54, 76, 41, 3, 79, 85, 50, 21],
        },
        "classes": ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    },
    "aug_cifar100": {
        "dataset": "cifar100",
        "total_classes": 100,
        "transform": {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5070757, 0.48654833, 0.4409185), (0.26733443, 0.25643864, 0.2761505))
            ]),
            "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070757, 0.48654833, 0.4409185), (0.26733443, 0.25643864, 0.2761505)),
            ])
        },
        "curriculums": {
            "basic": selector.curriculums.basic(100),
            "best": [68, 94, 48, 12, 53, 82, 9, 60, 76, 20, 87, 8, 56, 71, 0, 21, 28,
                     17, 23, 39, 66, 75, 85, 89, 90, 1, 24, 41, 49, 54, 69, 62, 43, 58,
                     97, 37, 57, 81, 91, 99, 6, 16, 36, 79, 15, 34, 88, 42, 86, 22, 51,
                     84, 5, 14, 52, 95, 70, 31, 29, 40, 47, 61, 63, 83, 92, 96, 18, 7,
                     32, 25, 26, 33, 45, 46, 59, 77, 13, 19, 78, 93, 30, 38, 80, 98, 2,
                     73, 4, 10, 27, 67, 50, 64, 74, 3, 44, 65, 11, 35, 55, 72],
            "best1": [60, 17, 0, 76, 56, 21, 87, 48, 68, 28, 82, 23, 94, 20, 39, 53, 8,
                      71, 12, 9, 58, 41, 69, 37, 24, 66, 90, 49, 89, 54, 99, 62, 97, 43,
                      57, 91, 75, 85, 1, 81, 79, 14, 86, 6, 31, 40, 88, 36, 51, 95, 34,
                      42, 70, 29, 84, 52, 16, 15, 5, 22, 92, 33, 47, 77, 93, 26, 13, 19,
                      25, 32, 46, 78, 63, 59, 45, 61, 96, 83, 7, 18, 4, 67, 30, 27, 55,
                      74, 73, 80, 3, 72, 50, 98, 11, 65, 35, 64, 10, 2, 38, 44],
            "best2": [9, 8, 23, 76, 68, 28, 82, 12, 21, 60, 48, 0, 20, 53, 39, 17, 94,
                      87, 71, 56, 62, 49, 75, 37, 91, 66, 85, 99, 43, 24, 57, 97, 69, 1,
                      58, 90, 89, 54, 81, 41, 52, 36, 29, 79, 34, 5, 86, 51, 84, 70, 14,
                      22, 95, 16, 6, 15, 31, 40, 88, 42, 61, 59, 63, 46, 18, 96, 83, 13,
                      45, 25, 47, 77, 93, 33, 92, 32, 78, 7, 26, 19, 10, 38, 74, 2, 50,
                      98, 64, 67, 11, 3, 27, 72, 35, 4, 65, 30, 44, 80, 73, 55],
            "best3": [28, 21, 9, 68, 53, 56, 82, 17, 8, 0, 20, 76, 94, 71, 12, 23, 39,
                      87, 48, 60, 66, 69, 37, 41, 81, 85, 49, 43, 57, 89, 97, 1, 54, 75,
                      58, 91, 90, 62, 99, 24, 70, 84, 95, 34, 16, 22, 42, 51, 15, 79, 86,
                      36, 31, 14, 29, 88, 52, 5, 40, 6, 78, 45, 77, 46, 47, 19, 33, 18,
                      61, 7, 96, 63, 83, 93, 92, 59, 32, 25, 26, 13, 80, 74, 30, 4, 3,
                      55, 64, 44, 27, 73, 65, 67, 11, 98, 35, 38, 72, 10, 2, 50],
            "best4": [48, 17, 87, 0, 12, 60, 23, 28, 20, 9, 56, 94, 82, 71, 21, 39, 76,
                      53, 8, 68, 85, 1, 37, 54, 66, 89, 75, 81, 57, 97, 62, 90, 91, 43,
                      49, 24, 41, 99, 58, 69, 84, 42, 6, 88, 52, 95, 36, 70, 14, 16, 79,
                      34, 15, 29, 86, 40, 51, 22, 5, 31, 47, 92, 18, 83, 78, 63, 25, 61,
                      7, 33, 46, 26, 45, 19, 32, 96, 77, 93, 13, 59, 67, 98, 73, 44, 72,
                      35, 30, 2, 4, 74, 3, 50, 11, 10, 38, 55, 27, 80, 64, 65],
            "best5": [20, 23, 28, 21, 71, 48, 39, 17, 82, 76, 60, 56, 94, 8, 53, 87, 0,
                      9, 12, 68, 24, 99, 75, 97, 49, 89, 91, 57, 1, 54, 90, 62, 81, 66,
                      37, 58, 69, 43, 41, 85, 6, 84, 16, 86, 15, 42, 52, 95, 34, 51, 88,
                      36, 29, 40, 22, 5, 79, 70, 14, 31, 83, 18, 47, 7, 32, 26, 13, 93,
                      63, 45, 96, 19, 77, 46, 59, 78, 25, 33, 92, 61, 2, 27, 30, 44, 73,
                      38, 98, 4, 3, 74, 10, 80, 11, 50, 72, 67, 65, 35, 55, 64],
            "worst": [50, 64, 74, 3, 44, 65, 11, 35, 55, 72, 30, 38, 80, 98, 2, 73, 4,
                      10, 27, 67, 26, 33, 45, 46, 59, 77, 13, 19, 78, 93, 47, 61, 63, 83,
                      92, 96, 18, 7, 32, 25, 51, 84, 5, 14, 52, 95, 70, 31, 29, 40, 6,
                      16, 36, 79, 15, 34, 88, 42, 86, 22, 69, 62, 43, 58, 97, 37, 57, 81,
                      91, 99, 66, 75, 85, 89, 90, 1, 24, 41, 49, 54, 87, 8, 56, 71, 0,
                      21, 28, 17, 23, 39, 68, 94, 48, 12, 53, 82, 9, 60, 76, 20],
            "rand1": [36, 35, 52, 91, 87, 48, 17, 31, 51, 40, 3, 18, 16, 74, 68, 23, 50,
                      75, 65, 5, 85, 73, 14, 2, 53, 64, 72, 61, 81, 41, 28, 59, 42, 60,
                      25, 63, 66, 76, 29, 24, 93, 82, 62, 69, 30, 47, 8, 96, 27, 88, 7,
                      34, 4, 6, 83, 1, 58, 67, 71, 92, 57, 49, 33, 95, 56, 9, 19, 80,
                      13, 46, 12, 43, 44, 99, 55, 10, 84, 89, 22, 79, 38, 21, 98, 0, 39,
                      86, 54, 97, 37, 26, 70, 15, 45, 78, 20, 32, 11, 90, 77, 94],
            "rand2": [4, 11, 92, 41, 42, 6, 29, 15, 47, 98, 53, 68, 8, 18, 25, 45, 59,
                      70, 65, 64, 90, 55, 88, 31, 9, 99, 7, 16, 51, 74, 73, 60, 5, 84,
                      97, 85, 1, 49, 40, 34, 14, 83, 48, 19, 17, 79, 50, 54, 2, 96, 36,
                      26, 43, 0, 58, 71, 22, 72, 82, 44, 21, 94, 62, 75, 52, 30, 33, 95,
                      89, 28, 76, 23, 63, 24, 81, 39, 93, 27, 37, 56, 86, 66, 67, 35, 77,
                      3, 32, 69, 61, 13, 80, 10, 78, 20, 12, 46, 87, 38, 57, 91],
            "base1": [30, 74, 77, 91, 67, 43, 81, 99, 45, 15, 16, 51, 49, 83, 39, 48, 29, 55, 47, 1, 23, 6, 8, 94, 56,
                      75, 0, 28, 50, 31, 36, 58, 60, 14, 11, 25, 97, 21, 26, 53, 70, 95, 13, 63, 52, 98, 89, 22, 3, 41,
                      32, 7, 93, 66, 17, 90, 78, 33, 5, 10, 37, 69, 86, 54, 64, 71, 96, 65, 20, 9, 85, 62, 34, 61, 27,
                      46, 2, 24, 57, 88, 79, 72, 84, 82, 38, 92, 59, 4, 73, 18, 80, 12, 19, 44, 68, 40, 87, 42, 76, 35],
            "base2": [25, 42, 11, 22, 88, 30, 34, 96, 44, 37, 4, 35, 62, 54, 48, 46, 26, 84, 28, 63, 17, 94, 74, 29, 40,
                      53, 57, 32, 16, 66, 6, 73, 12, 10, 20, 39, 72, 61, 27, 75, 33, 8, 69, 85, 98, 14, 49, 78, 64, 2,
                      13, 52, 50, 97, 80, 18, 67, 43, 89, 68, 81, 60, 24, 15, 51, 9, 3, 82, 76, 90, 77, 45, 1, 70, 91,
                      99, 55, 71, 7, 41, 65, 93, 92, 36, 83, 58, 31, 21, 79, 87, 38, 0, 86, 56, 59, 5, 19, 95, 23, 47],
            "super": [4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 54, 62, 70, 82, 92, 9, 10, 16, 28, 61, 0, 51, 53, 57, 83,
                      22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 6, 7, 14, 18, 24, 3, 42, 43, 88, 97, 12, 17, 37, 68, 76,
                      23, 33, 49, 60, 71, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 26, 45, 77, 79, 99, 2, 11, 35, 46, 98,
                      27, 29, 44, 78, 93, 36, 50, 65, 74, 80, 47, 52, 56, 59, 96, 8, 13, 48, 58, 90, 41, 69, 81, 85,
                      89],
            "another0": [30, 11, 64, 91, 27, 21, 89, 31, 28, 63, 1, 50, 85, 24, 37, 99, 73,
                         53, 3, 90, 45, 51, 47, 38, 10, 93, 56, 81, 67, 7, 72, 79, 15, 35,
                         82, 40, 76, 96, 23, 71, 32, 86, 46, 65, 88, 14, 98, 54, 44, 34, 62,
                         13, 36, 6, 18, 29, 60, 78, 42, 0, 20, 83, 41, 84, 39, 17, 5, 52,
                         57, 80, 94, 12, 8, 2, 26, 75, 19, 25, 61, 9, 49, 68, 16, 87, 92,
                         4, 59, 58, 70, 43, 77, 97, 55, 95, 48, 22, 74, 69, 33, 66],
            "another1": [16, 24, 70, 74, 26, 87, 73, 22, 4, 75, 27, 42, 21, 37, 9, 84, 90,
                         79, 40, 77, 57, 1, 86, 46, 3, 19, 45, 66, 83, 33, 88, 17, 95, 64,
                         55, 98, 76, 12, 7, 96, 52, 23, 43, 67, 39, 65, 44, 59, 48, 35, 58,
                         41, 93, 49, 6, 5, 68, 32, 18, 51, 20, 85, 2, 31, 80, 91, 82, 97,
                         94, 54, 72, 15, 71, 50, 13, 53, 47, 25, 62, 81, 99, 29, 10, 63, 11,
                         30, 36, 34, 14, 28, 8, 69, 89, 0, 92, 38, 56, 78, 60, 61],
            "another2": [68, 75, 55, 29, 24, 82, 32, 48, 85, 69, 60, 36, 77, 13, 35, 26, 39,
                         10, 81, 15, 45, 27, 65, 67, 30, 4, 51, 28, 18, 52, 59, 79, 86, 34,
                         70, 47, 9, 2, 44, 46, 7, 33, 11, 17, 40, 71, 19, 23, 62, 96, 73,
                         58, 63, 41, 6, 74, 20, 87, 94, 0, 91, 22, 42, 78, 72, 93, 31, 76,
                         56, 12, 37, 21, 61, 66, 84, 38, 89, 57, 25, 1, 16, 53, 88, 54, 64,
                         50, 49, 99, 5, 90, 3, 80, 95, 43, 14, 8, 98, 92, 97, 83],
            "another3": [16, 92, 35, 65, 4, 61, 38, 96, 68, 90, 45, 12, 1, 19, 37, 14, 34,
                         46, 39, 24, 55, 60, 71, 91, 87, 49, 62, 28, 6, 79, 7, 31, 70, 53,
                         72, 13, 88, 11, 95, 58, 73, 80, 75, 40, 27, 21, 67, 48, 99, 5, 52,
                         89, 63, 64, 36, 81, 56, 93, 97, 43, 74, 32, 20, 57, 85, 76, 9, 86,
                         10, 77, 98, 22, 42, 82, 59, 78, 94, 17, 8, 47, 3, 41, 69, 23, 50,
                         44, 18, 84, 30, 54, 66, 83, 0, 26, 2, 15, 25, 33, 29, 51],
            "another4": [37, 97, 15, 35, 81, 69, 41, 94, 7, 20, 60, 46, 55, 79, 14, 67, 34,
                         2, 43, 5, 26, 38, 61, 62, 58, 54, 51, 89, 47, 66, 80, 23, 63, 76,
                         93, 17, 1, 64, 91, 44, 73, 9, 92, 12, 90, 30, 70, 75, 49, 48, 59,
                         18, 86, 88, 83, 42, 33, 6, 24, 36, 68, 77, 11, 13, 98, 31, 56, 29,
                         50, 71, 78, 96, 28, 3, 25, 82, 53, 8, 32, 39, 57, 0, 10, 4, 22,
                         84, 40, 74, 65, 19, 99, 95, 16, 27, 45, 52, 72, 87, 85, 21],
            "another5": [86, 52, 32, 17, 11, 47, 24, 35, 46, 60, 16, 91, 9, 65, 72, 29, 6,
                         95, 66, 89, 85, 50, 14, 94, 68, 61, 27, 55, 2, 22, 74, 48, 13, 19,
                         12, 0, 81, 67, 88, 84, 98, 92, 64, 41, 37, 20, 43, 18, 87, 93, 39,
                         15, 62, 97, 44, 79, 1, 26, 31, 53, 4, 49, 28, 54, 38, 25, 69, 34,
                         77, 56, 63, 75, 3, 80, 23, 51, 96, 78, 21, 73, 71, 59, 45, 33, 42,
                         7, 5, 90, 99, 8, 40, 70, 36, 76, 57, 83, 30, 10, 58, 82],
            "another6": [49, 48, 78, 50, 15, 17, 58, 98, 96, 69, 25, 18, 14, 10, 56, 99, 28,
                         24, 76, 22, 39, 74, 40, 70, 2, 90, 68, 35, 6, 91, 12, 3, 66, 29,
                         87, 54, 92, 94, 34, 9, 97, 36, 81, 61, 45, 8, 37, 62, 79, 65, 57,
                         83, 21, 33, 82, 47, 59, 4, 67, 19, 95, 1, 31, 73, 88, 75, 27, 30,
                         5, 85, 84, 43, 46, 86, 23, 41, 13, 52, 38, 7, 63, 32, 77, 20, 80,
                         53, 11, 71, 0, 93, 16, 51, 72, 55, 44, 26, 64, 89, 42, 60],
            "another7": [73, 86, 18, 97, 24, 74, 88, 41, 7, 1, 65, 60, 49, 15, 26, 39, 17,
                         36, 20, 35, 61, 51, 91, 53, 13, 45, 55, 10, 27, 38, 11, 16, 85, 52,
                         14, 98, 33, 0, 90, 93, 28, 82, 58, 70, 87, 69, 8, 40, 92, 4, 78,
                         32, 25, 22, 96, 67, 21, 63, 19, 81, 71, 75, 47, 9, 6, 43, 37, 56,
                         3, 30, 66, 68, 79, 72, 99, 54, 34, 57, 59, 29, 46, 23, 5, 76, 42,
                         64, 77, 89, 31, 12, 44, 2, 80, 84, 94, 50, 48, 62, 83, 95],
            "another8": [4, 40, 69, 81, 1, 76, 80, 48, 35, 86, 63, 64, 78, 32, 96, 41, 74,
                         8, 20, 59, 55, 21, 9, 29, 28, 26, 17, 66, 6, 42, 77, 7, 16, 36,
                         5, 47, 30, 85, 43, 11, 73, 12, 50, 18, 82, 10, 38, 45, 79, 92, 58,
                         89, 15, 23, 68, 3, 49, 39, 71, 44, 95, 57, 90, 34, 83, 31, 0, 52,
                         53, 61, 98, 33, 25, 84, 65, 99, 67, 54, 88, 13, 97, 2, 94, 62, 27,
                         72, 75, 87, 24, 37, 22, 60, 14, 70, 91, 46, 19, 56, 93, 51],
            "another9": [35, 77, 62, 63, 88, 93, 73, 32, 16, 75, 27, 23, 48, 15, 56, 46, 0,
                         69, 59, 60, 26, 58, 64, 80, 72, 68, 97, 45, 81, 10, 11, 43, 20, 82,
                         49, 53, 83, 95, 61, 18, 84, 9, 89, 30, 40, 86, 44, 31, 25, 91, 52,
                         90, 8, 66, 7, 19, 65, 42, 96, 74, 51, 92, 5, 39, 4, 6, 78, 99,
                         38, 28, 55, 33, 29, 47, 36, 67, 87, 98, 71, 37, 94, 12, 14, 70, 24,
                         1, 13, 2, 57, 17, 34, 22, 54, 76, 41, 3, 79, 85, 50, 21],
        },
        "classes": ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    }
}
