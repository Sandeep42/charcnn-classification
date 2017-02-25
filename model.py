import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CharCNN(nn.Module):
    
    def __init__(self, n_classes, num_chars=70, max_seq_length=1014, kernel_sizes= [7,7,3,3,3,3], 
                 channel_size=256, pool_size=3):
        
        super(CharCNN, self).__init__()
        
        self.num_chars = num_chars
        self.max_seq_length = max_seq_length
        self.n_classes = n_classes
        self.pool_size = pool_size
        self.final_linear_len = (max_seq_length - 96)/ 27
        
        self.conv1 = nn.Conv1d(num_chars, channel_size, kernel_sizes[0])
        self.conv2 = nn.Conv1d(channel_size, channel_size, kernel_sizes[1])
        self.conv3 = nn.Conv1d(channel_size, channel_size, kernel_sizes[2])
        self.conv4 = nn.Conv1d(channel_size, channel_size, kernel_sizes[3])
        self.conv5 = nn.Conv1d(channel_size, channel_size, kernel_sizes[4])
        self.conv6 = nn.Conv1d(channel_size, channel_size, kernel_sizes[5])
        
        self.pool1 = nn.MaxPool1d(pool_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        
        self.FC1 = nn.Linear(channel_size * self.final_linear_len, 1024)
        self.FC2 = nn.Linear(1024, 1024)
        self.final_layer = nn.Linear(1024, n_classes)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.05)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)

        
    def forward(self, x):        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)))
        x = x.view(x.size(0),-1)
        x = self.FC1(x)
        x = self.dropout1(x)
        x = self.FC2(x)
        x = self.dropout2(x)
        x = self.final_layer(x)
        return F.log_softmax(x)

