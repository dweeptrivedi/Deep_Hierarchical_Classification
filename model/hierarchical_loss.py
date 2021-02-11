'''Hierarchical Loss Network
'''

import pickle
import torch
import torch.nn as nn


class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''

    @staticmethod
    def unpickle(file):
        '''Unpickle the given file
        '''

        with open(file, 'rb') as f:
            res = pickle.load(f, encoding='bytes')
        return res

    @staticmethod
    def classes(metafile):
        '''Reads the available classes from the meta file.
        '''
        meta_data = HierarchicalLossNetwork.unpickle(metafile)
        fine_label_names = [t.decode('utf8') for t in meta_data[b'fine_label_names']]
        coarse_label_names = [t.decode('utf8') for t in meta_data[b'coarse_label_names']]

        return coarse_label_names, fine_label_names


    def words_to_indices(self):
        '''Convert the classes from words to indices.
        '''
        numeric_hierarchy = {}
        for k, v in self.hierarchical_labels.items():
            numeric_hierarchy[self.level_one_labels.index(k)] = [self.level_two_labels.index(i) for i in v]

        return numeric_hierarchy


    def __init__(self, metafile_path, hierarchical_labels, device='cpu', total_level=2, alpha=1, beta=1, p_loss=5):
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.level_one_labels, self.level_two_labels = HierarchicalLossNetwork.classes(metafile=metafile_path)
        self.hierarchical_labels = hierarchical_labels
        self.numeric_hierarchy = self.words_to_indices()



    def check_hierarchy(self, current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''

        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [current_level[i] in self.numeric_hierarchy[previous_level[i].item()] for i in range(previous_level.size()[0])]

        return torch.FloatTensor(bool_tensor).to(self.device)



    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        lloss = 0
        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])

        return self.alpha * lloss

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred)

            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], 1.0, 0.)
            l_curr = torch.where(current_lvl_pred == true_labels[l], 1.0, 0.)

            dloss += torch.sum(-1*torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr))

        return self.beta * dloss











