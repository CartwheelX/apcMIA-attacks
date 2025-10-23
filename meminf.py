import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize
import base64
from torchmetrics.classification import BinaryConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
from sklearn.manifold import TSNE
import seaborn as sns
np.set_printoptions(threshold=np.inf)
from target_shadow_nn_models import *
from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from early_stopping_pytorch import EarlyStopping


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class shadow_train_class():
    def __init__(self, trainloader, testloader, dataset_name, model, device, use_DP, noise, norm, delta):
        self.delta = delta
        self.use_DP = use_DP
        self.device = device
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        # set a codition on weight decay to be 5e-3 in case if dataset is purchase
        if dataset_name == 'purchase':
            self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)


        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)

    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            # outputs = self.model(inputs)
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 100.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        # self.model.eval()
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 100.*correct/total
   
class target_train_class():
    def __init__(self, trainloader, testloader, dataset_name, model, device, use_DP, noise, norm, delta, arch, batch_size):
        self.use_DP = use_DP
        self.device = device
        self.delta = delta
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.arch = arch
        self.batch_size = batch_size
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        self.noise_multiplier, self.max_grad_norm = noise, norm

        
        if dataset_name == 'purchase':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)
        elif dataset_name == 'texas':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9,  weight_decay=1e-4)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif dataset_name == 'adult':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9,  weight_decay=1e-4)
            

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        elif dataset_name == 'location':
            # # Tabular MLP: AdamW with a higher LR and a bit more decay
            if ~self.use_DP:
                
                self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=2e-4)

                if self.use_DP:
                    self.privacy_engine = PrivacyEngine()
                    self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                        module=model,
                        optimizer=self.optimizer,
                        data_loader=self.trainloader,
                        noise_multiplier=self.noise_multiplier,
                        max_grad_norm=self.max_grad_norm,
                    )
                    print('noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

                # cosine decay to reduce LR earlier (helps peak generalization sooner)
                self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=12, eta_min=1e-5)

            else:
                # for dp training we needed SGD based optimizer and as well as the model must nust have batch normalization
                self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9,  weight_decay=5e-4)

                if self.use_DP:
                    self.privacy_engine = PrivacyEngine()
                    self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                        module=model,
                        optimizer=self.optimizer,
                        data_loader=self.trainloader,
                        noise_multiplier=self.noise_multiplier,
                        max_grad_norm=self.max_grad_norm,
                    )          
                    print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

                # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
                self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

        else:

            if self.arch == 'vgg16':
                self.optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay = 0.005, momentum = 0.9)
            elif self.arch == 'wrn':
                # optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
                self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum = 0.9)
            elif self.arch == 'wrn_rmia':
                # base_lr = 0.1   # use 0.2 if batch_size = 256
                # print(f"self.batch_size: {self.batch_size}")
                # exit()
                if self.batch_size == 128:
                    base_lr = 0.1
                elif self.batch_size == 256:
                    base_lr = 0.2
                else:
                    base_lr = 0.01 # for anyother datasets with different batch size
                
                self.optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

                # self.optimizer = torch.optim.SGD( model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
                # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
                # 0.1 -> 0.02 -> 0.004 -> 0.0008 at epochs 60, 120, 160
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)
                
               
                # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum = 0.9)
            elif self.arch == 'cnn':
                # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2,  weight_decay=5e-4, momentum=0.9)
                self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, min_lr=0.00001)
            elif self.arch == 'van_cnn':
                # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2,  weight_decay=5e-4, momentum=0.9)
                self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2,  weight_decay=5e-4, momentum=0.9)
                self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)
                # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, min_lr=0.00001)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
   
    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            # print(f"inputs size: {inputs.size()}, targets size: {targets.size()}")
            # exit()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            # gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            # self.scheduler.step()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))

            

        

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))
        
        if self.arch == 'cnn': # As per https://medium.com/@anderaquerretamontoro/the-best-cnn-for-cifar10-from-scratch-93-accuracy-bde35e17fca6
            # lr_scheduler.step(test_loss)
            acc_val, val_loss = self.test()
            self.scheduler.step(val_loss)
        else:
            acc_val, val_loss = self.test()
            self.scheduler.step()

        return 1.*correct/total, acc_val
    # Testing
    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total, test_loss

    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

   
def get_ent_lr(acc_gap, max_lr=0.005, k=10, mid=0.5):
    return max_lr * (1 - 1 / (1 + np.exp(-k * (acc_gap - mid))))

def get_cs_lr(acc_gap, max_lr=0.01, k=10, mid=0.5):
    return max_lr * (1 - 1 / (1 + np.exp(-k * (acc_gap - mid))))

def sigmoid_adaptive_lr(gap, mid_gap=0.475, gap_range=0.55, max_lr=0.1, min_lr=0.001):
    """
    Adaptive sigmoid-based learning rate scheduler.

    Parameters:
    - gap (float): Accuracy gap (0 to 1 scale)
    - mid_gap (float): Center point of the sigmoid (e.g., 0.475 for 47.5%)
    - gap_range (float): Controls steepness (difference between upper and lower bounds; smaller = steeper)
    - max_lr (float): Maximum learning rate
    - min_lr (float): Minimum learning rate

    Returns:
    - lr (float): Adapted learning rate
    """
    k = 10 / gap_range  # Steepness from gap_range (e.g., 0.55 gives k ≈ 18.18)
    sigmoid = 1 / (1 + np.exp(-k * (gap - mid_gap)))
    return min_lr + (max_lr - min_lr) * sigmoid

def sigmoid_adaptive_lr(gap, mid_gap=0.45, gap_range=0.65, max_lr=0.1, min_lr=0.003):
    k = 10 / gap_range  # Controls steepness
    sigmoid = 1 / (1 + np.exp(-k * (gap - mid_gap)))
    return min_lr + (max_lr - min_lr) * sigmoid

class attack_for_blackbox_com_NEW():
    def __init__(self,TARGET_PATH,SHADOW_PATH, Perturb_MODELS_PATH, ATTACK_SETS,ATTACK_SETS_PV_CSV, attack_train_loader, 
                 attack_test_loader, target_model, shadow_model, attack_model, perturb_model, device, dataset_name, attack_name, 
                 num_classes, attack_dataset_batch_size, acc_gap, arch='mlp', membership_flip_prob=0.1):
        # acc_gap, arch, flip_prob
        self.device = device
        self.batch_size = attack_dataset_batch_size # this is attack dataset batch size, can be similar to target model batch size pass via constructor
        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS
        self.ATTACK_SETS_PV_CSV = ATTACK_SETS_PV_CSV
        
        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)
        self.Perturb_MODELS_PATH = Perturb_MODELS_PATH
        self.attack_name = attack_name
        print( 'self.TARGET_PATH: %s' % self.TARGET_PATH)
        # exit()
        # self.pred_component_model = pred_component_model
        self.num_classes = num_classes
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH, weights_only=True))
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH, weights_only=True))
        self.arch = arch
        self.target_model.eval()
        # self.shadow_model.eval()
        self.member_mean = 0.0
        self.member_std = 0.0
        self.non_member_mean = 0.0
        self.non_member_std  = 0.0

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        self.perturb_model = perturb_model.to(self.device)
        self.patience = 20
        self.early_stopping = EarlyStopping(self.patience, verbose=True)
        # torch.manual_seed(0)
        self.attack_model.apply(weights_init)
        self.perturb_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        # fff
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)
        self.optimizer_perturb = optim.Adam(self.perturb_model.parameters(), lr=1e-3) # need to change the learning rate to see the effect latter
        
        

        self.dataset_name = dataset_name

        self.membership_flip_prob = max(0.0, min(1.0, float(membership_flip_prob)))
    
      
       
        dataset_k_values = {
            "stl10": 1000.0,
            "cifar100": 1000.0,
            # "location": 1000.0,
            "fmnist": 1000000.0,
            "purchase": 1000000.0,
            "cifar10": 10000.0,
            "utkface": 1000000,
            "texas": 10000,
            "adult": 1000000,
        }

        self.k = dataset_k_values.get(dataset_name)  # Default to 1000000 if dataset_name is not found
    
        if dataset_name == "fmnist" or dataset_name == "cifar10" or dataset_name == "utkface":
            self.k1 = 1000000.0
            # self.k1 = 10.0

        else:
            self.k1 = 10.0

        # NON-image datasets
        if dataset_name == "location":
            self.k = 1000.0
            self.k1 = 10000.0   
        if dataset_name == "texas":
            self.k = 10000.0
            self.k1 = 10.0
        if dataset_name == "adult":
            self.k = 1000000.0
            self.k1 = 100.0
        if dataset_name == "purchase":
            self.k = 1000000.0
            self.k1 = 100.0 
        

        # Image datasets
        if dataset_name == "stl10":
            self.k = 1000.0
            self.k1 = 10.0
        elif dataset_name == "cifar100":
            self.k = 1000.0
            self.k1 = 10.0
        elif dataset_name == "cifar10":
            self.k = 10000.0
            self.k1 = 1000000.0 
        elif dataset_name == "utkface":
            self.k = 1000000.0
            self.k1 = 1000000.0
        elif dataset_name == "fmnist":
            self.k = 1000000.0
            self.k1 = 1000000.0
        
    
    
        # Note: The following has been used as fixed optimized rates to generate results in the paper
        if dataset_name == "cifar100":
            cs_lr = 0.01
            ent_lr = 0.1
        elif dataset_name == "cifar10":
            cs_lr = 0.01
            ent_lr = 0.001
        elif dataset_name == "fmnist":
            cs_lr = 0.01
            ent_lr = 0.001
        elif dataset_name == "utkface":
            cs_lr = 0.01
            ent_lr = 0.001
        elif dataset_name == "purchase":
            cs_lr = 0.01 # was 0.001 both
            ent_lr = 0.01
        elif dataset_name == "location":
            cs_lr = 0.01
            ent_lr = 0.01
        elif dataset_name == "adult":
            cs_lr = 0.01
            ent_lr = 0.01 # was 0.1
        elif dataset_name == "texas":
            cs_lr = 0.001
            ent_lr = 0.001
        else: # stl10
            cs_lr = 0.01
            ent_lr = 0.001
        

        self.cosine_threshold = nn.Parameter(torch.tensor(0.5, device=self.device))
        self.Entropy_quantile_threshold = nn.Parameter(torch.tensor(0.5, device=self.device))


        self.optimizer_cosine = optim.Adam([self.cosine_threshold], cs_lr)
        self.optimizer_quantile_threshold = optim.Adam([self.Entropy_quantile_threshold], lr=ent_lr)

        self.kl_threshold = torch.nn.Parameter(torch.tensor(0.5))
        kl_lr = 0.01
        self.optimizer_kl = torch.optim.Adam([self.kl_threshold], kl_lr)

     
    
    def _maybe_flip_members(self, members):
        """Optionally flip membership indicators before using them as guidance signals."""
        if self.membership_flip_prob <= 0:
            return members

        if not torch.is_tensor(members):
            raise TypeError("members must be a torch.Tensor")

        flip_mask = torch.rand(members.shape, device=members.device) < self.membership_flip_prob
        flipped = members.clone()

        flipped[flip_mask] = 1 - flipped[flip_mask]
        return flipped.type_as(members)

    def _membership_masks(self, members):
        """Return optionally flipped memberships along with member/non-member masks."""
        noisy_members = self._maybe_flip_members(members)
        return noisy_members, noisy_members == 1, noisy_members == 0

    def _get_data(self, model, inputs, targets):
        
        result = model(inputs)
        # output, _ = torch.sort(result, descending=True) 
        # results = F.softmax(results[:,:5], dim=1)
        output = F.softmax(result, dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction.unsqueeze(-1)

    def prepare_dataset_analyse(self):
        print("Preparing  and analysing the dataset")
        
        
        # Save train dataset to CSV
        with open(self.ATTACK_SETS_PV_CSV, "w", newline='') as f:
            writer = csv.writer(f)
            # Write the header row (optional)
            # Write the header row (optional, adjust depending on the number of output dimensions)
            num_output_classes = 10  # Assuming output size is [batch_size, num_classes]
            header = ["Output_" + str(i) for i in range(num_output_classes)] + ["Prediction", "Members", "Targets"]
            writer.writerow(header)
            
            # writer.writerow(["Output", "Prediction", "Members", "Targets"])
            
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if 1:
                    output, prediction = self._get_data(self.shadow_model, inputs, targets)

                    # Write each batch as rows in the CSV file
                    for i in range(output.shape[0]):
                        # Unpack the output (PV) to individual elements
                        row = output[i].cpu().tolist() + [  # Unpacking the prediction vector (each element becomes a column)
                            prediction[i].item(),           # Correct or wrong (0/1)
                            members[i].item(),              # Membership (0/1)
                            targets[i].item()               # Target label
                        ]
                        writer.writerow(row)

        print("Finished Saving Train Dataset")

    
    def prepare_dataset(self):
        print("Preparing dataset")
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # if inputs.size()[0] == 64:
                
                # change the self.shadow_model to self.target_model to get the PVs assuming shadow model performance is the same as the target model
                output, prediction = self._get_data(self.target_model, inputs, targets)
                
                pickle.dump((output, prediction, members, targets), f)
               
        print("Finished Saving Train Dataset")
    

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                
                output, prediction = self._get_data(self.shadow_model, inputs, targets)
               
            
                pickle.dump((output, prediction, members, targets), f)
                
        
        self.dataset = AttackDataset(self.ATTACK_SETS + "train.p")

        print("Finished Saving Test Dataset")
        return self.dataset
        # exit()

    def prepare_dataset_new(self):
        import os
        os.makedirs(self.ATTACK_SETS, exist_ok=True)

        def dump_split(loader, out_path):
            print(f"Dumping to {out_path}")
            outputs_list, preds_list, members_list, targets_list = [], [], [], []
            self.target_model.eval()  # or self.shadow_model for shadow PVs

            with torch.no_grad():
                for inputs, targets, members in loader:
                    inputs  = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # get posterior vectors + “prediction” from the model
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # ensure tensors, detach, and move to cpu
                    if isinstance(output, torch.Tensor):
                        output = output.detach().cpu()
                    if isinstance(prediction, torch.Tensor):
                        prediction = prediction.detach().cpu()

                    if isinstance(members, torch.Tensor):
                        members = members.detach().cpu()
                    if isinstance(targets, torch.Tensor):
                        targets = targets.detach().cpu()

                    outputs_list.append(output)        # [B, C]
                    preds_list.append(prediction)      # [B, 1] or [B]
                    members_list.append(members)       # [B]
                    targets_list.append(targets)       # [B]

            # Concatenate along batch dimension
            outputs  = torch.cat(outputs_list,  dim=0)
            preds    = torch.cat(preds_list,    dim=0)
            members  = torch.cat(members_list,  dim=0)
            targets  = torch.cat(targets_list,  dim=0)

            # Save once, atomically
            torch.save(
                {
                    "posteriors": outputs.float(),     # [N, C]
                    "pred": preds,                     # [N] or [N,1]
                    "members": members.to(torch.int64),
                    "targets": targets.to(torch.int64),
                    "meta": {
                        "class_num": outputs.shape[-1],
                        "count": outputs.shape[0],
                    },
                },
                out_path,
            )

        print("Preparing dataset")
        dump_split(self.attack_train_loader, self.ATTACK_SETS + "train.pt")
        print("Finished Saving Train attack Dataset")
        dump_split(self.attack_test_loader,  self.ATTACK_SETS + "test.pt")
        print("Finished Saving Test attack Dataset")

    def bind_cached_attack_loaders(self):
        num_workers=4
        batch_size=self.batch_size
        """Build DataLoaders from ATTACK_SETS/train.pt and test.pt and attach them to self."""
        def _load_split(path):
            # path = split
            blob = torch.load(path, map_location="cpu", weights_only=True)
            post = blob["posteriors"].float()     # [N, C]
            pred = blob["pred"]                   # [N] or [N,1]
            memb = blob["members"].long()         # [N]
            targ = blob["targets"].long()         # [N]
            if pred.dim() == 2 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)           # -> [N]
            ds = TensorDataset(post, pred, targ, memb)  # (output, prediction, targets, members)
            return ds

        self._cached_train_ds = _load_split(self.ATTACK_SETS + "train.pt")
        self._cached_test_ds  = _load_split(self.ATTACK_SETS + "test.pt")

        self.attack_cached_train_loader = DataLoader(
            self._cached_train_ds, batch_size=batch_size, shuffle=False,
            drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
        )
        self.attack_cached_test_loader = DataLoader(
            self._cached_test_ds, batch_size=batch_size, shuffle=False,
            drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)
        )
        print("Bound cached attack loaders worked.")
        

        
    def prepare_dataset_mul(self, num_classes):

        batch_size = 8
        # read the whole attack_train_loader batche by batch 
        # put the samples into corresponding buckets
        
        # transform each k-bucket into batches using dataloader
        # loade from dataloader, get the corresponing predictions
        # save into train_i.p file
        
    
        #! Traing Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                # class_name = 29
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_train_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
                with open(file_path, "wb") as f:
                    # This loop will iterate over all batches and put samples in their corresponding class file
                    for inputs, targets, members in self.attack_train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        if 32 == 32:
                            counter+=1
                            # Assuming _get_data returns output and prediction based on the class
                            output, prediction = self._get_data(self.shadow_model, inputs, targets)
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            # print(f"targets: {targets}\n class_name: {class_name}")
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices in the batch: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            
                            
                        
                            
                        else:
                            print("skipping: ", inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    # exit()
                    # print(f"counter: {counter}")
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_train size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_trainloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                     
            #     exit()
            # exit()
        print(f"Finished Saving {num_classes} Train Dataset")   
            
            
        
        #! Test Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_test_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
            
                with open(file_path, "wb") as f:
                    for inputs, targets, members in self.attack_test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        if 32 == 32:
                            counter+=1
                            output, prediction = self._get_data(self.target_model, inputs, targets)
                            # output = output.cpu().detach().numpy()
                            
                            
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            # pickle.dump((output, prediction, members), f)
                        else:
                            print("test data skipping: ",inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information Training")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_test size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_testloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_testloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                        
                
         
        print(f"Finished Saving {num_classes} Test Dataset")   
        # exit()


    def contrastive_loss(self, embeddings, labels, margin):
        """
        A simple contrastive loss function.
        For a pair of embeddings, if they belong to the same class (both member or both non-member),
        the target is 1 (and we penalize a large distance);
        if they belong to different classes, the target is -1 (and we penalize similarity if too high).
        
        This implementation computes the loss over all pairs in the batch.
        
        Args:
            embeddings: Tensor of shape (N, D) where N is batch size and D is embedding dimension.
            labels: Binary labels of shape (N,) indicating membership (1 for member, 0 for non-member).
            margin: Margin for dissimilar pairs.
        
        Returns:
            A scalar contrastive loss value.
        """
        # Normalize embeddings to unit vectors
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarity matrix
        cosine_sim = torch.matmul(embeddings, embeddings.t())  # shape: (N, N)
        
        # Create label matrix: 1 if same class, -1 if different
        labels = labels.unsqueeze(1)  # shape (N, 1)
        label_matrix = (labels == labels.t()).float()
        # Convert same/different to targets of 1 and -1
        target = label_matrix * 2 - 1  # 1 for same, -1 for different
        
        # For same pairs, we want cosine similarity to be close to 1,
        # for different pairs, we want similarity to be at most a margin.
        # One simple loss is mean squared error from target:
        loss_same = F.mse_loss(cosine_sim * (target==1).float(), torch.ones_like(cosine_sim) * (target==1).float())
        # For different pairs, we want the cosine similarity to be less than some margin.
        # If similarity > margin, we incur a loss.
        diff_sim = cosine_sim * (target==-1).float()
        loss_diff = F.relu(diff_sim - margin).pow(2).mean()
        
        return loss_same + loss_diff

   
    def train(self, epoch, result_path, result_path_csv, mode):
        # models to train
        self.attack_model.train()
        if self.attack_name == "apcmia":
            self.perturb_model.train()

        # ---- choose loader (must be bound earlier via bind_cached_attack_loaders) ----
        if not hasattr(self, "attack_cached_train_loader"):
            raise RuntimeError(
                "attack_cached_train_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before train()."
            )
        loader = self.attack_cached_train_loader

        # ---- bookkeeping ----
        batch_idx = 0
        train_loss = 0.0
        correct = 0
        prec = 0.0
        recall = 0.0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth, final_train_predict, final_train_probabe = [], [], []

        lambda_contrast = 0.5  # weight for contrastive term

        # ---- training loop over cached batches: (output, pred, targets, members) ----
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("Expected batch=(output, pred, targets, members).")
            output, prediction, targets, members = batch

            # move to GPU
            output     = output.to(self.device, non_blocking=True)    # [B, C]
            prediction = prediction.to(self.device, non_blocking=True) # [B] or [B,1]
            targets    = targets.to(self.device, non_blocking=True)    # [B]
            members    = members.to(self.device, non_blocking=True)    # [B]

            # shape expected by your attack model
            pred_for_model = prediction.unsqueeze(-1) if prediction.dim() == 1 else prediction

            if self.attack_name == "apcmia":
                # ----- perturbation block (unchanged) -----
                noisy_members, member_mask, non_member_mask = self._membership_masks(members)
                if member_mask.any() and non_member_mask.any():
                    member_indices = member_mask.nonzero(as_tuple=True)[0]
                    non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                    member_pvs = output[member_indices]
                    non_member_pvs = output[non_member_indices]

                    # overlap via cosine similarity
                    cos_sim = F.cosine_similarity(
                        non_member_pvs.unsqueeze(1), member_pvs.unsqueeze(0), dim=2
                    )
                    max_cos_sim, _ = cos_sim.max(dim=1)

                    tau = 0.5
                    cosine_threshold = torch.sigmoid(self.cosine_threshold)
                    logits = (max_cos_sim - cosine_threshold) * self.k1
                    binary_logits = torch.stack([-logits, logits], dim=1)
                    gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                    binary_selection = gumbel_selection[:, 1]

                    alpha = 1.0
                    epsilon = 1e-10
                    tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * \
                                        self.perturb_model(non_member_pvs, targets[non_member_indices])
                    entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                    quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                    quantile_val = torch.quantile(entropy, quantile_threshold)
                    entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                    final_selection = binary_selection * entropy_mask

                    perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * \
                                            self.perturb_model(non_member_pvs, targets[non_member_indices])
                    perturbed_pvs = output.clone()
                    perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                else:
                    perturbed_pvs = output.clone()

                # re-normalize posteriors
                perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                # forward
                results = self.attack_model(perturbed_pvs, pred_for_model, targets)

                # contrastive loss (optional)
                margin = 0.5
                embeddings = self.attack_model.get_embeddings(perturbed_pvs, pred_for_model, targets)
                contrast_loss = self.contrastive_loss(embeddings, members, margin)
            else:
                results = self.attack_model(output, pred_for_model, targets)
                contrast_loss = 0.0

            # main loss + (optional) contrastive
            attack_loss = self.criterion(results, members)
            total_loss = attack_loss + (lambda_contrast * contrast_loss if isinstance(contrast_loss, torch.Tensor) else 0.0)

            # optim
            self.optimizer.zero_grad(set_to_none=True)
            if self.attack_name == "apcmia":
                self.optimizer_perturb.zero_grad(set_to_none=True)
                self.optimizer_cosine.zero_grad(set_to_none=True)
                self.optimizer_quantile_threshold.zero_grad(set_to_none=True)

            total_loss.backward()
            self.optimizer.step()
            if self.attack_name == "apcmia":
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

            # metrics
            train_loss += attack_loss.item()
            _, predicted = results.max(1)
            total += members.size(0)
            correct += predicted.eq(members).sum().item()

            conf_mat = bcm(predicted, members)
            prec   += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
            recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

            final_train_gndtrth.append(members.detach().cpu())
            final_train_predict.append(predicted.detach().cpu())
            final_train_probabe.append(results[:, 1].detach().cpu())

            batch_idx += 1

        # ---- end epoch stats ----
        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        print(
            f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | "
            f"Loss: {train_loss/max(batch_idx,1):.3f} "
            f"Precision: {100.*prec/max(batch_idx,1):.3f} "
            f"Recall: {100.*recall/max(batch_idx,1):.3f}"
        )

        ent_thre = torch.sigmoid(self.Entropy_quantile_threshold)
        cos_thre = torch.sigmoid(self.cosine_threshold)
        if self.attack_name == "apcmia":
            print(f"CosineT: {cos_thre:.4f}, Quantile Threshold: {ent_thre:.4f}")

        return cos_thre, ent_thre



    def test(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        if self.attack_name == "apcmia":
            self.perturb_model.eval()

        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before test()."
            )
        loader = self.attack_cached_test_loader

        batch_idx = 0
        correct = 0
        total = 0
        prec = 0.0
        recall = 0.0
        total_test_loss = 0.0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_test_gndtrth, final_test_predict, final_test_probabe = [], [], []

        with torch.no_grad():
            for batch in loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output     = output.to(self.device, non_blocking=True)     # [B, C]
                prediction = prediction.to(self.device, non_blocking=True)  # [B] or [B,1]
                targets    = targets.to(self.device, non_blocking=True)     # [B]
                members    = members.to(self.device, non_blocking=True)     # [B]

                pred_for_model = prediction.unsqueeze(-1) if prediction.dim() == 1 else prediction

                # ----- apcmia perturbation path (same as train) -----
                if self.attack_name == "apcmia":
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                    if member_mask.any() and non_member_mask.any():
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]

                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1), member_pvs.unsqueeze(0), dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)

                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        alpha = 1.0
                        epsilon = 1e-10
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])

                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                        entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)

                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # normalize posteriors
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    logits_attack = self.attack_model(perturbed_pvs, pred_for_model, targets)   # raw logits
                else:
                    logits_attack = self.attack_model(output, pred_for_model, targets)          # raw logits

                # ---- loss on logits (CrossEntropy expects logits, not probabilities) ----
                loss = self.criterion(logits_attack, members)
                total_test_loss += loss.item()

                # ---- predictions & metrics ----
                probs = F.softmax(logits_attack, dim=1)           # for AUC/ROC
                _, predicted = probs.max(dim=1)

                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec   += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_test_gndtrth.append(members.detach().cpu())
                final_test_predict.append(predicted.detach().cpu())
                final_test_probabe.append(probs[:, 1].detach().cpu())

                batch_idx += 1

        # ----- aggregate -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / max(batch_idx, 1)

        final_result = [
            correct / max(total, 1),
            (prec / max(batch_idx, 1)).item(),
            (recall / max(batch_idx, 1)).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss,
        ]

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(
            f"Test Acc: {100.0 * correct / max(1.0 * total, 1.0):.3f}% "
            f"({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"precision: {100.0 * prec / max(batch_idx, 1):.3f}, "
            f"recall: {100.0 * recall / max(batch_idx, 1):.3f}"
        )

        return final_result, fpr, tpr


    
    def train_KL(self, epoch, result_path, result_path_csv, mode):
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for the contrastive loss term

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)

                if self.attack_name == "apcmia":
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]

                        # KL divergence similarity
                        epsilon = 1e-8
                        p = non_member_pvs.unsqueeze(1).clamp(min=epsilon, max=1)
                        q = member_pvs.unsqueeze(0).clamp(min=epsilon, max=1)

                        kl_div = (p * (p / q).log()).sum(dim=2)
                        min_kl_div, _ = kl_div.min(dim=1)

                        tau = 0.5
                        kl_threshold = torch.sigmoid(self.kl_threshold)
                        logits = (kl_threshold - min_kl_div) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        alpha = 1
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * \
                                            self.perturb_model(non_member_pvs, targets[non_member_indices])
                        entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * \
                                                self.perturb_model(non_member_pvs, targets[non_member_indices])

                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    results = self.attack_model(perturbed_pvs, prediction, targets)

                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0

                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss

                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_kl.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_kl.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"KL Threshold: {self.kl_threshold.item():.4f}, Quantile Threshold: {self.Entropy_quantile_threshold.item():.4f}")

        return self.kl_threshold.item(), self.Entropy_quantile_threshold.item()
    

    def train_ecld(self, epoch, result_path, result_path_csv, mode): 
        
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for the contrastive loss term; adjust as necessary

        # Open your training data file
        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)
                # targets can remain on CPU or be moved if needed

                if self.attack_name == "apcmia":
                    # --- Perturbation using negative Euclidean distance instead of cosine similarity ---
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]       # (n_members, C)
                        non_member_pvs = output[non_member_indices] # (n_non_members, C)

                        # Compute pairwise Euclidean distances: shape (n_non_members, n_members)
                        distances = torch.cdist(non_member_pvs, member_pvs, p=2)
                        # Convert distances to similarity scores (smaller distance -> higher similarity)
                        euclid_sim = -distances
                        # For each non-member, select maximum similarity (i.e. the most similar member)
                        max_sim, _ = euclid_sim.max(dim=1)  # (n_non_members,)

                        tau = 0.5
                        # Use a learnable threshold (same parameter reused; can be renamed)
                        similarity_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_sim - similarity_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]
                        
                        alpha = 1

                        # --- Entropy-Based Weighting ---
                        epsilon = 1e-10
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * \
                                                self.perturb_model(non_member_pvs, targets[non_member_indices])
                        entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * \
                                                    self.perturb_model(non_member_pvs, targets[non_member_indices])
                        
                        # Replace non-member PVs in output with perturbed values.
                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize the perturbed probability vectors:
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    # Forward pass through the attack model:
                    results = self.attack_model(perturbed_pvs, prediction, targets) # Using original output (or you can try perturbed_pvs)

                    # --- Contrastive Loss ---
                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)
                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0  # no contrastive loss for non-apcmia attacks

                # Compute primary loss (attack loss)
                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss 

                # Backpropagation
                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)

                # print(f"Grad of cosine_threshold: {self.cosine_threshold.grad}")

                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                # print(f"Grad of cosine_threshold: {self.cosine_threshold.grad}")

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        # Post Epoch Evaluation
        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"Cosine Threshold: {torch.sigmoid(self.cosine_threshold).item():.4f}, Quantile Threshold: {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")
        # print(f"Cosine Threshold (sigmoid): {torch.sigmoid(self.cosine_threshold).item():.4f}")
        # print(f"Quantile Threshold (sigmoid): {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")

        return self.cosine_threshold.item(), self.Entropy_quantile_threshold.item()


    def test_KL(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # ----- KL Divergence Similarity -----
                            epsilon = 1e-8
                            p = non_member_pvs.unsqueeze(1).clamp(min=epsilon, max=1)
                            q = member_pvs.unsqueeze(0).clamp(min=epsilon, max=1)

                            kl_div = (p * (p / q).log()).sum(dim=2)  # shape: [n_non_members, n_members]
                            min_kl_div, _ = kl_div.min(dim=1)

                            # ----- Gumbel-Softmax Selection -----
                            tau = 0.5
                            kl_threshold = torch.sigmoid(self.kl_threshold)
                            logits = (kl_threshold - min_kl_div) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            # ----- Entropy Filtering & Perturbation -----
                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        # ----- Normalize & Clip -----
                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post-Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx
        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr

    def test_ecld(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # ----- Euclidean Similarity -----
                            distances = torch.cdist(non_member_pvs, member_pvs, p=2)
                            euclid_sim = -distances
                            max_sim, _ = euclid_sim.max(dim=1)

                            tau = 0.5
                            similarity_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_sim - similarity_threshold) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post-Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx

        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr


    def train_pearson(self, epoch, result_path, result_path_csv, mode): 
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for contrastive loss

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)

                if self.attack_name == "apcmia":
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]         # (m, C)
                        non_member_pvs = output[non_member_indices] # (n, C)

                        # Normalize PVs: subtract mean
                        member_mean = member_pvs.mean(dim=1, keepdim=True)
                        member_std = member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                        member_norm = (member_pvs - member_mean) / member_std  # (m, C)

                        non_member_mean = non_member_pvs.mean(dim=1, keepdim=True)
                        non_member_std = non_member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                        non_member_norm = (non_member_pvs - non_member_mean) / non_member_std  # (n, C)

                        # Compute Pearson correlation (dot product after normalization)
                        pearson_corr = torch.matmul(non_member_norm, member_norm.T) / non_member_norm.size(1)  # (n, m)
                        max_corr, _ = pearson_corr.max(dim=1)  # Most similar member for each non-member

                        # Use threshold and gumbel-softmax to determine which non-members to perturb
                        tau = 0.5
                        pearson_threshold = torch.sigmoid(self.cosine_threshold)  # reuse threshold param
                        logits = (max_corr - pearson_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]  # (n,)

                        # Apply perturbation
                        alpha = 1
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        # Entropy-based filtering
                        epsilon = 1e-10
                        entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize & forward pass
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    results = self.attack_model(perturbed_pvs, prediction, targets)

                    # Contrastive Loss
                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0

                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss

                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"Pearson Threshold (used): {torch.sigmoid(self.cosine_threshold).item():.4f}, "
            f"Quantile Threshold: {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")

        return self.cosine_threshold.item(), self.Entropy_quantile_threshold.item()
    
    def test_pearson(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # Pearson similarity calculation
                            member_mean = member_pvs.mean(dim=1, keepdim=True)
                            member_std = member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                            member_norm = (member_pvs - member_mean) / member_std

                            non_member_mean = non_member_pvs.mean(dim=1, keepdim=True)
                            non_member_std = non_member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                            non_member_norm = (non_member_pvs - non_member_mean) / non_member_std

                            pearson_corr = torch.matmul(non_member_norm, member_norm.T) / non_member_norm.size(1)
                            max_corr, _ = pearson_corr.max(dim=1)

                            # Gumbel-softmax selection
                            tau = 0.5
                            pearson_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_corr - pearson_threshold) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            # Entropy-filtered perturbation
                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx

        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/total:.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"Precision: {100.*prec/batch_idx:.3f}, Recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr


    def train_mahalanobis(self, epoch, result_path, result_path_csv, mode): 
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for contrastive loss

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)

                if self.attack_name == "apcmia":
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]         # (m, C)
                        non_member_pvs = output[non_member_indices] # (n, C)

                        # --- Compute Covariance Matrix for Member PVs ---
                        member_mean = member_pvs.mean(dim=0, keepdim=True)  # (1, C)
                        centered_member = member_pvs - member_mean
                        cov = centered_member.T @ centered_member / (member_pvs.size(0) - 1)  # (C, C)

                        # Invert covariance matrix (or use pseudo-inverse if singular)
                        try:
                            cov_inv = torch.inverse(cov)
                        except RuntimeError:
                            cov_inv = torch.pinverse(cov)

                        # --- Mahalanobis Distance: Each non-member to each member ---
                        delta = non_member_pvs.unsqueeze(1) - member_pvs.unsqueeze(0)  # (n, m, C)
                        dists = torch.einsum("nmc,cd,nmd->nm", delta, cov_inv, delta)  # (n, m)
                        mahala_sim = -dists  # Negative = similarity

                        max_sim, _ = mahala_sim.max(dim=1)  # (n,)

                        # --- Gumbel Selection ---
                        tau = 0.5
                        similarity_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_sim - similarity_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        # --- Perturbation + Entropy Mask ---
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        epsilon = 1e-10
                        entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize perturbed PVs
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    results = self.attack_model(perturbed_pvs, prediction, targets)

                    # Contrastive Loss
                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0

                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss

                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"Mahalanobis Threshold (used): {torch.sigmoid(self.cosine_threshold).item():.4f}, "
            f"Quantile Threshold: {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")

        return self.cosine_threshold.item(), self.Entropy_quantile_threshold.item()
    

    def test_mahalanobis(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # Compute covariance matrix from member PVs
                            member_mean = member_pvs.mean(dim=0, keepdim=True)
                            centered_member = member_pvs - member_mean
                            cov = centered_member.T @ centered_member / (member_pvs.size(0) - 1)

                            try:
                                cov_inv = torch.inverse(cov)
                            except RuntimeError:
                                cov_inv = torch.pinverse(cov)

                            # Compute Mahalanobis distance
                            delta = non_member_pvs.unsqueeze(1) - member_pvs.unsqueeze(0)  # (n, m, C)
                            dists = torch.einsum("nmc,cd,nmd->nm", delta, cov_inv, delta)  # (n, m)
                            mahala_sim = -dists  # Use as similarity

                            max_sim, _ = mahala_sim.max(dim=1)

                            # Gumbel-softmax based binary selection
                            tau = 0.5
                            similarity_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_sim - similarity_threshold) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx

        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/total:.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"Precision: {100.*prec/batch_idx:.3f}, Recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr


    def test_saved_model_apcmia(self, atk_model, prt_model, consin_thr, entrp_thr):
        # Bind external models & thresholds
        self.attack_model  = atk_model.to(self.device)
        self.perturb_model = prt_model.to(self.device)
        # store as tensors; we still use sigmoid() at call sites
        self.cosine_threshold           = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        self.attack_model.eval()
        self.perturb_model.eval()

        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before test_saved_model_apcmia()."
            )

        batch_idx = 0
        correct = 0
        total = 0
        prec = 0.0
        recall = 0.0
        total_test_loss = 0.0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_test_gndtrth, final_test_predict, final_test_probabe = [], [], []

        with torch.no_grad():
            for batch in self.attack_cached_test_loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output     = output.to(self.device, non_blocking=True)      # [B,C]
                prediction = prediction.to(self.device, non_blocking=True)   # [B] or [B,1]
                targets    = targets.to(self.device, non_blocking=True)      # [B]
                members    = members.to(self.device, non_blocking=True)      # [B]

                pred_for_model = prediction.unsqueeze(-1) if prediction.dim() == 1 else prediction

                # ----- apcmia perturbation path -----
                noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                if member_mask.any() and non_member_mask.any():
                    member_indices = member_mask.nonzero(as_tuple=True)[0]
                    non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                    member_pvs = output[member_indices]
                    non_member_pvs = output[non_member_indices]

                    # Step 2: overlap via cosine sim
                    cos_sim = F.cosine_similarity(
                        non_member_pvs.unsqueeze(1), member_pvs.unsqueeze(0), dim=2
                    )
                    max_cos_sim, _ = cos_sim.max(dim=1)

                    tau = 0.5
                    cosine_threshold = torch.sigmoid(self.cosine_threshold)   # (0,1)
                    logits = (max_cos_sim - cosine_threshold) * self.k1
                    binary_logits = torch.stack([-logits, logits], dim=1)
                    gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                    binary_selection = gumbel_selection[:, 1]  # [n_non_members]

                    alpha = 1.0
                    epsilon = 1e-10

                    # Step 4: perturb + entropy filter
                    learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                    tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                    entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)

                    quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)  # (0,1)
                    quantile_val = torch.quantile(entropy, quantile_threshold)
                    entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                    final_selection = binary_selection * entropy_mask

                    perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                    perturbed_pvs = output.clone()
                    perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                else:
                    perturbed_pvs = output.clone()

                # normalize posteriors
                perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                # Forward (raw logits). Attack model signature: (output, prediction, targets)
                logits_attack = self.attack_model(perturbed_pvs, pred_for_model, targets)

                # Loss on logits (not softmaxed)
                loss = self.criterion(logits_attack, members)
                total_test_loss += loss.item()

                # Probs for metrics
                probs = F.softmax(logits_attack, dim=1)
                _, predicted = probs.max(dim=1)

                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec   += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_test_gndtrth.append(members.detach().cpu())
                final_test_predict.append(predicted.detach().cpu())
                final_test_probabe.append(probs[:, 1].detach().cpu())

                batch_idx += 1

        # ----- aggregate -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        avg_test_loss = total_test_loss / max(batch_idx, 1)

        final_result = [
            correct / max(total, 1),
            (prec / max(batch_idx, 1)).item(),
            (recall / max(batch_idx, 1)).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ]

        print(
            f"Final: Test Acc: {100.0*correct/max(1,total):.3f}% ({correct}/{total}), "
            f"Loss: {avg_test_loss:.3f}, precision: {100.0*prec/max(batch_idx,1):.3f}, "
            f"recall: {100.0*recall/max(batch_idx,1):.3f}"
        )
        return final_result


    def test_saved_model_rest(self, model):
        # Bind the provided model to this runner and eval mode
        self.attack_model = model.to(self.device)
        self.attack_model.eval()

        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before test_saved_model_rest()."
            )

        batch_idx = 0
        correct = 0
        total = 0
        prec = 0.0
        recall = 0.0
        total_test_loss = 0.0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_test_gndtrth, final_test_predict, final_test_probabe = [], [], []

        with torch.no_grad():
            for batch in self.attack_cached_test_loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output     = output.to(self.device, non_blocking=True)      # [B,C]
                prediction = prediction.to(self.device, non_blocking=True)   # [B] or [B,1]
                targets    = targets.to(self.device, non_blocking=True)      # [B]
                members    = members.to(self.device, non_blocking=True)      # [B]

                pred_for_model = prediction.unsqueeze(-1) if prediction.dim() == 1 else prediction

                # Forward (raw logits)
                logits = self.attack_model(output, pred_for_model, targets)

                # CE expects logits
                loss = self.criterion(logits, members)
                total_test_loss += loss.item()

                # Probs only for metrics
                probs = F.softmax(logits, dim=1)
                _, predicted = probs.max(dim=1)

                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec   += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_test_gndtrth.append(members.detach().cpu())
                final_test_predict.append(predicted.detach().cpu())
                final_test_probabe.append(probs[:, 1].detach().cpu())

                batch_idx += 1

        # Aggregate
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        avg_test_loss = total_test_loss / max(batch_idx, 1)

        final_result = [
            correct / max(total, 1),
            (prec / max(batch_idx, 1)).item(),
            (recall / max(batch_idx, 1)).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss,
        ]

        print(
            f"Final: Test Acc: {100.0*correct/max(1,total):.3f}% ({correct}/{total}), "
            f"Loss: {avg_test_loss:.3f}, precision: {100.0*prec/max(batch_idx,1):.3f}, "
            f"recall: {100.0*recall/max(batch_idx,1):.3f}"
        )
        return final_result


    def compute_roc_curve_rest(self, model):
        self.attack_model = model.to(self.device)
        self.attack_model.eval()

        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before compute_roc_curve_rest()."
            )

        batch_idx = 0
        correct = 0
        total = 0
        prec = 0.0
        recall = 0.0
        total_test_loss = 0.0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_test_gndtrth, final_test_predict, final_test_probabe = [], [], []

        with torch.no_grad():
            for batch in self.attack_cached_test_loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output     = output.to(self.device, non_blocking=True)
                prediction = prediction.to(self.device, non_blocking=True)
                targets    = targets.to(self.device, non_blocking=True)
                members    = members.to(self.device, non_blocking=True)

                pred_for_model = prediction.unsqueeze(-1) if prediction.dim() == 1 else prediction

                logits = self.attack_model(output, pred_for_model, targets)
                loss = self.criterion(logits, members)
                total_test_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                _, predicted = probs.max(dim=1)

                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec   += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_test_gndtrth.append(members.detach().cpu())
                final_test_predict.append(predicted.detach().cpu())
                final_test_probabe.append(probs[:, 1].detach().cpu())

                batch_idx += 1

        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        avg_test_loss = total_test_loss / max(batch_idx, 1)

        final_result = [
            correct / max(total, 1),
            (prec / max(batch_idx, 1)).item(),
            (recall / max(batch_idx, 1)).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss,
        ]

        # ROC stuff
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, thresholds, roc_auc


    def compute_roc_curve_apcmia(self, atk_model, prt_model, consin_thr, entrp_thr):
        """
        Compute ROC curve and AUC for APCMIA using cached test loader (output, pred, targets, members).
        Prereq: prepare_dataset_new() + bind_cached_attack_loaders(...) have been called.
        """
        # bind models + thresholds
        self.attack_model  = atk_model.to(self.device)
        self.perturb_model = prt_model.to(self.device)
        self.cosine_threshold           = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        self.attack_model.eval()
        self.perturb_model.eval()

        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before compute_roc_curve_apcmia()."
            )

        final_ground_truth = []
        final_probabilities = []

        with torch.no_grad():
            for batch in self.attack_cached_test_loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output     = output.to(self.device, non_blocking=True)     # [B,C]
                prediction = prediction.to(self.device, non_blocking=True)  # [B] or [B,1]
                targets    = targets.to(self.device, non_blocking=True)     # [B]
                members    = members.to(self.device, non_blocking=True)     # [B]

                pred_for_model = prediction.unsqueeze(-1) if prediction.dim() == 1 else prediction

                # ---- APCMIA perturbation (same as test) ----
                noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                if member_mask.any() and non_member_mask.any():
                    member_indices = member_mask.nonzero(as_tuple=True)[0]
                    non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                    member_pvs = output[member_indices]
                    non_member_pvs = output[non_member_indices]

                    # cosine overlap
                    cos_sim = F.cosine_similarity(
                        non_member_pvs.unsqueeze(1), member_pvs.unsqueeze(0), dim=2
                    )
                    max_cos_sim, _ = cos_sim.max(dim=1)

                    tau = 0.5
                    cosine_threshold = torch.sigmoid(self.cosine_threshold)   # (0,1)
                    logits = (max_cos_sim - cosine_threshold) * self.k1
                    binary_logits = torch.stack([-logits, logits], dim=1)
                    gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                    binary_selection = gumbel_selection[:, 1]

                    alpha = 1.0
                    epsilon = 1e-10

                    learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                    tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                    entropy = - (tentative_perturbed * torch.log2(tentative_perturbed + epsilon)).sum(dim=1)

                    quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)  # (0,1)
                    quantile_val = torch.quantile(entropy, quantile_threshold)
                    entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                    final_selection = binary_selection * entropy_mask

                    perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                    perturbed_pvs = output.clone()
                    perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                else:
                    perturbed_pvs = output.clone()

                # normalize perturbed posteriors
                perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                # attack forward -> logits; softmax only for ROC probs
                logits_attack = self.attack_model(perturbed_pvs, pred_for_model, targets)
                probs = F.softmax(logits_attack, dim=1)          # [B,2]
                memb_prob = probs[:, 1]                          # P(member)

                final_ground_truth.append(members.detach().cpu())
                final_probabilities.append(memb_prob.detach().cpu())

        # concat and compute ROC
        final_ground_truth = torch.cat(final_ground_truth, dim=0).numpy()
        final_probabilities = torch.cat(final_probabilities, dim=0).numpy()

        fpr, tpr, thresholds = roc_curve(final_ground_truth, final_probabilities)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, thresholds, roc_auc


    def compute_entropy_distribution_2(self, models_apth, dataset="test", plot=True, save_path=None):
        """
        Loads the saved models and thresholds from 'models_apth', then computes and (optionally)
        plots the entropy distributions of probability vectors (PVs) before and after perturbation.
        The resulting plot has two subplots:
        - Top: members vs. non-members (original entropies, before perturbation)
        - Bottom: members vs. non-members (perturbed entropies, after perturbation)
        """

        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # 1. Load checkpoint and restore attack/perturb models + thresholds.
        checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)

        self.attack_model.eval()
        self.perturb_model.eval()

        # 2. Select the file based on dataset (train/test).
        if dataset.lower() == "test":
            file_path = self.ATTACK_SETS + "test.p"
        elif dataset.lower() == "train":
            file_path = self.ATTACK_SETS + "train.p"
        else:
            raise ValueError("Dataset must be 'test' or 'train'.")

        # We'll store four lists of entropies:
        # (a) Members' original entropies
        # (b) Non-members' original entropies
        # (c) Members' perturbed entropies
        # (d) Non-members' perturbed entropies
        members_orig_list = []
        nonmembers_orig_list = []
        members_pert_list = []
        nonmembers_pert_list = []

        epsilon = 1e-10
        k = self.k if hasattr(self, "k") else 50.0  # steepness factor for entropy filtering

        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device
                    output = output.to(self.device)
                    members = members.to(self.device)
                    targets = targets.to(self.device)

                    # If output is not already a probability distribution, you might do:
                    # output = F.softmax(output, dim=1)

                    # (A) Original entropies
                    orig_entropy = -torch.sum(output * torch.log2(output + epsilon), dim=1)

                    # Separate members vs. non-members for original entropies
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)
                    # Extend to CPU lists
                    members_orig_list.extend(orig_entropy[member_mask].cpu().numpy())
                    nonmembers_orig_list.extend(orig_entropy[non_member_mask].cpu().numpy())

                    # (B) Compute perturbed output using the same logic as in your test function
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]

                        # Overlap detection
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),
                            member_pvs.unsqueeze(0),
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)

                        temperature = 10.0
                        tau = 0.5
                        cos_thresh = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cos_thresh) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        # Entropy filtering
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log2(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)

                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                        # Build the final perturbed output
                        perturbed_output = output.clone()
                        perturbed_output[non_member_indices] = perturbed_non_member_pvs
                    else:
                        # If no members or no non-members, just keep the output as-is
                        perturbed_output = output.clone()

                    # Normalize & clip
                    perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                    perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)

                    # (C) Perturbed entropies
                    pert_entropy = -torch.sum(perturbed_output * torch.log2(perturbed_output + epsilon), dim=1)

                    # Separate members vs. non-members for perturbed entropies
                    members_pert_list.extend(pert_entropy[member_mask].cpu().numpy())
                    nonmembers_pert_list.extend(pert_entropy[non_member_mask].cpu().numpy())

        # If plot=True, we create two subplots: top for "before", bottom for "after"
        if plot:
            # Choose a style
            if "seaborn-darkgrid" in plt.style.available:
                plt.style.use("seaborn-darkgrid")
            elif "seaborn" in plt.style.available:
                plt.style.use("seaborn")
            else:
                plt.style.use("default")

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
            fig.suptitle("Entropy Distribution Before and After Perturbation", fontsize=16, fontweight="bold")

            # Colors
            color_members = "#E74C3C"     # Professional red
            color_nonmembers = "#2E86C1"  # Steel blue

            # Top subplot: original entropies
            axes[0].hist(members_orig_list, bins=50, alpha=0.5, label="Members (before)",
                        color=color_members, edgecolor="black")
            axes[0].hist(nonmembers_orig_list, bins=50, alpha=0.5, label="Non-members (before)",
                        color=color_nonmembers, edgecolor="black")
            axes[0].set_ylabel("Frequency", fontsize=13)
            axes[0].legend(fontsize=12)
            axes[0].grid(True, linestyle="--", alpha=0.5)
            axes[0].set_title("Before Perturbation", fontsize=14, fontweight="bold")

            # Bottom subplot: perturbed entropies
            axes[1].hist(members_pert_list, bins=50, alpha=0.7, label="Members (after)",
                        color=color_members, edgecolor="black")
            axes[1].hist(nonmembers_pert_list, bins=50, alpha=0.7, label="Non-members (after)",
                        color=color_nonmembers, edgecolor="black")
            axes[1].set_xlabel("Entropy", fontsize=13)
            axes[1].set_ylabel("Frequency", fontsize=13)
            axes[1].legend(fontsize=12)
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].set_title("After Perturbation", fontsize=14, fontweight="bold")

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for the suptitle
            if save_path is not None:
                plt.savefig(save_path, dpi=300)
            plt.show()

        # Return the arrays if you need them
        return (np.array(members_orig_list),
                np.array(nonmembers_orig_list),
                np.array(members_pert_list),
                np.array(nonmembers_pert_list))



    def compute_entropy_distribution_new_norm(self, atk_model, prt_model, consin_thr, entrp_thr, entropy_dis_dr):
        """
        Compute entropy distributions (before/after perturbation) for members vs non-members,
        using cached test loader (output, pred, targets, members). Saves a PDF of the 'Before'
        distributions; returns the four numpy arrays.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        import torch.nn.functional as F

        # Bind models & thresholds
        self.attack_model  = atk_model.to(self.device)
        self.perturb_model = prt_model.to(self.device)
        self.cosine_threshold           = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        self.attack_model.eval()
        self.perturb_model.eval()

        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) before this method."
            )

        epsilon = 1e-10
        orig_all_list, pert_all_list = [], []
        members_all_list, targets_all_list = [], []
        orig_members_list, orig_nonmembers_list = [], []
        pert_members_list, pert_nonmembers_list = [], []

        with torch.no_grad():
            for batch in self.attack_cached_test_loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output  = output.to(self.device, non_blocking=True)   # [B,C], probabilities from cache
                targets = targets.to(self.device, non_blocking=True)  # [B]
                members = members.to(self.device, non_blocking=True)  # [B]

                # ---- Original entropies (log base 2) ----
                orig_entropy = -torch.sum(output * torch.log2(output + epsilon), dim=1)  # [B]
                orig_all_list.append(orig_entropy.detach().cpu())
                members_all_list.append(members.detach().cpu())
                targets_all_list.append(targets.detach().cpu())
                orig_members     = orig_entropy[members == 1]
                orig_nonmembers  = orig_entropy[members == 0]
                orig_members_list.append(orig_members.detach().cpu())
                orig_nonmembers_list.append(orig_nonmembers.detach().cpu())

                # ---- APCMIA perturbation on non-members (same as test) ----
                noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                if member_mask.any() and non_member_mask.any():
                    member_indices     = member_mask.nonzero(as_tuple=True)[0]
                    non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                    member_pvs    = output[member_indices]        # [Nm, C]
                    non_member_pvs= output[non_member_indices]    # [Nn, C]

                    # Step 2: overlap via cosine similarity
                    cos_sim = F.cosine_similarity(
                        non_member_pvs.unsqueeze(1),  # [Nn,1,C]
                        member_pvs.unsqueeze(0),      # [1,Nm,C]
                        dim=2
                    )                                  # [Nn,Nm]
                    max_cos_sim, _ = cos_sim.max(dim=1)  # [Nn]

                    tau = 0.5
                    cosine_threshold = torch.sigmoid(self.cosine_threshold)   # (0,1)
                    # temperature: keep consistent with your training/test setting (k1)
                    temperature = self.k1
                    logits = (max_cos_sim - cosine_threshold) * temperature
                    binary_logits = torch.stack([-logits, logits], dim=1)     # [Nn,2]
                    gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                    binary_selection = gumbel_selection[:, 1]                  # [Nn]

                    alpha = 1.0
                    learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                    tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                    # entropy filter
                    entropy_vals = -torch.sum(tentative_perturbed * torch.log2(tentative_perturbed + epsilon), dim=1)  # [Nn]
                    quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)  # (0,1)
                    quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                    entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * self.k)
                    final_selection = binary_selection * entropy_mask

                    perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                    perturbed_output = output.clone()
                    perturbed_output[non_member_indices] = perturbed_non_member_pvs
                else:
                    perturbed_output = output.clone()

                # Normalize/clamp to ensure valid PVs
                perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)

                # Perturbed entropies (log base 2)
                pert_entropy = -torch.sum(perturbed_output * torch.log2(perturbed_output + epsilon), dim=1)
                pert_all_list.append(pert_entropy.detach().cpu())
                pert_members     = pert_entropy[members == 1]
                pert_nonmembers  = pert_entropy[members == 0]
                pert_members_list.append(pert_members.detach().cpu())
                pert_nonmembers_list.append(pert_nonmembers.detach().cpu())

        # Concatenate all batches
        original_members    = torch.cat(orig_members_list, dim=0).numpy()
        original_nonmembers = torch.cat(orig_nonmembers_list, dim=0).numpy()
        pert_members        = torch.cat(pert_members_list, dim=0).numpy()
        pert_nonmembers     = torch.cat(pert_nonmembers_list, dim=0).numpy()

        original_all = torch.cat(orig_all_list, dim=0).numpy() if orig_all_list else np.array([])
        perturbed_all = torch.cat(pert_all_list, dim=0).numpy() if pert_all_list else np.array([])
        members_all = torch.cat(members_all_list, dim=0).numpy() if members_all_list else np.array([])
        targets_all = torch.cat(targets_all_list, dim=0).numpy() if targets_all_list else np.array([])
        # ---- Normalize by maximum possible entropy with log base 2: log2(C) ----
        max_entropy = np.log2(self.num_classes)
        original_members    = original_members / max_entropy
        original_nonmembers = original_nonmembers / max_entropy
        pert_members        = pert_members / max_entropy
        pert_nonmembers     = pert_nonmembers / max_entropy
        if original_all.size:
            original_all = original_all / max_entropy
        if perturbed_all.size:
            perturbed_all = perturbed_all / max_entropy
        members_all = members_all.astype(int) if members_all.size else members_all
        targets_all = targets_all.astype(int) if targets_all.size else targets_all

        # ---- Plot side-by-side histograms for before/after ----
        size = 30
        params = {
            'axes.labelsize': size,
            'font.size': size,
            'legend.fontsize': size - 6,
            'xtick.labelsize': size - 6,
            'ytick.labelsize': size - 6,
            'figure.figsize': [18, 8],
            'font.family': 'arial',
        }
        plt.rcParams.update(params)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        bins = 20

        def _plot_hist(ax, members_data, nonmembers_data, title, show_ylabel):
            ax.grid(linestyle='dotted')
            ax.set_axisbelow(True)
            counts_m, bin_edges = np.histogram(members_data, bins=bins, density=True)
            counts_n, _ = np.histogram(nonmembers_data, bins=bins, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            width = bin_width * 0.95 / 2.0
            ax.bar(bin_edges[:-1] - width / 2, counts_m, width=width, color='#2421f7', label='Members')
            ax.bar(bin_edges[:-1] + width / 2, counts_n, width=width, color='#f10219', label='Non-Members')
            ax.set_title(title, fontsize=size, fontweight='bold')
            ax.set_xlabel('Prediction Uncertainty')
            if show_ylabel:
                ax.set_ylabel('Normalized Frequency')

        _plot_hist(axes[0], original_members, original_nonmembers, 'Before', show_ylabel=True)
        _plot_hist(axes[1], pert_members, pert_nonmembers, 'After', show_ylabel=False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs(entropy_dis_dr, exist_ok=True)
        output_path = os.path.join(entropy_dis_dr, f"{self.dataset_name}_entropy_dist.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._plot_entropy_ridgeline(original_members, original_nonmembers, pert_members, pert_nonmembers, entropy_dis_dr)
        # self._plot_entropy_cdf(original_members, original_nonmembers, pert_members, pert_nonmembers, entropy_dis_dr)
        # self._plot_entropy_scatter(original_all, perturbed_all, members_all, targets_all, entropy_dis_dr)

        print(f"Saved entropy distribution plot to {output_path}")


    def compute_entropy_distribution_from_checkpoint(self, bundle_path, entropy_dis_dr):
        """Load a saved attack/perturb bundle and produce entropy histograms without retraining."""
        map_location = self.device
        if isinstance(map_location, str):
            map_location = torch.device(map_location)

        bundle = torch.load(bundle_path, map_location=map_location)

        attack_state = bundle.get('attack_model_state_dict')
        perturb_state = bundle.get('perturb_model_state_dict')
        if attack_state is None or perturb_state is None:
            raise ValueError("Checkpoint missing model state dicts; expected 'attack_model_state_dict' and 'perturb_model_state_dict'.")

        self.attack_model.load_state_dict(attack_state)
        self.perturb_model.load_state_dict(perturb_state)

        cosine_thr = bundle.get('cosine_threshold')
        entropy_thr = bundle.get('Entropy_quantile_threshold', bundle.get('entropy_threshold'))
        if cosine_thr is None or entropy_thr is None:
            raise ValueError("Checkpoint missing learned thresholds.")

        cosine_thr = cosine_thr.to(self.device) if isinstance(cosine_thr, torch.Tensor) else torch.tensor(cosine_thr, device=self.device)
        entropy_thr = entropy_thr.to(self.device) if isinstance(entropy_thr, torch.Tensor) else torch.tensor(entropy_thr, device=self.device)

        if not hasattr(self, 'attack_cached_test_loader'):
            raise RuntimeError(
                "attack_cached_test_loader not found. Call prepare_dataset_new() and bind_cached_attack_loaders(...) before this method."
            )

        return self.compute_entropy_distribution_new_norm(
            self.attack_model,
            self.perturb_model,
            cosine_thr,
            entropy_thr,
            entropy_dis_dr,
        )

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved entropy distribution plot to {output_path}")
        print(f"Saved entropy distribution plot to {output_path}")

        # If you want both Before/After in one figure, uncomment your earlier dual-subplot block.

        return original_members, original_nonmembers, pert_members, pert_nonmembers

   
    def _use_arial_fonts(self):
        import matplotlib as mpl
        from matplotlib import font_manager

        # Prefer Arial, fall back gracefully if not installed
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "Nimbus Sans", "sans-serif"],
            "axes.unicode_minus": False,   # proper minus sign
            "pdf.fonttype": 42,            # embed TrueType in PDF
            "ps.fonttype": 42,             # embed TrueType in PS
            "svg.fonttype": "none",        # keep text selectable in SVG (if you export SVGs)
        })
        
    def _plot_entropy_ridgeline(self, original_members, original_nonmembers, pert_members, pert_nonmembers, entropy_dis_dr=None):
        import os
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from pathlib import Path

        self._use_arial_fonts()
        sns.set_theme(style="white", context="talk")

        records = []
        for values, membership, state in [
            (original_members, "Members", "Before"),
            (original_nonmembers, "Non-Members", "Before"),
            (pert_members, "Members", "After"),
            (pert_nonmembers, "Non-Members", "After"),
        ]:
            if values.size:
                records.extend((float(v), membership, state) for v in values)

        if not records:
            return

        df = pd.DataFrame(records, columns=["entropy", "membership", "state"])
        membership_order = ["Members", "Non-Members"]
        df["membership"] = pd.Categorical(df["membership"], categories=membership_order, ordered=True)

        palette = {"Before": "#2421f7", "After": "#f10219"}

        g = sns.FacetGrid(
            df,
            row="membership",
            hue="state",
            hue_order=["Before", "After"],
            height=2.6,
            aspect=3.2,
            sharex=True,
            sharey=False,
            palette=palette,
        )

        g.map_dataframe(sns.kdeplot, x="entropy", fill=True,  alpha=0.55, linewidth=0,   bw_adjust=0.9)
        g.map_dataframe(sns.kdeplot, x="entropy", fill=False, linewidth=1.2,             bw_adjust=0.9)

        # Typography controls
        xlabel_fs = 25
        xtick_fs  = max(1, xlabel_fs - 5)  # 5 less than x-label font size

        # Subplot formatting
        for ax, membership in zip(g.axes.flat, membership_order):
            ax.set_ylabel("Density", fontsize=25)
            ax.set_xlabel("")                       # global x-label below
            ax.tick_params(axis="x", labelsize=xtick_fs)
            ax.set_yticks([])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        # Subplot titles (“tiles”)
        try:
            g.set_titles(row_template="{row_name}", size=25)
        except Exception:
            # Fallback for older seaborn: set manually
            for ax, membership in zip(g.axes.flat, membership_order):
                ax.set_title(membership, fontsize=25)

        g.fig.supxlabel("Prediction Uncertainty", fontsize=xlabel_fs)
        g.fig.subplots_adjust(top=0.9, hspace=0.4)

        handles = [
            Patch(facecolor=palette["Before"], edgecolor=palette["Before"], label="Before"),
            Patch(facecolor=palette["After"],  edgecolor=palette["After"],  label="After"),
        ]
        g.fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 1.00), fontsize=25-3, frameon=False)


        from pathlib import Path

        # Folder next to this .py file
        try:
            _BASE_DIR = Path(__file__).resolve().parent
        except NameError:
            _BASE_DIR = Path.cwd()  # fallback if running interactively

        _ALL_DIS = _BASE_DIR / "all_dis"
        _ALL_DIS.mkdir(parents=True, exist_ok=True)
        out_path = _ALL_DIS / f"{self.dataset_name}_{self.arch}_entropy_ridgeline.pdf"
        g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(g.fig)


    def _plot_entropy_cdf(self, original_members, original_nonmembers, pert_members, pert_nonmembers, entropy_dis_dr):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        combinations = [
            ("Members", "Before", original_members),
            ("Members", "After", pert_members),
            ("Non-Members", "Before", original_nonmembers),
            ("Non-Members", "After", pert_nonmembers),
        ]

        plt.figure(figsize=(10, 6))
        for membership, state, values in combinations:
            if values.size == 0:
                continue
            sorted_vals = np.sort(values)
            cdf = np.linspace(0, 1, sorted_vals.size, endpoint=True)
            plt.plot(sorted_vals, cdf, label=f"{membership} - {state}")

        plt.xlabel("Prediction Uncertainty")
        plt.ylabel("Empirical CDF")
        plt.title("Entropy Empirical CDF")
        plt.legend()
        plt.grid(True, linestyle="dotted", alpha=0.6)

        os.makedirs(entropy_dis_dr, exist_ok=True)
        out_path = os.path.join(entropy_dis_dr, f"{self.dataset_name}_entropy_cdf.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_entropy_scatter(self, original_all, perturbed_all, memberships, targets, entropy_dis_dr):
            import os
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            if original_all.size == 0 or memberships.size == 0 or targets.size == 0:
                return

            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.2, 0.2, size=targets.size if targets.size else 0)
            target_jitter = targets.astype(float) + jitter if targets.size else targets

            df = pd.DataFrame({
                "entropy": np.concatenate([original_all, perturbed_all]),
                "state": ["Before"] * original_all.size + ["After"] * perturbed_all.size,
                "membership": np.where(np.concatenate([memberships, memberships]) == 1, "Members", "Non-Members"),
                "target": np.concatenate([target_jitter, target_jitter]) if targets.size else np.concatenate([targets, targets])
            })

            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=df, x="target", y="entropy", hue="state", style="membership", alpha=0.6)
            plt.xlabel("Class Index (jittered)")
            plt.ylabel("Prediction Uncertainty")
            plt.title("Entropy vs. Class Scatter")
            plt.grid(True, linestyle="dotted", alpha=0.5)

            os.makedirs(entropy_dis_dr, exist_ok=True)
            out_path = os.path.join(entropy_dis_dr, f"{self.dataset_name}_entropy_scatter.pdf")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

    def compute_cosine_similarity_distribution(self, checkpoint_path, dataset="test", plot=True, save_path=None):
        """
        Loads the saved models and learned thresholds from checkpoint, then computes the cosine
        similarity between the original PVs and the perturbed PVs for each sample. It separates
        the similarity values for member and non-member samples.

        Args:
            checkpoint_path (str): Path to the checkpoint saved via save_att_per_thresholds_models.
            dataset (str): "test" or "train" to choose which dataset to use.
            plot (bool): Whether to plot the histogram of cosine similarities.
            save_path (str): If provided, the plot is saved to this file.

        Returns:
            member_similarities (np.ndarray): Cosine similarities for member samples.
            non_member_similarities (np.ndarray): Cosine similarities for non-member samples.
        """
        # Load checkpoint (using weights_only=True for security).
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)
        
        self.attack_model.eval()
        self.perturb_model.eval()

        member_sim_list = []
        non_member_sim_list = []
        
        # Choose dataset file.
        if dataset.lower() == "test":
            file_path = self.ATTACK_SETS + "test.p"
        elif dataset.lower() == "train":
            file_path = self.ATTACK_SETS + "train.p"
        else:
            raise ValueError("Dataset must be 'test' or 'train'")
        
        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Assume 'output' already represents a probability distribution.
                    original = output  # Original PVs

                    # Compute perturbed output using the same procedure as in test().
                    noisy_members, member_mask, non_member_mask = self._membership_masks(members)
                    
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]
                        
                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]
                        
                        # Overlap detection via cosine similarity.
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
                            member_pvs.unsqueeze(0),      # (1, n_members, C)
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)
                        
                        # Gumbel–Softmax binary selection.
                        temperature = 10.0
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]  # (n_non_members,)
                        
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                        
                        # Entropy filtering.
                        epsilon = 1e-10
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log2(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        k = self.k if hasattr(self, "k") else 50.0
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)
                        
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        
                        # Construct perturbed output.
                        perturbed = output.clone()
                        perturbed[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed = output.clone()
                    
                    # Ensure perturbed output is normalized.
                    perturbed = torch.clamp(perturbed, min=1e-6, max=1)
                    perturbed = perturbed / perturbed.sum(dim=1, keepdim=True)
                    
                    # Now, compute cosine similarity for each sample between original and perturbed.
                    # This computes the cosine similarity for each row (sample).
                    sim = F.cosine_similarity(original, perturbed, dim=1)
                    
                    # Append the similarity values separately for members and non-members.
                    member_sim_list.append(sim[member_mask].cpu())
                    non_member_sim_list.append(sim[non_member_mask].cpu())
        

        
        if member_sim_list:
            member_similarities = torch.cat(member_sim_list, dim=0).numpy()
        else:
            member_similarities = np.array([])
        print(f"size: {member_similarities.size}")
        if non_member_sim_list:
            non_member_similarities = torch.cat(non_member_sim_list, dim=0).numpy()
        else:
            non_member_similarities = np.array([])

        print(f"First few member similarities: {member_similarities[:5]}")
        print(f"First few non-member similarities: {non_member_similarities[:5]}")
        exit()
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,6))
            if member_similarities.size > 0:
                plt.hist(member_similarities, bins=50, alpha=0.5, label="Members (Unperturbed)", color="blue")
            if non_member_similarities.size > 0:
                plt.hist(non_member_similarities, bins=50, alpha=0.5, label="Non-Members (Perturbed)", color="orange")
            plt.xlabel("Cosine Similarity (Original vs. Perturbed)")
            plt.ylabel("Frequency")
            plt.title("Cosine Similarity Distribution Before and After Perturbation")
            plt.legend()
            if save_path is not None:
                plt.savefig(save_path)
            plt.show()

        return member_similarities, non_member_similarities

            
    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
    
    def save_pertub_Model(self, path):
        torch.save(self.perturb_model.state_dict(), self.Perturb_MODELS_PATH)
    # chch
    def save_att_per_thresholds_models(self, checkpoint_path, path):

        map_location = torch.device(self.device) if isinstance(self.device, str) else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        self.cosine_threshold.data.copy_(checkpoint['cosine_threshold'].to(self.cosine_threshold.device))
        self.Entropy_quantile_threshold.data.copy_(checkpoint['entropy_threshold'].to(self.Entropy_quantile_threshold.device))
        
        models_threshold_params = {
            'attack_model_state_dict': self.attack_model.state_dict(),
            'perturb_model_state_dict': self.perturb_model.state_dict(),
            'cosine_threshold': self.cosine_threshold.detach().cpu(),
            'Entropy_quantile_threshold': self.Entropy_quantile_threshold.detach().cpu()
        }
        torch.save(models_threshold_params, path)
        
    def load_perturb_model(self):

        # gan_path = self.Perturb_MODELS_PATH
        # generator = Generator(input_dim).to(device)
        # self.perturb_model = perturb_model.to(self.device)

        self.perturb_model.load_state_dict(torch.load(self.Perturb_MODELS_PATH, weights_only=True))
        self.perturb_model.eval()  # Set the generator to evaluation mode
        return self.perturb_model


    def visualize_transformed_pvs_classwise(self, target_class, atk_model, prt_model, consin_thr, entrp_thr, sub_folder):
        """
        Visualize PVs for a single target class before/after APCMIA perturbation.
        Uses cached loader batches: (output, pred, targets, members).
        Saves a PDF to <sub_folder>/class_<target_class>.pdf
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE
        import torch
        import torch.nn.functional as F

        # --- setup ---
        if not hasattr(self, "attack_cached_test_loader"):
            raise RuntimeError(
                "attack_cached_test_loader not found. "
                "Call prepare_dataset_new() and bind_cached_attack_loaders(...) first."
            )

        self.attack_model  = atk_model.to(self.device)   # not used here but kept for consistency
        self.perturb_model = prt_model.to(self.device)
        self.cosine_threshold           = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        self.attack_model.eval()
        self.perturb_model.eval()

        sns.set_context("paper", font_scale=1.5)
        sns.set_style("whitegrid")

        # storage
        original_pvs_list  = []
        perturbed_pvs_list = []
        class_labels_list  = []
        members_list       = []
        pert_flags_list    = []

        epsilon = 1e-10

        with torch.no_grad():
            for batch in self.attack_cached_test_loader:
                if len(batch) != 4:
                    raise ValueError("Expected batch=(output, pred, targets, members).")
                output, prediction, targets, members = batch

                output  = output.to(self.device, non_blocking=True)   # [B,C] (probabilities)
                targets = targets.to(self.device, non_blocking=True)  # [B]
                members = members.to(self.device, non_blocking=True)  # [B]

                # default: no perturbation
                perturbed_output = output.clone()
                batch_perturb_flags = torch.zeros_like(members, dtype=torch.bool)

                noisy_members, member_mask, non_member_mask = self._membership_masks(members)

                if member_mask.any() and non_member_mask.any():
                    member_idx     = member_mask.nonzero(as_tuple=True)[0]
                    non_member_idx = non_member_mask.nonzero(as_tuple=True)[0]

                    member_pvs     = output[member_idx]        # [Nm, C]
                    non_member_pvs = output[non_member_idx]    # [Nn, C]

                    # Step 2: cosine overlap
                    cos_sim = F.cosine_similarity(
                        non_member_pvs.unsqueeze(1),  # [Nn,1,C]
                        member_pvs.unsqueeze(0),      # [1,Nm,C]
                        dim=2
                    )                                 # [Nn,Nm]
                    max_cos_sim, _ = cos_sim.max(dim=1)  # [Nn]

                    tau = 0.5
                    cosine_threshold = torch.sigmoid(self.cosine_threshold)  # (0,1)
                    temperature = self.k1                                    # keep consistent with your train/test
                    logits = (max_cos_sim - cosine_threshold) * temperature  # [Nn]
                    binary_logits = torch.stack([-logits, logits], dim=1)    # [Nn,2]
                    gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                    binary_selection = gumbel_selection[:, 1]                # [Nn]

                    alpha = 1.0
                    learned_values = self.perturb_model(non_member_pvs, targets[non_member_idx])
                    tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                    # entropy filter
                    entropy_vals = -torch.sum(tentative_perturbed * torch.log2(tentative_perturbed + epsilon), dim=1)
                    q = torch.sigmoid(self.Entropy_quantile_threshold)
                    q_val = torch.quantile(entropy_vals, q)
                    entropy_mask = torch.sigmoid((entropy_vals - q_val) * self.k)
                    final_selection = binary_selection * entropy_mask

                    perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                    perturbed_output[non_member_idx] = perturbed_non_member_pvs
                    batch_perturb_flags[non_member_idx] = (final_selection > 0.5)

                # normalize/clamp PVs
                perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)

                # collect
                original_pvs_list.append(output.detach().cpu().numpy())
                perturbed_pvs_list.append(perturbed_output.detach().cpu().numpy())
                class_labels_list.append(targets.detach().cpu().numpy())
                members_list.append(members.detach().cpu().numpy())
                pert_flags_list.append(batch_perturb_flags.detach().cpu().numpy())

        if not original_pvs_list:
            print("No PVs found.")
            return

        # stack arrays
        original_pvs  = np.vstack(original_pvs_list)   # [N,C]
        perturbed_pvs = np.vstack(perturbed_pvs_list)  # [N,C]
        class_labels  = np.hstack(class_labels_list)   # [N]
        membership    = np.hstack(members_list)        # [N]
        pert_flags    = np.hstack(pert_flags_list)     # [N]

        # filter target class
        mask = (class_labels == target_class)
        if mask.sum() == 0:
            print(f"No samples for target class {target_class}.")
            return

        original_pvs  = original_pvs[mask]
        perturbed_pvs = perturbed_pvs[mask]
        membership    = membership[mask]
        pert_flags    = pert_flags[mask]

        # ---- t-SNE ----
        # keep perplexity valid: must be < n_samples
        n_samp = original_pvs.shape[0]
        perp = min(30, max(5, n_samp // 3))  # a safe heuristic
        reducer = TSNE(n_components=2, perplexity=perp, learning_rate=200, random_state=42)
        stacked = np.vstack([original_pvs, perturbed_pvs])
        reduced = reducer.fit_transform(stacked)
        orig_2d = reduced[:n_samp]
        pert_2d = reduced[n_samp:]

        # ---- Plot ----
        size = 20
        plt.rcParams.update({
            'axes.labelsize': size, 'font.size': size, 'legend.fontsize': size,
            'xtick.labelsize': size, 'ytick.labelsize': size,
            'figure.figsize': [16, 8], "font.family": "arial",
        })

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        palette = {"Members": "#1f77b4", "Non-Members": "#ff7f0e"}
        labels_str = np.where(membership == 1, "Members", "Non-Members")

        # Before
        sns.scatterplot(
            x=orig_2d[:, 0], y=orig_2d[:, 1],
            hue=labels_str, palette=palette, alpha=0.7, s=100, edgecolor='black', ax=ax[0]
        )
        ax[0].set_title("Before", fontsize=20, fontweight='bold')
        ax[0].set_xlabel("t-SNE Component 1", fontsize=20)
        ax[0].set_ylabel("t-SNE Component 2", fontsize=20)
        ax[0].grid(True, linestyle="--", alpha=0.5)
        if ax[0].get_legend(): ax[0].get_legend().remove()

        # After
        sns.scatterplot(
            x=pert_2d[:, 0], y=pert_2d[:, 1],
            hue=labels_str, palette=palette, alpha=0.7, s=100, edgecolor='black', ax=ax[1]
        )
        ax[1].set_title("After", fontsize=20, fontweight='bold')
        ax[1].set_xlabel("t-SNE Component 1", fontsize=20)
        ax[1].grid(True, linestyle="--", alpha=0.5)
        if ax[1].get_legend(): ax[1].get_legend().remove()

        # highlight perturbed non-members on "After"
        nm_pert = (membership != 1) & (pert_flags == 1)
        ax[1].scatter(
            pert_2d[nm_pert, 0], pert_2d[nm_pert, 1],
            facecolors="#ff7f0e", edgecolors="red", s=120, marker='o', label='Perturbed Non-Members'
        )

        # global legend
        h_left, l_left = ax[0].get_legend_handles_labels()
        h_right, l_right = ax[1].get_legend_handles_labels()
        handles, labels = [], []
        for h, l in zip(h_left + h_right, l_left + l_right):
            if l not in labels:
                handles.append(h); labels.append(l)

        fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=20, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        os.makedirs(sub_folder, exist_ok=True)
        save_path = os.path.join(sub_folder, f"class_{target_class}.pdf")
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches="tight")
        plt.close()
        print(f"Saved t-SNE PV visualization to {save_path}")

    def liRA_offline(self):
        
        
        

        # Question? does the loaders have both members and non-members
        #  yes, 
        # self.attack_train_loader # Contains raw samples (made loader) used to train the target model
        # self.attack_test_loader #  Contains raw samples (made loader) used to test the target model,
        # also called target samples, these are used to obtain test PVs to test attack model
        
        # here the test.p contains conf_ob: confobs = (f(x)y)
        # train is the confs of target model that did see target point (x,y), in this case its in test_loader
        from scipy.stats import norm

        outputs_list = []
        members_list = []
        targets_list = []

        # Load data from the saved file (train.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "train.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_list.append(output.cpu())    # output: [batch, num_classes]
                    members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)

        
       
        out_signals = all_outputs     # non-members (out)
        
        
        mean_out = np.median(out_signals.numpy(), 1).reshape(-1, 1)

       
        std_out = np.std(out_signals.numpy(), 1).reshape(-1, 1)

        print("Estimated distribution parameters:")
        # print("Mean In-Signal:", mean_in)
        print("Mean Out-Signal:", mean_out)
        # print("Std In-Signal:", std_in)
        print("Std Out-Signal:", std_out)
        # exit()
        # Now, for each sample, compute the negative log-likelihood under the two distributions.
        # Here, sc (signal observed) is the correct confidence for each sample.
        
        outputs_test_list = []
        members_test_list = []
        targets_test_list = []

        # Load data from the saved file (test.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
                    members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_test_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
        all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
        all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

        # print("Loaded data from test.p:")
        # print(f"Outputs shape: {all_outputs_test.shape}")
        # print(f"Members shape: {all_members_test.shape}")
        # print(f"Targets shape: {all_targets_test.shape}")

        # exit()
        sc = all_outputs_test

        # mean_out = np.median(out_signals, 1).reshape(-1, 1)
        
        
        # std_out = np.std(out_signals, 1).reshape(-1, 1)

        # If running in "offline" mode, you might choose to ignore the in-part
        # (i.e., set pr_in=0) and only use the out-distribution.
        

        # print(f"Shape of all_outputs_test: {all_outputs_test.shape}")
        # print(f"Shape of mean_out: {mean_out.shape}")
        # print(f"Shape of std_out: {std_out.shape}")
        # exit()
        prediction = []
        answers = []

      
        pr_in = 0
        
        pr_out = -norm.logpdf(all_outputs_test, mean_out, std_out + 1e-30) # gaussian approximation
        score = pr_in - pr_out

        prediction = np.array(score.mean(1))

        # prediction_2 = np.array(score.mean(1))
        prediction_2 = np.where(-score.mean(1) >= 0, 1, 0)
        # For each sample, the membership score is given by 'score'.
        # (You can decide on a threshold—for example, if score < 0, predict member.)
        # print("First 10 LiRA scores:", score[:10])
        # print(f"size of score: {score.shape}")
        # print("Predictions (first 10):", prediction[:10])
        # print("Predictions size:", prediction.shape)
        # # Check if all prediction values are probabilities and print
        
        # # Check if all prediction values are less than 1
        # print(f"Max of prediction: {np.max(prediction)}")
        # # Print a few predictions to compare
        # print("First 10 Predictions (Threshold Applied):", prediction_2[:10])
        # print("First 10 Predictions (Raw Scores):", prediction[:10])
        # print(f"Max of prediction_2: {np.max(prediction_2)}")
        # print(f"Min of prediction_2: {np.min(prediction_2)}")

        # correct = predicted.eq(all_members_test).sum().item()
        answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        fpr_list, tpr_list, thresholds = roc_curve(answers.ravel(), (-prediction).ravel())
        auc_score = auc(fpr_list, tpr_list)
        # bcm = BinaryConfusionMatrix().to(self.device)
        # conf_mat = bcm((-prediction).ravel(), all_members_test.ravel())
        

       

        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        
        recall = np.max(tpr_list)
        print(f"acc: {acc}, recall: {recall:.3f}")
        # prec = np.sum((prediction > 0) & (answers > 0)) / np.sum(prediction > 0)
        # recall = np.sum((prediction > 0) & (answers > 0)) / np.sum(answers > 0)
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print("First few FPR values:", fpr_list[:10])
        # print("First few TPR values:", tpr_list[:10])
        
          
        # Plot the ROC curve
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr_list, tpr_list, label="ROC curve (AUC = %0.2f)" % auc(fpr_list, tpr_list), lw=2, color='blue')
        # plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate", fontsize=14)
        # plt.ylabel("True Positive Rate", fontsize=14)
        # plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
        # plt.legend(loc="lower right", fontsize=12)
        # plt.grid(True, linestyle="--", alpha=0.5)
        # plt.tight_layout()
        # plt.show()
        # print(f"Correct predictions: {correct}")
        # exit()
        
        return fpr_list, tpr_list, thresholds, auc_score


    def liRA_offline_mul(self):
        
        
        # load N target models (closely mimicking shadow models) trained already
        # the output of these trained models is outsignals
        # for each loaded target model, get their corresponding training dataset
        # use its corresponding dataset to generate outsignals 
        # concatinate for all N models, 
        # for each dataset the N will be different 
        # for testing we will target model trained on the entire dataset

        

        # Question? does the loaders have both members and non-members
        #  yes, 
        # self.attack_train_loader # Contains raw samples (made loader) used to train the target model
        # self.attack_test_loader #  Contains raw samples (made loader) used to test the target model,
        # also called target samples, these are used to obtain test PVs to test attack model
        
        # here the test.p contains conf_ob: confobs = (f(x)y)
        # train is the confs of target model that did see target point (x,y), in this case its in test_loader
        from scipy.stats import norm


        outputs_list = []
        members_list = []
        targets_list = []

        # Load data from the saved file (train.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "train.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_list.append(output.cpu())    # output: [batch, num_classes]
                    members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)

        
       
        out_signals = all_outputs     # non-members (out)
        
        
        mean_out = np.median(out_signals.numpy(), 1).reshape(-1, 1)

       
        std_out = np.std(out_signals.numpy(), 1).reshape(-1, 1)

        print("Estimated distribution parameters:")
        # print("Mean In-Signal:", mean_in)
        print("Mean Out-Signal:", mean_out)
        # print("Std In-Signal:", std_in)
        print("Std Out-Signal:", std_out)
        # exit()
        # Now, for each sample, compute the negative log-likelihood under the two distributions.
        # Here, sc (signal observed) is the correct confidence for each sample.
        
        outputs_test_list = []
        members_test_list = []
        targets_test_list = []

        # Load data from the saved file (test.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
                    members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_test_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
        all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
        all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

        print("Loaded data from test.p:")
        print(f"Outputs shape: {all_outputs_test.shape}")
        print(f"Members shape: {all_members_test.shape}")
        print(f"Targets shape: {all_targets_test.shape}")

        # exit()
        sc = all_outputs_test

        # mean_out = np.median(out_signals, 1).reshape(-1, 1)
        
        
        # std_out = np.std(out_signals, 1).reshape(-1, 1)

        # If running in "offline" mode, you might choose to ignore the in-part
        # (i.e., set pr_in=0) and only use the out-distribution.
        

        prediction = []
        answers = []

      
        pr_in = 0
        
        pr_out = -norm.logpdf(all_outputs_test, mean_out, std_out + 1e-30) # gaussian approximation
        score = pr_in - pr_out

        prediction = np.array(score.mean(1))

        # prediction_2 = np.array(score.mean(1))
        prediction_2 = np.where(-score.mean(1) >= 0, 1, 0)
        # For each sample, the membership score is given by 'score'.
        # (You can decide on a threshold—for example, if score < 0, predict member.)
        # print("First 10 LiRA scores:", score[:10])
        # print(f"size of score: {score.shape}")
        # print("Predictions (first 10):", prediction[:10])
        # print("Predictions size:", prediction.shape)
        # # Check if all prediction values are probabilities and print
        
        # # Check if all prediction values are less than 1
        # print(f"Max of prediction: {np.max(prediction)}")
        # # Print a few predictions to compare
        # print("First 10 Predictions (Threshold Applied):", prediction_2[:10])
        # print("First 10 Predictions (Raw Scores):", prediction[:10])
        # print(f"Max of prediction_2: {np.max(prediction_2)}")
        # print(f"Min of prediction_2: {np.min(prediction_2)}")

        # correct = predicted.eq(all_members_test).sum().item()
        answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        fpr_list, tpr_list, betas = roc_curve(answers.ravel(), (-prediction).ravel())
        # bcm = BinaryConfusionMatrix().to(self.device)
        # conf_mat = bcm((-prediction).ravel(), all_members_test.ravel())
        

        
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        
        recall = np.max(tpr_list)
        print(f"acc: {acc}, recall: {recall:.3f}")
        # prec = np.sum((prediction > 0) & (answers > 0)) / np.sum(prediction > 0)
        # recall = np.sum((prediction > 0) & (answers > 0)) / np.sum(answers > 0)
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print("First few FPR values:", fpr_list[:10])
        # print("First few TPR values:", tpr_list[:10])
        
          
        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_list, tpr_list, label="ROC curve (AUC = %0.2f)" % auc(fpr_list, tpr_list), lw=2, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        # print(f"Correct predictions: {correct}")
        exit()
    
    def enh_mia(self):
       
        # Attack-R with linear interpolation (Ye et al.) https://arxiv.org/pdf/2111.09679.pdf
        # Taken and repadated from https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/plot_attack_via_reference_or_distill.py
        # if len(target_signal.shape) == 2:
        #     sc = target_signal[target_indices,0].reshape(-1, 1) # 50k x 1 , no augmentation
        #     out_signals = out_signals[target_indices,:,0]
        # else:


        outputs_test_list = []
        members_test_list = []
        targets_test_list = []

        # Load data from the saved file (test.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
                    members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_test_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
        all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
        all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

        print("Loaded data from test.p:")
        print(f"Outputs shape: {all_outputs_test.shape}")
        print(f"Members shape: {all_members_test.shape}")
        print(f"Targets shape: {all_targets_test.shape}")

        # exit()
        sc = all_outputs_test
        
        outputs_list = []
        members_list = []
        targets_list = []

        # Load data from the saved file (train.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "train.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_list.append(output.cpu())    # output: [batch, num_classes]
                    members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)

        #
        # Now split the signals by membership.
        # Here we assume that 'members' is 1 for training (member) and 0 for non-member.
        # in_signals = correct_confidences[all_members.bool()]      # members (in)
        out_signals = all_outputs     # non-members (out)


        # sc = target_signal[target_indices] # 50k x 1
        
        def from_correct_logit_to_loss(array): # convert correct logit to the cross entropy loss
            return np.log((1+np.exp(array))/np.exp(array)) # positive
        
        losses = from_correct_logit_to_loss(out_signals).T.numpy() # shape nb_models x nb_target, ref lossses
        check_losses = from_correct_logit_to_loss(sc).T.numpy() # shape nb_target x 1, target losses

        
        dummy_min = np.zeros((1, len(losses[0]))) # shape 1 x nb_target

        dummy_max = dummy_min + 1000 # shape 1 x nb_target

        dat_reference_or_distill = np.sort(np.concatenate((losses, dummy_max, dummy_min), axis=0), axis=0) # shape nb_models + 2 x nb_target 

        prediction = np.array([])
        
        discrete_alpha = np.linspace(0, 1, len(dat_reference_or_distill))
        for i in range(len(dat_reference_or_distill[0])):
            losses_i =  dat_reference_or_distill[:, i]

            # Create the interpolator
            pr = np.interp(check_losses[0,i], losses_i, discrete_alpha)
            
            prediction = np.append(prediction, pr)

       

        answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        # answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        fpr_list, tpr_list, betas = roc_curve(answers.ravel(), (prediction).ravel())

         # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_list, tpr_list, label="ROC curve (AUC = %0.2f)" % auc(fpr_list, tpr_list), lw=2, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        print("OKAY")
        exit()



    def metric_results(fpr_list, tpr_list, thresholds):
        fprs = [0.01,0.001,0.0001,0.00001,0.0] # 1%, 0.1%, 0.01%, 0.001%, 0%
        tpr_dict = {}
        thresholds_dict = {}
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        roc_auc = auc(fpr_list, tpr_list)

        for fpr in fprs:
            tpr_dict[fpr] = tpr_list[np.where(fpr_list <= fpr)[0][-1]] # tpr at fpr
            thresholds_dict[fpr] = thresholds[np.where(fpr_list <= fpr)[0][-1]] # corresponding threshold

        return roc_auc, acc, tpr_dict, thresholds_dict

    def get_attack_dataset_without_shadow(train_set, test_set, batch_size):
        mem_length = int(len(train_set)*0.45)
        nonmem_length = int(len(test_set)*0.45)
        mem_train, mem_test, _ = torch.utils.data.random_split(train_set, [mem_length, mem_length, len(train_set)-(mem_length*2)])
        nonmem_train, nonmem_test, _ = torch.utils.data.random_split(test_set, [nonmem_length, nonmem_length, len(test_set)-(nonmem_length*2)])
        mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(nonmem_test)

        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)
        for i in range(len(nonmem_test)):
            nonmem_test[i] = nonmem_test[i] + (0,)
        for i in range(len(mem_test)):
            mem_test[i] = mem_test[i] + (1,)
            
        attack_train = mem_train + nonmem_train
        attack_test = mem_test + nonmem_test

        attack_trainloader = torch.utils.data.DataLoader(
            attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
        attack_testloader = torch.utils.data.DataLoader(
            attack_test, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)

        return attack_trainloader, attack_testloader


def save_best_checkpoint(val_metric, attack_obj, checkpoint_path):
    """Persist the best observed attack state (models, thresholds, optimisers)."""
    checkpoint = {
        'epoch': getattr(attack_obj, 'current_epoch', None),
        'val_metric': val_metric,
        'attack_model_state_dict': attack_obj.attack_model.state_dict(),
        'perturb_model_state_dict': attack_obj.perturb_model.state_dict(),
        'cosine_threshold': attack_obj.cosine_threshold.detach().cpu(),
        'entropy_threshold': attack_obj.Entropy_quantile_threshold.detach().cpu(),
    }

    if hasattr(attack_obj, 'optimizer') and attack_obj.optimizer is not None:
        checkpoint['attack_optimizer_state'] = attack_obj.optimizer.state_dict()
    if hasattr(attack_obj, 'optimizer_perturb') and attack_obj.optimizer_perturb is not None:
        checkpoint['perturb_optimizer_state'] = attack_obj.optimizer_perturb.state_dict()
    if hasattr(attack_obj, 'optimizer_cosine') and attack_obj.optimizer_cosine is not None:
        checkpoint['cosine_optimizer_state'] = attack_obj.optimizer_cosine.state_dict()
    if hasattr(attack_obj, 'optimizer_quantile_threshold') and attack_obj.optimizer_quantile_threshold is not None:
        checkpoint['entropy_optimizer_state'] = attack_obj.optimizer_quantile_threshold.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved best checkpoint to {checkpoint_path}")

# Combined attack
def attack_mode0_com(
        TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader,
        target_model, shadow_model, attack_model, perturb_model, num_classes, mode, dataset_name,
        attack_name, entropy_dis_dr, apcmia_cluster, arch,
        attack_dataset_batch_size, acc_gap, flip_prob,
        entropy_from_checkpoint=False, entropy_checkpoint_path=""):
    Perturb_MODELS_PATH = ATTACK_PATH + "_perturb_model.pth"

    RESULT_PATH = ATTACK_PATH + "_meminf_attack0_com.p"
    RESULT_PATH_csv = ATTACK_PATH + "_meminf_attack0_com.csv"
    
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0__com"
    ATTACK_SETS_PV_CSV = ATTACK_PATH + "_meminf_attack_pvs.csv"

    fpr_tpr_file_path = ATTACK_PATH + "_FPR_TPR_" + attack_name + "_.csv"



    
    MODELS_PATH_att_per_thr = ATTACK_PATH + "_attack_pertubr_thresholds_"+attack_name+".pth" # will store all 
    bundle_path = entropy_checkpoint_path if entropy_checkpoint_path else MODELS_PATH_att_per_thr

    
    epoch_data = []
    train_accuracy_list = []
    test_accuracy_list = []
    res_list = []
    cosine_entropy_threshold_list = []

        
    attack = attack_for_blackbox_com_NEW(TARGET_PATH, SHADOW_PATH, Perturb_MODELS_PATH, ATTACK_SETS,ATTACK_SETS_PV_CSV, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model,perturb_model, device, dataset_name, attack_name, num_classes, attack_dataset_batch_size, acc_gap, arch, flip_prob)
           
    
        
   
    attack.delete_pickle()
    attack.prepare_dataset_new() # uses shahow model to first obtain PVs and and the, combines it into [Pv, prediction, members, targets]
    attack.bind_cached_attack_loaders()
    if entropy_from_checkpoint:
        if os.path.exists(bundle_path):
            print(f"Using entropy checkpoint {bundle_path} to generate plots.")
            attack.compute_entropy_distribution_from_checkpoint(bundle_path, entropy_dis_dr)
            return
        else:
            print(f"Entropy checkpoint requested but not found at {bundle_path}; continuing with training.")
        

    epochs = 100
    tr_sum = 0.0
    ts_sum = 0.0

    res_list = []  # store final test metrics per epoch

    threshold_progress = []  # list of tuples (cosine_threshold, entropy_threshold)
    test_loss_progress = []  # list of average test losses

    # initialize the early_stopping object
    if dataset_name == "purchase":
        patience = 7
    elif dataset_name == "cifar10" and arch == "cnn":
        patience = 5
    else:
        patience = 15


    early_stopping = EarlyStopping(patience=patience, verbose=True)

    best_metric = float('inf')
    best_epoch = None
    best_checkpoint_path = ATTACK_PATH + "_best_attack_state.pt"

    for ep in range(epochs):
        flag = 1 if ep == (epochs - 1) else 0
        attack.current_epoch = ep + 1
        print("Epoch %d:" % (attack.current_epoch))

        res_train = attack.train(flag, RESULT_PATH, RESULT_PATH_csv, mode)
        res_test, fpr, tpr = attack.test(flag, RESULT_PATH, mode)

        current_cosine_threshold = torch.sigmoid(attack.cosine_threshold).item()
        current_entropy_threshold = torch.sigmoid(attack.Entropy_quantile_threshold).item()

        res_list.append({
            'epoch': attack.current_epoch,
            'test_acc': res_test[0]*100,
            'test_prec': res_test[1]*100,
            'test_recall': res_test[2]*100,
            'test_f1': res_test[3]*100,
            'test_auc': res_test[4]*100,
            'test_loss': res_test[-1],
            'cosine_threshold': current_cosine_threshold,
            'entropy_threshold': current_entropy_threshold,
        })

        early_stopping(res_test[-1], attack.attack_model)

        if res_test[-1] < best_metric:
            best_metric = res_test[-1]
            best_epoch = attack.current_epoch
            save_best_checkpoint(best_metric, attack, best_checkpoint_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    map_location = torch.device(device) if isinstance(device, str) else device
    best_checkpoint_available = os.path.exists(best_checkpoint_path)
    if best_checkpoint_available:
        checkpoint = torch.load(best_checkpoint_path, map_location=map_location)
        attack.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        attack.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])

        attack.cosine_threshold.data.copy_(checkpoint['cosine_threshold'].to(attack.cosine_threshold.device))
        attack.Entropy_quantile_threshold.data.copy_(checkpoint['entropy_threshold'].to(attack.Entropy_quantile_threshold.device))

        best_epoch = checkpoint.get('epoch', best_epoch)
        print(f"Loaded best checkpoint from epoch {best_epoch} with validation metric {best_metric:.4f}")
    else:
        print("Warning: no checkpoint saved during training; using final epoch parameters.")

    best_cosine_threshold_param = attack.cosine_threshold.detach().cpu()
    best_entropy_threshold_param = attack.Entropy_quantile_threshold.detach().cpu()
    best_cosine_threshold = torch.sigmoid(best_cosine_threshold_param).item()
    best_entropy_threshold = torch.sigmoid(best_entropy_threshold_param).item()
    print(
        f"Best Cosine Threshold: {best_cosine_threshold:.4f}, "
        f"Best Entropy Threshold: {best_entropy_threshold:.4f}"
    )

    df = pd.DataFrame(res_list)
    df['best_epoch'] = best_epoch
    df['best_metric'] = best_metric
    file_path = ATTACK_SETS + f"_Results-Mean_mode-{attack_name}_.csv"
    df.to_csv(file_path, index=False)

    if best_checkpoint_available:
        attack.save_att_per_thresholds_models(best_checkpoint_path, MODELS_PATH_att_per_thr)
        print(f'models and thresholds are savedQ')
    else:
        print('Skipping attack/threshold export; best checkpoint unavailable.')

    best_cosine_threshold_raw = best_cosine_threshold_param.item()
    best_entropy_threshold_raw = best_entropy_threshold_param.item()

    fpr = tpr = thresholds = roc_auc = None

    if attack_name == "apcmia":
        print(f'computing ROC for apcmia')
        attack.test_saved_model_apcmia(
            attack.attack_model,
            attack.perturb_model,
            best_cosine_threshold_raw,
            best_entropy_threshold_raw,
        )
        fpr, tpr, thresholds, roc_auc = attack.compute_roc_curve_apcmia(
            attack.attack_model,
            attack.perturb_model,
            best_cosine_threshold_raw,
            best_entropy_threshold_raw,
        )
        attack.compute_entropy_distribution_new_norm(
            attack.attack_model,
            attack.perturb_model,
            best_cosine_threshold_raw,
            best_entropy_threshold_raw,
            entropy_dis_dr,
        )
    else:  # computing ROC for rest of the attacks
        attack.test_saved_model_rest(attack.attack_model)
        fpr, tpr, thresholds, roc_auc = attack.compute_roc_curve_rest(attack.attack_model)

    if fpr is not None and tpr is not None:
        df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        df_fpr_tpr.to_csv(fpr_tpr_file_path, index=False)
        print(f"saved ROC curve info")
    else:
        print('ROC curve was not computed; skipping CSV export.')


    if attack_name == "apcmia" and apcmia_cluster:

          # 1. Create the root directory "cluster_results" if it doesn't exist.
        cluster_root = f"cluster_results/{arch}/"
        if not os.path.exists(cluster_root):
            os.makedirs(cluster_root)

        # 2. Create a subdirectory for this dataset (e.g., "test") inside "cluster_results".
        sub_folder = os.path.join(cluster_root, dataset_name)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        for target_class in range(num_classes):
            attack.visualize_transformed_pvs_classwise(target_class, attack.attack_model, attack.perturb_model, best_cosine_threshold_raw, best_entropy_threshold_raw, sub_folder)
