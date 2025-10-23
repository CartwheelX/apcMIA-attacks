import os
import torch
import pandas
import torchvision
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.autograd import Variable
from torch.utils.data import random_split, ConcatDataset
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
torch.manual_seed(0)
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import pickle


class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNN_rmia(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN_rmia, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5),   # 64 -> 60
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 60 -> 30
            nn.Conv2d(6, 16, kernel_size=5),              # 30 -> 26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 26 -> 13
        )

        # For 64x64 input: out channels=16, H=W=13  =>  16*13*13 = 2704
        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

     
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable
import functools

BN_MOM = 0.9
BN_EPS  = 1e-5

class WRNBlock(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        super().__init__()
        self.proj_conv = None
        if nin != nout or stride > 1:
            self.proj_conv = nn.Conv2d(nin, nout, kernel_size=1, stride=stride, padding=0, bias=False)

        self.norm_1 = bn(nin)
        self.conv_1 = nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_2 = bn(nout)
        self.conv_2 = nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        o1 = torch.relu(self.norm_1(x))
        y  = self.conv_1(o1)
        o2 = torch.relu(self.norm_2(y))
        z  = self.conv_2(o2)
        return z + (self.proj_conv(o1) if self.proj_conv is not None else x)
# -------------------------------------------------
# WideResNet core
# -------------------------------------------------
class WRNBlock(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        super().__init__()
        self.proj_conv = None
        if nin != nout or stride > 1:
            self.proj_conv = nn.Conv2d(nin, nout, kernel_size=1, stride=stride, padding=0, bias=False)

        self.norm_1 = bn(nin)
        self.conv_1 = nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_2 = bn(nout)
        self.conv_2 = nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        o1 = torch.relu(self.norm_1(x))
        y  = self.conv_1(o1)
        o2 = torch.relu(self.norm_2(y))
        z  = self.conv_2(o2)
        return z + (self.proj_conv(o1) if self.proj_conv is not None else x)
class WideResNetGeneral(nn.Module):
    def __init__(
        self,
        nin: int,
        nclass: int,
        blocks_per_group: List[int],
        width: int,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        super().__init__()
        widths = [int(v * width) for v in [16 * (2**i) for i in range(len(blocks_per_group))]]

        n = 16
        ops = [nn.Conv2d(nin, n, kernel_size=3, padding=1, bias=False)]
        for i, (block, w) in enumerate(zip(blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            ops.append(WRNBlock(n, w, stride, bn))
            for _ in range(1, block):
                ops.append(WRNBlock(w, w, 1, bn))
            n = w

        # tail
        ops += [
            bn(n),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n, nclass),
        ]
        self.layers = nn.Sequential(*ops)

    def forward(self, x):
        return self.layers(x)
class WideResNet(WideResNetGeneral):
    def __init__(
        self,
        nin: int,
        nclass: int,
        depth: int = 28,
        width: int = 2,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        assert (depth - 4) % 6 == 0, "depth should be 6n+4 (e.g., 16, 22, 28, 40)"
        n = (depth - 4) // 6
        blocks_per_group = [n, n, n]
        super().__init__(nin, nclass, blocks_per_group, width, bn)

# -------------------------------------------------
# Compatibility wrapper (matches your CNN init)
# -------------------------------------------------
class WideResNet_CNNStyle(WideResNet):
    """
    Same API as your other CNNs:
      __init__(input_channel=..., num_classes=..., depth=..., width=...)
    Forward returns logits [B, num_classes].
    """
    def __init__(
        self,
        input_channel: int = 3,
        num_classes: int = 10,
        depth: int = 28,
        width: int = 2,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        super().__init__(nin=input_channel, nclass=num_classes, depth=depth, width=width, bn=bn)


# vanilla CNN 
class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# Advanced VGG style CNN 
class AdvancedCNN_CNNStyle(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super().__init__()
        # --- features ---
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16 (or 64->32)

            nn.Conv2d(128,256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(256,512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Force a fixed 4x4 map so FC dims are stable for 32x32 or 64x64
            nn.AdaptiveAvgPool2d(4),
        )

        # 512 * 4 * 4 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 8192), nn.ReLU(inplace=True),
            nn.Linear(8192, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class simpleNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_p: float = 0.5, use_layernorm: bool = True, input_dropout_p: float = 0.0):
        super().__init__()
        # Widen hidden layers for more capacity on Location
        hidden1, hidden2 = 256, 128
        norm1 = nn.LayerNorm(hidden1) if use_layernorm else nn.BatchNorm1d(hidden1)
        norm2 = nn.LayerNorm(hidden2) if use_layernorm else nn.BatchNorm1d(hidden2)
        self.net = nn.Sequential(
            nn.Dropout(input_dropout_p),
            nn.Linear(input_size, hidden1),
            norm1,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(hidden1, hidden2),
            norm2,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
    
class simpleNN_Target_purchase(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN_Target_purchase, self).__init__()
        
        self.classifier = nn.Sequential(
            
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),      
            nn.Linear(100, num_classes),
            
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x

class simpleNN_Target_texas(nn.Module):
    def __init__(self, input_size, num_classes=100):
        super(simpleNN_Target_texas, self).__init__()
     
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Continue replacing BatchNorm1d with LayerNorm or GroupNorm as needed
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes)
        )

    def forward(self, X_Batch):
        return self.classifier(X_Batch)
    
class simpleNN_Shaddow_purchase(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN_Shaddow_purchase, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Dropout with probability 50%
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x

class Adult(nn.Module):
    def __init__(self, input_size=12, num_classes=2):
        super(Adult, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, X_Batch):
        return self.classifier(X_Batch)

class UTKFaceDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        
        
        self.files = os.listdir(self.processed_path)
        print("self.root: ", self.root)    
        print("in the UTKFace dataset class constructor", self.processed_path)
        print("self files: ", self.files)
        # exit()
        
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            txt_file_path = os.path.join(self.processed_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])
        # print("in the __getitem__ method")
        
        image_path = os.path.join(self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

# Define the Perturbation Model
class PerturbationModel(nn.Module):
    def __init__(self, class_num,  device,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=64):
        super(PerturbationModel, self).__init__()

        self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.Linear(1, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
          
		)
        self.model = nn.Sequential(
            # nn.Linear(class_num, 128),  # First hidden layer with 64 neurons
            # nn.ReLU(),                 # Activation function
            # nn.BatchNorm1d(128),        # Batch normalization
            
            nn.Linear(class_num, 256),         # Second hidden layer with 32 neurons
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            
            nn.Linear(256, 128),         # Second hidden layer with 32 neurons
            nn.ReLU(),

            nn.Linear(128, 64),         # Second hidden layer with 32 neurons
            nn.ReLU(),
            # nn.BatchNorm1d(32),

            nn.Linear(64, 32),         # Second hidden layer with 32 neurons
            nn.ReLU(),

            nn.Linear(32, class_num), # Output layer matches input size
            nn.Sigmoid()               # Sigmoid ensures perturbation values are bounded (0, 1)
        )
    
    def forward(self, PV_batch, target_label_batch):
        # Ensure target_label_batch is a float tensor with shape (batch_size, 1)
        # target_label_batch = target_label_batch.float().unsqueeze(1)
        
        # Process target_label_batch through the Prediction Component
        # pred_component = self.Prediction_Component(target_label_batch)
        
        # Concatenate the PV_batch and the prediction component result along the feature dimension
        # combined_features = torch.cat([PV_batch, pred_component], dim=1)
        
        return self.model(PV_batch)
        # Forward the concatenated features through the model
        # return self.model(torch.cat([PV_batch, self.Prediction_Component(target_label_batch.float().unsqueeze(1))], dim=1))
    

class AttackDataset(Dataset):
    def __init__(self, pickle_path):
        outputs = []
        predictions = []
        members = []
        targets = []
        with open(pickle_path, 'rb') as f:
            while True:
                try:
                    # Each pickle load returns a batch of (output, prediction, members, targets)
                    out, pred, mem, targ = pickle.load(f)
                    outputs.append(out)
                    predictions.append(pred)
                    members.append(mem)
                    targets.append(targ)
                except EOFError:
                    break
        # Concatenate all batches along the first dimension
        self.outputs = torch.cat(outputs, dim=0)
        self.predictions = torch.cat(predictions, dim=0)
        self.members = torch.cat(members, dim=0)
        self.targets = torch.cat(targets, dim=0)

    def __len__(self):
        return self.outputs.size(0)

    def __getitem__(self, idx):
        return self.outputs[idx], self.predictions[idx], self.members[idx], self.targets[idx]

class CombinedShadowAttackModel_NEW(nn.Module):
    def __init__(self, class_num,  device, mode, attack_name,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=64):
        
        super(CombinedShadowAttackModel_NEW, self).__init__()
        
       
        self.input_dim = class_num
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = device

        self.mode = mode
        self.attack_name = attack_name
        
    
        
        self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
            # nn.ReLU(),
			# nn.Linear(256, 64),
		)
    
        
        self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.Linear(1, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
          
		)

        
        self.Encoder_Component = nn.Sequential(
			nn.Linear(class_num+64, 512), #mia_actual
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        
        self.pertubed_attack_Component = nn.Sequential(
        nn.Linear(class_num+64, 512), 
        nn.ReLU(),
        # nn.BatchNorm1d(512),

        nn.Linear(512, 256),
        nn.ReLU(),
        # nn.BatchNorm1d(256),
        
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        # nn.BatchNorm1d(128)
        
    )

   
    
   
    def pertubed_attack(self, output, prediction):
        Prediction_Component_result = self.Prediction_Component(prediction)
        return self.pertubed_attack_Component(torch.cat((Prediction_Component_result, output), 1))
    

    def get_embeddings(self, output, prediction, label):
        """
        For the apcmia attack, returns feature embeddings for contrastive loss computation.
        Here, we first compute the concatenated features (Prediction_Component output concatenated with output),
        then run them through all but the final layer of the pertubed_attack_Component.
        """
        if self.attack_name == "apcmia":
            Prediction_Component_result = self.Prediction_Component(prediction)
            features = torch.cat((Prediction_Component_result, output), 1)
            # Get a list of layers from the sequential model:
            layers = list(self.pertubed_attack_Component.children())
            # Run through all layers except the final one:
            for layer in layers[:-1]:
                features = layer(features)
            return features  # This is the embedding representation
        else:
            raise NotImplementedError("get_embeddings is implemented only for apcmia attack.")
            
    def forward(self, output, prediction, label):
       
        if self.attack_name == "apcmia":
            return self.pertubed_attack(output, prediction)
    

            



class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class SmallSTLNet(nn.Module):
    """
    Overfit-resistant CNN for STL-10.
    - Uses padding to keep spatial sizes stable within a stage.
    - Dropout2d after each downsample to regularize.
    - GAP head avoids large fully-connected layers.
    """
    def __init__(self, input_channel=3, num_classes=10, drop_p=(0.1, 0.2, 0.3)):
        super().__init__()
        # Stage 1: 96x96 -> 48x48
        self.stage1 = nn.Sequential(
            ConvBNAct(input_channel, 64),
            ConvBNAct(64, 64),
            nn.MaxPool2d(2),          # downsample
            nn.Dropout2d(drop_p[0]),
        )
        # Stage 2: 48x48 -> 24x24
        self.stage2 = nn.Sequential(
            ConvBNAct(64, 128),
            ConvBNAct(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop_p[1]),
        )
        # Stage 3: 24x24 -> 12x12
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256),
            ConvBNAct(256, 256),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop_p[2]),
        )
        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)    # -> (B, C, 1, 1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
    

class CNN_STL10(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN_STL10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),           nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 256),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          # <— add this
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    

class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd
import math

class VGG16_new(nn.Module):
    def __init__(self,input_channel, num_classes ):
        super(VGG16_new, self).__init__()

        self.input_size = 64
        self.num_classes = num_classes
        self.conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.fc_layer_sizes = [512, 512]

        self.max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        self.conv_batch_norm = True
        self.init_weights = True
        self.augment_training = False
        self.num_output = 1

        

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        # input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

            
class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
import torch.nn.init as init
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class Wide_ResNet28(nn.Module):
    def __init__(self, num_classes):
        super(Wide_ResNet28, self).__init__()
        self.in_planes = 16
        dropout_rate = 0.3
        depth = 4
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = 10

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        # self.linear = nn.Linear(nStages[3], num_classes)
        self.linear = nn.Linear(nStages[3] * 4, num_classes)



    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out          
            

            
