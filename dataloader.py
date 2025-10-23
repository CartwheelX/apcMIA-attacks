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

from target_shadow_nn_models import *


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import rcParams

# Main Library
import torch 
from torch import nn

torch.manual_seed(0)



class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"

    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

def Location(num_classes):
    
    dataset = []
    locations = []
    labels = []
    output_coll =  torch.empty((0, num_classes))
    
    file_path = "data/location/bangkok"

    if os.path.exists(file_path):
        print(f"yes")
    else:
        print(f"No.")
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Split the line by commas
            if len(parts) >= 2:
                # Convert the first part to an integer (label)
                cleaned_string = parts[0].replace('"', '')
                label = int(cleaned_string)
                location = [int(token) for token in parts[1:]]  # Convert the rest to integers (locations)
                # print(len(location))
                labels.append(label)
                locations.append(location)
            # break

    # Convert lists to NumPy arrays
    Y = np.array(labels)
    Y = Y.reshape(-1, 1)
    Y = Y - 1
    # print(labels_array.shape)
    X = torch.tensor(locations, dtype=torch.float)
   
    
    for i in range(X.size()[0]):
        dataset.append((X[i], Y[i].item()))
    
    
    return dataset



def texas(num_classes):
    # Path to store the processed dataset.
    processed_file_path = "data/texas/processed_dataset.pt"
    
    # If the processed file exists, load it directly.
    if os.path.exists(processed_file_path):
        print("Loading dataset from processed file.")
        dataset = torch.load(processed_file_path, weights_only=True)
        # print("Sample labels from the dataset:")
        # for i in range(min(10, len(dataset))):  # Print up to 10 samples
        #     print(dataset[i][1])
        return dataset

    # If not, process the raw data files.
    dataset = []
    feats = []
    labels = []

    feats_file_path = "data/texas/feats"
    labels_file_path = "data/texas/labels"

    # Check for existence of both raw files.
    if not os.path.exists(feats_file_path):
        print("Feats file does not exist.")
        return None
    if not os.path.exists(labels_file_path):
        print("Labels file does not exist.")
        return None

    # Read features from the feats file.
    with open(feats_file_path, 'r') as f:
        for line in f:
            # Assume each line is a comma-separated list of numbers.
            tokens = line.strip().split(',')
            feature_vector = [float(token) for token in tokens]
            feats.append(feature_vector)

    # Read labels from the labels file.
    with open(labels_file_path, 'r') as f:
        for line in f:
            # Each line is expected to have one label.
            label = int(line.strip())
            labels.append(label)

    # Convert features to a PyTorch tensor.
    X = torch.tensor(feats, dtype=torch.float)
    # Convert labels to a NumPy array and adjust shape.
    Y = np.array(labels).reshape(-1, 1)
    # Adjust labels to be zero-indexed (if needed).
    Y = Y - 1

    # Create the dataset as a list of tuples.
    for i in range(X.size(0)):
        dataset.append((X[i], Y[i].item()))

    # Save the processed dataset to disk for faster future loading.
    torch.save(dataset, processed_file_path)
    print("Processed dataset saved to disk.")

    return dataset


import random

def texas_random(num_classes, random_sample_size=40000):
    """
    Loads the Texas dataset. On the first run, it processes the raw data,
    takes a random sample of 40k samples (if available), and saves this subset.
    On subsequent runs, the function loads the stored random sample directly.
    
    Args:
        num_classes (int): Number of classes (can be used for further processing).
        random_sample_size (int): Number of random samples to extract.
    
    Returns:
        list: A list of tuples (feature_tensor, label).
    """
    # File path for the random sample dataset.
    random_sample_file_path = "data/texas/random_40k_dataset.pt"
    
    # If the random sample file exists, load and return it.
    if os.path.exists(random_sample_file_path):
        print("Loading random 40k dataset from disk.")
        return torch.load(random_sample_file_path)
    
    # Process the raw data if the random sample file does not exist.
    dataset = []
    feats = []
    labels = []
    feats_file_path = "data/texas/feats"
    labels_file_path = "data/texas/labels"
    
    # Ensure that both raw data files exist.
    if not os.path.exists(feats_file_path):
        print("Feats file does not exist.")
        return None
    if not os.path.exists(labels_file_path):
        print("Labels file does not exist.")
        return None

    # Read features from the feats file.
    with open(feats_file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            # Convert each token to float (or int if desired)
            feature_vector = [float(token) for token in tokens]
            feats.append(feature_vector)
    
    # Read labels from the labels file.
    with open(labels_file_path, 'r') as f:
        for line in f:
            label = int(line.strip())
            labels.append(label)
    
    # Convert the features and labels to tensors/arrays.
    X = torch.tensor(feats, dtype=torch.float)
    Y = np.array(labels).reshape(-1, 1)
    # Adjust labels to zero-indexed, if needed.
    Y = Y - 1
    
    # Create the dataset as a list of (feature, label) tuples.
    for i in range(X.size(0)):
        dataset.append((X[i], Y[i].item()))
    
    # If the dataset has fewer samples than requested, use the full dataset.
    if len(dataset) < random_sample_size:
        print("Dataset size is less than the requested random sample size.")
        random_dataset = dataset
    else:
        random_dataset = random.sample(dataset, random_sample_size)
    
    # Save the random sample to disk so that it can be reused.
    torch.save(random_dataset, random_sample_file_path)
    print("Random 40k dataset saved to disk.")
    
    return random_dataset

def Purchase(num_classes):
    
    
    dataset = []
    purchase_feats = []
    purchase_labels = []
    output_coll =  torch.empty((0, num_classes))
    

    file_path = "data/purchase/purchase"
    if os.path.exists(file_path):
        print(f"yes")
    else:
        print(f"No.")
    

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Split the line by commas
            if len(parts) >= 2:
               
                purchase_label = int(parts[0])
                # print(f"purchase_label: {type(purchase_label)}")
                
                purchase_feat = [int(token) for token in parts[1:]]  # Convert the rest to integers (locations)
            
                purchase_labels.append(purchase_label)
                purchase_feats.append(purchase_feat)
    
    # Convert lists to NumPy arrays
    Y = np.array(purchase_labels)
    Y = Y.reshape(-1, 1)
    Y = Y-1
    
    X = torch.tensor(purchase_feats, dtype=torch.float)
   
    for i in range(X.size()[0]):
        dataset.append((X[i], Y[i].item()))
    
    
    print(f"dataSet size: {Y[:19]}")
    print(f"type {type(dataset)}")
    print(f"size of dataset: {len(dataset)}")
    
    # exit()

    return dataset

def find_features_with_missing_values(df):
    # Replace characters with NaN
    replace_chars = ["\n", "\n?\n", "?","\n?"," ?","? "," ? "," ?\n"]
    if any(char in df.values for char in replace_chars):
        df.replace(replace_chars, np.nan, inplace=True)
        print("Successfully replaced characters with NaN.")
    
    # Find features with missing values
    features_with_null = [feature for feature in df.columns if df[feature].isnull().sum() > 0]
    if not features_with_null:
        print("No missing values found in any features.")
    else:
        for feature in features_with_null:
            print(f"{feature}: {round(df[feature].isnull().mean() * 100, 2)}%")
    return features_with_null

def adult(num_classes):
    # Setup device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")
    
    # Load CSV
    adult_file = './data/adult/adult.csv'
    df_0 = pd.read_csv(adult_file)
    df = df_0.copy()
    
    # Check and replace missing values
    find_features_with_missing_values(df)
    df = df.fillna("Missing")
    
    # Rename the columns
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                  'income']
    
    # Visualize missing 'native-country' values for income distribution (optional)
    print(df[df["native-country"] == "Missing"]["income"].value_counts(normalize=True))
    
    # Clean up occupation and recategorize
    df['occupation'] = df['occupation'].str.strip()
    df['new_occupation'] = df['occupation'].replace({
        'Prof-specialty': 'Professional_Managerial',
        'Craft-repair': 'Skilled_Technical',
        'Exec-managerial': 'Professional_Managerial',
        'Adm-clerical': 'Sales_Administrative',
        'Sales': 'Sales_Administrative',
        'Other-service': 'Service_Care',
        'Machine-op-inspct': 'Skilled_Technical',
        'Missing': 'Unclassified Occupations',
        'Transport-moving': 'Skilled_Technical',
        'Handlers-cleaners': 'Service_Care',
        'Farming-fishing': 'Service_Care',
        'Tech-support': 'Skilled_Technical',
        'Protective-serv': 'Professional_Managerial',
        'Priv-house-serv': 'Service_Care',
        'Armed-Forces': 'Unclassified Occupations',
    })
    df.drop(['occupation'], axis=1, inplace=True)
    print(f"New narrowed categories : \n{df['new_occupation'].value_counts()}")
    
    # Set proper data types
    data_types = {'age': 'uint8',
                  'workclass': 'category',
                  'fnlwgt': 'int32',
                  'education': 'category',
                  'education-num': 'uint8',
                  'marital-status': 'category',
                  'new_occupation': 'category',
                  'relationship': 'category',
                  'race': 'category',
                  'sex': 'category',
                  'capital-gain': 'int32',
                  'capital-loss': 'int32',
                  'hours-per-week': 'uint8',
                  'native-country': 'category',
                  'income': 'category'}
    df = df.astype(data_types)
    
    # Drop columns not used for modeling
    df.drop(['education'], axis=1, inplace=True)
    df.drop(['native-country'], axis=1, inplace=True)
    
    # # Visualize a KDE plot for a numeric column (age) by income
    # sns.kdeplot(data=df, x='age', hue="income", multiple="stack")
    # plt.show()
    print("done")
    
    rcParams['figure.figsize'] = 8, 8
    df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].hist()
    # plt.show()
    
    sns.countplot(x='income', hue='new_occupation', data=df)
    # plt.show()
    # df.to_csv('df_output.csv', index=False)
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    df_encoded = df.copy()  # Copy the DataFrame for encoding
    
    categorical_columns = ['workclass', 'marital-status', 'new_occupation', 'relationship', 'race', 'sex', 'income']
    for column in categorical_columns:
        df_encoded[column] = label_encoder.fit_transform(df[column])
    
    # Normalize continuous features
    continuous_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    df_encoded[continuous_columns] = scaler.fit_transform(df_encoded[continuous_columns])
    
    # Save the processed DataFrame (optional)
    df_encoded.to_csv('df_output.csv', index=False)
    print("DataFrame saved to output.csv")
    
    # Split into features and target
    X = df_encoded.drop(['income'], axis=1)
    y = df_encoded['income']
    X, y = X.to_numpy(), y.to_numpy()
    
    # Convert features to torch tensor (float) and create dataset as list of tuples
    X = torch.from_numpy(X).type(torch.float)
    dataset = []
    for i in range(X.size(0)):
        dataset.append((X[i], int(y[i])))
    
    print(f"Dataset target sample: {y[:19]}")
    print(f"Type: {type(dataset)}")
    print(f"Dataset size: {len(dataset)}")
    
    return dataset

def prepare_dataset(dataset_name, attr, root, device, arch, DSize):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(
        dataset_name, device, arch, attr=attr, root=root
    )
    length = len(dataset)
    print(f"Dataset {dataset_name},  shape: {len(dataset)} samples")

    # Use a stratified 50/50 split to preserve class distribution in train/test.
    indices = np.arange(length)
    labels = []
    for idx in indices:
        _, target = dataset[idx]
        # handle tuple/list targets as used by some datasets
        if isinstance(target, (list, tuple)):
            target = target[0]
        if torch.is_tensor(target):
            target = target.item()
        labels.append(int(target))

    train_idx, test_idx = train_test_split(
        indices,
        train_size=0.5,
        stratify=labels,
        random_state=42,
    )

    target_train = torch.utils.data.Subset(dataset, train_idx.tolist())
    target_test = torch.utils.data.Subset(dataset, test_idx.tolist())

    shadow_train = torch.utils.data.Subset(dataset, train_idx.tolist())
    shadow_test = torch.utils.data.Subset(dataset, test_idx.tolist())
 
    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model


def get_model_dataset(dataset_name, device, arch, attr, root):
    print(f"device in get_model_dataset: {device}")
    # exit()
    if dataset_name.lower() == "utkface": # OK for all arch
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print("attributes: ", attr)
        
        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)
        input_channel = 3
        

    elif dataset_name.lower() == "stl10": # only van_cnn and Advcnn, VGG16 maybe, rmia not here
        num_classes = 10

        if arch.lower() == 'van_cnn':
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_tf = transform
            test_tf = transform

        elif arch.lower() == "cnn":
            
            train_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Ajustes aleatorios de brillo, contraste, saturación y tono
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])

            test_tf = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        elif arch.lower() == "wrn_rmia": # 
                
                train_tf = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Ajustes aleatorios de brillo, contraste, saturación y tono
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ])

                test_tf = transforms.Compose([
                    transforms.ToTensor(),
                ])

        else: # maybe for vgg16 revert back to orioginal transform

                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
                
                train_tf = transform
                test_tf = transform
            
        train_set = torchvision.datasets.STL10(
                root=root, split='train', transform=train_tf, download=True)
            
        test_set = torchvision.datasets.STL10(
                root=root, split='test', transform=test_tf, download=True)
        
        dataset = train_set + test_set
        input_channel = 3
        print(f"size of STL10 dataset: {len(train_set), len(test_set)}")
        
        # exit()
    
    elif dataset_name.lower() == "cifar10": # OK for all arch
        # root = "./data"
        print(f"CIFAR10")
        
        num_classes = 10
        if arch.lower() == 'van_cnn':

            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            train_tf = transform
            test_tf = transform

        elif arch.lower() == "cnn":
            
            train_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Ajustes aleatorios de brillo, contraste, saturación y tono
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])

            test_tf = transforms.Compose([
                transforms.ToTensor(),
            ])

        elif arch.lower() == "wrn_rmia":

                train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616)),
                ])
                test_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2470, 0.2435, 0.2616)),
                ])

        else: # maybe for vgg16 revert back to orioginal transform

                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
                
                train_tf = transform
                test_tf = transform

        train_set = torchvision.datasets.CIFAR10(
                root=root, train=True, transform=train_tf, download=True)
        test_set = torchvision.datasets.CIFAR10(
                root=root, train=False, transform=test_tf, download=True)

        dataset = train_set + test_set
        input_channel = 3
        print(f"size of CIFAR10 dataset: {len(train_set), len(test_set)} and T dataSize: {len(dataset)}")
        img, label = dataset[0]
        # print(dataset[1])
        print(f"type of cifar dataset: {type(train_set)}")
        # exit()
    
    elif dataset_name.lower() == "cifar100": # OK for all arch
        
        print(f"CIFAR100")
        
        num_classes = 100
        if arch.lower() == 'van_cnn':
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_tf = transform
            test_tf = transform    

        elif arch.lower() == "cnn":
            
            train_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Ajustes aleatorios de brillo, contraste, saturación y tono
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])

            test_tf = transforms.Compose([
                transforms.ToTensor(),
            ])

        elif arch.lower() == "wrn_rmia":

                train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                    (0.2675, 0.2565, 0.2761)),
                ])
                test_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761)),
                ])
                

        else: # maybe for vgg16 revert back to orioginal transform

                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
                
                train_tf = transform
                test_tf = transform
    

        train_set = torchvision.datasets.CIFAR100(
                root=root, train=True, transform=train_tf, download=True)
        test_set = torchvision.datasets.CIFAR100(
                root=root, train=False, transform=test_tf, download=True)

        dataset = train_set + test_set
        input_channel = 3
        print(f"size of CIFAR100 dataset: {len(train_set), len(test_set), len(dataset)}")
   
    elif dataset_name.lower() == "fmnist": # OK for all arch
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
                root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
                root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 1
        print(f"size of FMNIST dataset: {len(train_set), len(test_set)}")
        # exit()
   
    elif dataset_name.lower() == "imagenet": # OK for all arch
        # print(f"Imagenet")
        
        from torchvision.datasets import Imagenette
        from torchvision.models import ResNet50_Weights

        root = root
        weights = ResNet50_Weights.IMAGENET1K_V2
        tf = weights.transforms()
        train_ds = Imagenette(root, split="train", download=False, transform=tf)
        test_ds = Imagenette(root, split="val", download=False, transform=tf)
        dataset = train_ds
        num_classes = 10
        input_channel = 3
        print(f"size of Imagenet dataset: {len(train_ds), len(test_ds), len(dataset)}")
        exit()

    elif dataset_name.lower() == "country":
        
        print(f"country211")
        
        num_classes = 100

        # Adjust the transformations as needed for your task.
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images to 256x256 pixels.
            transforms.ToTensor(),          # Convert images to PyTorch tensors.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means and stds.
                                std=[0.229, 0.224, 0.225]),
        ])

       
        train_set = torchvision.datasets.Country211(root=root, split='train', transform=transform, download=True)
        test_set = torchvision.datasets.Country211(root=root, split='test', transform=transform, download=True)
        val_set = torchvision.datasets.Country211(root=root, split='valid', transform=transform, download=True)

        print(len(train_set.classes)) 

        dataset = train_set + test_set + val_set
        input_channel = 3
        print(f"size of country dataset: {len(train_set), len(test_set), len(val_set), len(dataset)}")
        exit()
    
    elif dataset_name.lower() == "location":
        
        print(f"Location dataset")
        
        num_classes = 30
        dataset = Location(num_classes)
        first_sample, _ = dataset[0]
        input_size = len(first_sample)
        print(f"size of location dataset: {len(first_sample)}")

    elif dataset_name.lower() == "purchase":
        num_classes = 100
        print(f"purchase dataset")
        dataset = Purchase(num_classes)
        first_sample, _ = dataset[0]
        input_size = len(first_sample)
        print(f"input size of purchase dataset: {input_size}")
        # exit()
   
    elif dataset_name.lower() == "texas":
        num_classes = 100
        print(f"Texas dataset")
        dataset = texas(num_classes)
        first_sample, _ = dataset[0]
        input_size = len(first_sample)
        print(f"input size of Texas dataset: {input_size}")
        # exit()

    elif dataset_name.lower() == "adult":
        num_classes = 2
        print(f"Texas dataset")
        dataset = adult(num_classes)
        first_sample, _ = dataset[0]
        input_size = len(first_sample)
        print(f"input size of Texas dataset: {input_size}")
        # exit()

      

    if isinstance(num_classes, int):
        classes = num_classes
    else:
        classes = num_classes[0]

    if arch.lower() == 'mlp':    
        if dataset_name.lower() == "location":
            # Use LayerNorm + moderate dropout and slight input dropout for tabular features
            target_model = simpleNN(input_size=input_size, num_classes=classes, dropout_p=0.25, use_layernorm=True, input_dropout_p=0.05)
            shadow_model = simpleNN(input_size=input_size, num_classes=classes, dropout_p=0.25, use_layernorm=True, input_dropout_p=0.05)
        elif dataset_name.lower() == "purchase":
            target_model = simpleNN_Target_purchase(input_size=input_size, num_classes=classes)
            shadow_model = simpleNN_Shaddow_purchase(input_size=input_size, num_classes=classes)
        elif dataset_name.lower() == "texas":
            target_model = simpleNN_Target_texas(input_size=input_size, num_classes=classes)
            shadow_model = simpleNN_Target_texas(input_size=input_size, num_classes=classes)
        elif dataset_name.lower() == "adult":
            target_model = Adult(input_size=input_size, num_classes=classes)
            shadow_model = Adult(input_size=input_size, num_classes=classes)
    else:
        if arch.lower() == 'vgg16':
            print("getting vgg16 mode")
            # exit()
            target_model = VGG16_new(input_channel=input_channel, num_classes=classes)
            shadow_model = VGG16_new(input_channel=input_channel, num_classes=classes)
        elif arch.lower() == 'cnn':
            print("getting AdvancedCNN model")
            # exit()
            target_model = AdvancedCNN_CNNStyle(input_channel=input_channel, num_classes=classes)
            shadow_model = AdvancedCNN_CNNStyle(input_channel=input_channel, num_classes=classes)
        elif arch.lower() == 'van_cnn':
            print("getting van_cnn model")
            # exit()
            target_model = CNN(input_channel=input_channel, num_classes=classes)
            shadow_model = CNN(input_channel=input_channel, num_classes=classes)
        elif arch.lower() == 'wrn_rmia':
            print("getting wrn_rmia model")

            if dataset_name.lower() == 'fmnist':
                target_model =  WideResNet_CNNStyle(input_channel=1, num_classes=num_classes, depth=28, width=2)
                shadow_model =  WideResNet_CNNStyle(input_channel=1, num_classes=num_classes, depth=28, width=2)
            else:
                target_model =  WideResNet_CNNStyle(input_channel=3, num_classes=num_classes, depth=28, width=2)
                shadow_model =  WideResNet_CNNStyle(input_channel=3, num_classes=num_classes, depth=28, width=2)

    shadow_model = target_model
    return num_classes, dataset, target_model, shadow_model

# thigs in need add capability for, right 
