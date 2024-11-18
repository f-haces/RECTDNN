# PYTHON IMPORTS
import os, copy, traceback
from tqdm.notebook import trange, tqdm
import glob

# IMAGE IMPORTS 
from PIL import Image
import cv2

# DATA IMPORTS 
import numpy as np

# PLOTTING
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# NEURAL NETWORK
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage, GaussianBlur
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# MY OWN CLASSES
from TPNN import *
from PlottingUtils import *
from DataUtils import * 

Image.MAX_IMAGE_PIXELS = 933120000

class NN_Dataset_V3(Dataset):
    def __init__(self, image_dir, mask_dir, input_transform=None, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            transform (callable, optional): Transformation to apply to both images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))  # Ensure matching order
        self.mask_names = sorted(os.listdir(mask_dir))   # Ensure matching order
        self.transform = transform
        self.input_transform = input_transform

        # Preload all images and masks into memory
        self.images = []
        self.masks = []
        for fn in self.image_names:
            image = Image.open(os.path.join(image_dir, fn)).convert('L')
            image = Image.fromarray(np.array(image).astype(np.uint8))
            self.images.append(image)
                
        self.targets = loadClasses(mask_dir, fns=self.image_names, flip=False)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return preloaded image and mask
        image = self.images[idx]
        target_image = self.targets[idx]

        seed = np.random.randint(2147483647)            
        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)            
            random.seed(seed)
            torch.manual_seed(seed)
            target_image = self.transform(target_image)
        
        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, target_image, self.image_names[idx]


# In[3]:


base_dir = r"/project/glennie/fhacesga/RECTDNN/FLNN/tiled_master/"

input_folder        = f"{base_dir}/in"
val_folder          = f"{base_dir}/val"
train_target_folder = f"{base_dir}/out"
val_target_folder   = f"{base_dir}/val_out"
batch_size          = 20
verbose_level       = 2

tensor = transforms.Compose([
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

input_transform = transforms.RandomApply(torch.nn.ModuleList([
    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
]), 0.5)

tensor = transforms.Compose([transforms.ToTensor()])

train_dataset = NN_Dataset_V3(input_folder, train_target_folder, transform=transform, 
                              input_transform=input_transform)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataset = NN_Dataset_V3(val_folder, val_target_folder, transform=transform, 
                              input_transform=input_transform)
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

loaders = {'train' : train_loader, "test" : val_loader }


def notify(mess, level=4):
    if verbose_level >= level:
        print(mess, flush=True)

def saveImages(prob_img_or, outputs, filenames, minputs, outputs_folder, rep_id, labels,):
    
    # IF FOR SOME REASON OUTPUT IS UNBATCHED, BATCH IT
    if prob_img_or.ndim == 3:
        prob_img_or = prob_img_or.unsqueeze(0)
        
    # CONVERT TO NUMPY ARRAY
    prob_img_or = prob_img_or.numpy()
    
    # FOR EACH BATCHED OUTPUT
    for i in range(len(outputs)):
        filename = filenames[i]

        # CONVERT TO IMAGE NUMPY ARRAY
        prob_img = prob_img_or[i, :, :, :]
        prob_img = (prob_img * 255).astype(np.uint8) 
        prob_img = np.moveaxis(prob_img, [0, 1, 2], [2, 0, 1])

        # SAVE TO FILE
        myout = probability_to_rgb(prob_img)
        myout.save(os.path.join(outputs_folder, f"{rep_id}_{filename[:-4]}_out.png"))

        # SAVE INPUTS
        myinp = Image.fromarray(np.uint8(minputs[i, 0, :, :] * 255))
        myinp.save(os.path.join(outputs_folder, f"{rep_id}_{filename[:-4]}_inp.png"))

        # SAVE OUTPUTS
        mylab = class_to_one_hot(labels[i, :, :], prob_img.shape[-1])
        mylab = probability_to_rgb(mylab)
        mylab.save(os.path.join(outputs_folder, f"{rep_id}_{filename[:-4]}_lab.png"))
            
def train(model, dataloaders, num_epochs=50, 
          output_dir=r'/project/glennie/fhacesga/RECTDNN/FLNN/intermediate/', 
          learning_rate=5e-4,
          device = torch.device("cuda:0"),
          continue_from=None,
          weights=[1, 100, 100]):
    
    # TRAINING PARAMETERS
    weights = torch.tensor(weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    start_epoch = 0
        
    if continue_from is not None:
        checkpoint = torch.load(continue_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']
        start_loss = checkpoint['loss']
        
        epoch_losses = checkpoint['epoch_losses']
        iou_lists    = checkpoint['iou_lists']
        lr_list      = checkpoint['lr_list']
        
    else:
         # STRUCTURES FOR PLOTTING
        epoch_losses = {'train' : [], 'test' : []}
        iou_lists    = {'train' : [], 'test' : []}
        lr_list      = []
        
    # SEND MODEL TO GPU
    model = model.to(device)
    
    # MAKE SURE DIRS FOR TEMP OUTPUTS EXIST
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig, axs = init_plotting()
        
    # LOOP THROUGH EPOCHS
    for epoch in range(start_epoch, num_epochs):
        # notify('Epoch {}/{}'.format(epoch+1, num_epochs), level=1)
        # notify('-' * 10, level=1)
        phases = ['train', 'test']
        # phases = ['test']
        epoch_loss = {'train' : 0, 'test' : 0}
        iou_list = {'train' : [], 'test' : []}      
        
        # FOR BOTH PHASES
        for phase in phases: 
            if phase == 'train':
                model.train()
                repeats = range(1)
            else:
                model.eval()
                repeats = range(1)
            
            # MAKE DIR FOR CURRENT PHASE IF IT DOES NOT EXIST
            outputs_folder = os.path.join(output_dir, phase)
            if not os.path.exists(outputs_folder):
                os.makedirs(outputs_folder)
                
            # BASELINE MEMORY USAGE                
            notify(f"Prior to Iterations\t {torch.cuda.memory_allocated() / 1e6}")
            
            # COUNT HOW MANY IMAGES
            n_images = 0
            
            # ITERATE OVER REPEATS
            for rep_id in tqdm(repeats, disable=verbose_level < 4):
                for i, (inputs, labels, filenames) in enumerate(dataloaders[phase]):
                    
                    # SEND TO GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels = labels[:, 0, :, :].long()
                    notify(f"Datasets Moved\t\t {torch.cuda.memory_allocated()/ 1e6}")
                    
                    # ZERO GRADIENTS AND THROUGH MODEL
                    optimizer.zero_grad()
                    outputs = model(inputs, resize=False)
                    
                    notify(f"Outputs Processed\t {torch.cuda.memory_allocated() / 1e6}")
                    
                    # CALCULATE LOSS AND KEEP TRACK OF IT FOR CURRENT EPOCH
                    """print(outputs.shape)
                    print(labels.shape)
                    print(outputs)
                    print(labels)"""
                    loss = criterion(outputs, labels) 
                    if phase is 'train':
                        epoch_loss[phase] += loss
                    else:
                        epoch_loss[phase] += loss.detach().cpu().numpy()
                        
                    iou_list[phase].append(calculate_iou(outputs, labels).detach().cpu().numpy())
                    notify(f"Loss Calculated\t\t {torch.cuda.memory_allocated() / 1e6}")
                    
                    # COUNT HOW MANY IMAGES
                    n_images += inputs.shape[0]
                    
                    # OPTIMIZE IF PHASE IS TRAINING
                    if phase is 'train':
                        notify("Optimizing")
                        loss.backward()
                        optimizer.step()
                        notify(f"Backwards and optimized\t {torch.cuda.memory_allocated() / 1e6}")
                    
                    
                    # SAVE TRAINING IMAGES IF CURRENT STEP REQUIRES IT
                    
                    # if rep_id % 2 == 0 and rep_id != 0:
                    if random.randrange(0,1000) < 2:
                        prob_img_or = outputs.detach().cpu()
                        minputs     = inputs.detach().cpu().numpy()
                        mlabels     = labels.detach().cpu().numpy()
                        saveImages(prob_img_or, outputs, filenames, minputs, outputs_folder, rep_id, mlabels)
                        
                    # EMPTY CACHE
                    torch.cuda.empty_cache()
                    notify(f"Cache Emptied\t\t {torch.cuda.memory_allocated() / 1e6}")
                    
                    notify(f"B: {i} / {len(dataloaders[phase])} \t Loss: \t {loss:.4e}", level=4)
                    
            # METRICS - NORMALIZE EPOCH LOSS AND IOU BY HOW MANY IMAGES
            if phase is 'train':
                epoch_loss[phase] = epoch_loss[phase].cpu().detach().numpy() / (len(repeats) + n_images)
            iou_list[phase]   = np.mean(np.vstack(iou_list[phase]), axis=0)
        
        # UPDATE LEARNING RATE SCHEDULER
        lr_list.append(learning_rate_scheduler.get_lr()[0])
        learning_rate_scheduler.step()
        
        # PLOT, FIRST BY TRANSFERING TO LISTS
        for i in phases:
            epoch_losses[i].append(epoch_loss[i])
            iou_lists[i].append(iou_list[i])
            notify(f"Epoch {epoch}\t {i} Loss: \t {epoch_loss[i]:.4e} IOUs: \t {iou_list[i][0]:.4e} \t {iou_list[i][1]:.4e}\t {iou_list[i][2]:.4e}", level=0)
        
        """
        update_plots([epoch_losses[i] for i in phases], [iou_lists[i] for i in phases], fig, axs, 
                     colors=["r", "g", "b", "k", "orange"]*2,
                     annotations=f"Learning Rate: {lr_list[-1]:.2e}, Epochs: {len(lr_list)}"
                    )
        """
        
        # SAVE MODEL EVERY TEN EPOCHS
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'iou_lists' : iou_lists,
                'epoch_losses' : epoch_losses,
                'lr_list' : lr_list
            }, r'/project/glennie/fhacesga/RECTDNN/FLNN/checkpoint_111724_tiled_3_lessinputaugmentation.pth')
            
            torch.save(model, r"/project/glennie/fhacesga/RECTDNN/FLNN/111724_tiled_3_lessinputaugmentation.pth")
        
    return model


class WrapperTPNN(nn.Module):
    def __init__(self, tpnn_model, input_channels=1, target_channels=2):
        """
        A wrapper around the TPNN model that adapts single-channel input for the existing model.

        Args:
            tpnn_model (nn.Module): The pretrained TPNN model.
            input_channels (int): Number of input channels (e.g., 1 for grayscale).
            target_channels (int): Target number of channels for the adapted model.
        """
        super(WrapperTPNN, self).__init__()
        self.tpnn = tpnn_model
        
        # Upchanneling layer to convert single-channel input to multi-channel
        self.upchannel = nn.Conv2d(input_channels, target_channels, kernel_size=1)

    def forward(self, x, resize=False):
        # Upchannel the input
        x = self.upchannel(x)
        
        # Pass through the original TPNN model
        return self.tpnn(x, resize=resize)

or_model = TPNN(num_classes=3, finalpadding=1, inputsize=2)
checkpoint = torch.load(r"/project/glennie/fhacesga/RECTDNN/FLNN/checkpoint_021524.pth")
or_model.load_state_dict(checkpoint['model_state_dict'])

model = WrapperTPNN(or_model, input_channels=1, target_channels=2)

model = train(model, loaders, num_epochs=5000, learning_rate=1e-5,
             continue_from=r"/project/glennie/fhacesga/RECTDNN/FLNN/checkpoint_111724_tiled_2.pth")

