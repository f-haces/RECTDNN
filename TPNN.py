# PYTHON IMPORTS
from tqdm.notebook import trange, tqdm

# PLOTTING
import matplotlib.pyplot as plt

# NEURAL NETWORK
import torch
import torch.nn as nn
import torchvision.models as models

def split_and_run_cnn(image_path, model, tilesize=1024, overhang_size=2):
        
    tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    num_classes = 2
    
    # Load the image
    image = Image.open(image_path)
    
    # Calculate the number of tiles needed
    width, height = image.size
    num_tiles_x = (width + tilesize-1) // tilesize
    num_tiles_y = (height + tilesize-1) // tilesize
    
    # Create an empty list to store the output tiles
    output_tiles = []
    
    output_gen = np.zeros((width, height, num_classes))
    
    # Iterate over each tile
    for tile_x in range(num_tiles_x):
        for tile_y in range(num_tiles_y):
                        
            # Calculate the coordinates for the current tile
            x0 = tile_x * tilesize
            y0 = tile_y * tilesize
            x1 = min(x0 + tilesize, width)
            y1 = min(y0 + tilesize, height)
            
            # Crop the image to the current tile
            tile = image.crop((x0, y0, x1, y1))
            
            # Pad the tile if needed
            pad_width = tilesize - tile.width
            pad_height = tilesize - tile.height
            if pad_width > 0 or pad_height > 0:
                padding = ((0, pad_height), (0, pad_width))
                tile = np.pad(tile, padding, mode='constant')
            
            # Preprocess the tile
            tile = np.array(tile)
            
            #if np.max(tile) == 1:
            #    tile = tile * 255
            
            # tile = np.where(tile > 127, 255, 0).astype(np.uint8)
            
            tile_tensor = tensor(tile).unsqueeze(0).to("cuda")
            
            # Run the CNN on the tile
            output = model(tile_tensor)
            output = output[0, 1:, :, :].cpu().detach().numpy().T
            
            # Store the output tile
            
            x_fin = tilesize - pad_width
            y_fin = tilesize - pad_height
            
            temp = output[0:x_fin, 0:y_fin, :]
            
            temp[:, :overhang_size, :] = 0
            temp[:, overhang_size:, :] = 0
            temp[:, :, overhang_size:] = 0
            temp[:, :, :overhang_size] = 0
            
            output_gen[x0:x1, y0:y1, :] = temp
        torch.cuda.empty_cache()
    return output_gen.T

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Calculate channel-wise average pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Calculate channel-wise max pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate both pooling results along the channel dimension
        pool = torch.cat([avg_pool, max_pool], dim=1)

        # Apply convolutional layer and sigmoid activation
        attention = torch.sigmoid(self.conv(pool))

        # Multiply the input feature map by the attention map
        attended_feature_map = x * attention

        return attended_feature_map

class TPNN(nn.Module):

    def __init__(self, num_classes=2, finalpadding=0, inputsize=1, verbose_level=1, legacy=False):
        super(TPNN, self).__init__()
        
        self.num_classes = num_classes

        self.softmax = nn.Softmax(dim=1)
        
        self.attention = SpatialAttention()
        
        self.verbose_level = int(verbose_level)
        
        # ResNet backbone
        self.resnet = models.resnet34(pretrained=True)
        
        # Adjust the first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(inputsize, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Encoder
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.decoder4 = self._make_decoder_block(512, 256, 256)
        self.decoder3 = self._make_decoder_block(512, 128, 128)
        self.decoder2 = self._make_decoder_block(256, 64, 64)
        if legacy:
            self.decoder1 = self._make_decoder_block(128, 64, 64, s=4)
        else:
            self.decoder1 = nn.Sequential(self._make_decoder_block(128, 64, 64), self._make_decoder_block(64, 64, 64))
        
        # Final convolutional layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, padding=finalpadding)
      
    def notify(self, mess, level=4):
        if self.verbose_level >= level:
            print(mess)
      
    def _make_decoder_block(self, in_channels, mid_channels, out_channels, s=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=s),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, resize=True):
                
        # Spatial self-attention layer
        x    = self.attention(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        self.notify((enc1.shape))
        enc2 = self.encoder2(enc1)
        self.notify((enc2.shape))
        enc3 = self.encoder3(enc2)
        self.notify((enc3.shape))
        enc4 = self.encoder4(enc3)
        self.notify((enc4.shape))
        enc5 = self.encoder5(enc4)
        self.notify((enc5.shape))
        
        
        # Decoder with residual connections
        dec4 = self.decoder4(enc5)
        self.notify((dec4.shape, enc4.shape))
        dec3 = self.decoder3(torch.cat([dec4, enc4], dim=1))
        self.notify((dec3.shape, enc3.shape))
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        self.notify((dec2.shape, enc2.shape))
        dec1 = self.decoder1(torch.cat([dec2, enc2], dim=1))
        self.notify((dec1.shape, enc1.shape))
        # Final convolutional layer
        output = self.final_conv(dec1)
        
        # Reshape output to match input dimensions
        if resize:
            output = nn.functional.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        output = self.softmax(output)
        
        return output