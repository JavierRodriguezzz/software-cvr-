import torch
import torch.nn as nn
import numpy as np
import cv2

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, base_channels=32):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels*2)
        self.enc3 = DoubleConv(base_channels*2, base_channels*4)
        self.enc4 = DoubleConv(base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels*8, base_channels*16)
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels*16, base_channels*8)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels*2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

class UNetSegmenter:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_channels=1, n_classes=3).to(self.device)
        state = torch.load(str(model_path), map_location=self.device)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, img_bgr):
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (544, 336), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy((img_resized/255.0).astype(np.float32)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img_t).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if (h, w) != (336, 544):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        return pred
