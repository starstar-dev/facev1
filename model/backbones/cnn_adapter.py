"""
CNN backbone adapter: wraps ResNet/OSNet to produce ViT-compatible token sequences.
Output: (B, N_patches+1, 768) — CLS token + patch tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAdapter(nn.Module):
    """
    Adapts a CNN feature map to the ViT token format expected by CrossModalFusion.
    
    Args:
        cnn: CNN backbone (e.g., ResNet-50, OSNet)
        cnn_dim: output channels of the CNN
        out_dim: target token dimension (default 768 for CrossModalFusion compat)
        img_size: input image size (H, W)
        last_stride: CNN last stride (controls feature map size)
    """
    def __init__(self, cnn, cnn_dim, out_dim=768, img_size=(224, 224), last_stride=1):
        super().__init__()
        self.cnn = cnn
        self.cnn_dim = cnn_dim
        self.out_dim = out_dim
        
        # Project CNN channels → target dimension
        self.proj = nn.Conv2d(cnn_dim, out_dim, kernel_size=1)
        
        # Compute feature map size
        h = img_size[0] // (16 // last_stride)  # Approximate for ResNet with stride
        w = img_size[1] // (16 // last_stride)
        # More precisely: compute by running a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size[0], img_size[1])
            feat = self._extract_feature_map(dummy)
            self.feat_h, self.feat_w = feat.shape[2], feat.shape[3]
        self.num_patches = self.feat_h * self.feat_w
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        print(f"CNNAdapter: cnn_dim={cnn_dim}, out_dim={out_dim}, "
              f"feat_map=({self.feat_h},{self.feat_w}), num_patches={self.num_patches}")
    
    def _extract_feature_map(self, x):
        """Extract CNN feature map. Override for different CNN architectures."""
        return self.cnn(x)
    
    def forward(self, x):
        B = x.size(0)
        feat_map = self._extract_feature_map(x)          # (B, C, H, W)
        feat_map = self.proj(feat_map)                     # (B, 768, H, W)
        feat_map = feat_map.flatten(2).transpose(1, 2)    # (B, N, 768)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)     # (B, 1, 768)
        x = torch.cat([cls_tokens, feat_map], dim=1)      # (B, 1+N, 768)
        x = x + self.pos_embed
        return x


class ResNet50Adapter(CNNAdapter):
    """ResNet-50 backbone adapted for QTA-ReID."""
    def __init__(self, pretrained=True, out_dim=768, img_size=(224, 224)):
        from model.backbones.resnet import ResNet, Bottleneck
        cnn = ResNet(last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3])
        if pretrained:
            import torchvision.models as tvm
            state_dict = tvm.resnet50(weights='IMAGENET1K_V1').state_dict()
            # Remove fc layer weights
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('fc.')}
            cnn.load_state_dict(state_dict, strict=False)
            print("Loaded ImageNet pretrained ResNet-50")
        super().__init__(cnn, cnn_dim=2048, out_dim=out_dim, img_size=img_size)
        self.cnn_dim = 2048


class OSNetAdapter(CNNAdapter):
    """
    OSNet backbone adapted for QTA-ReID.
    Requires: pip install torchreid
    Or implement a simplified OSNet inline.
    """
    def __init__(self, out_dim=768, img_size=(224, 224)):
        try:
            from torchreid.models import build_model as build_osnet
            cnn = build_osnet(name='osnet_x1_0', num_classes=1000, pretrained=False)
            # Remove classifier head
            cnn.classifier = nn.Identity()
            # OSNet typically outputs (B, 512) after pooling
            # Need feature map before pooling → modify forward
            cnn_dim = 512
        except ImportError:
            # Fallback: Simplified OSNet-lite
            print("torchreid not available, using simplified OSNet-lite")
            cnn = self._build_osnet_lite()
            cnn_dim = 512

    def __init__(self, out_dim=768, img_size=(224, 224)):
        cnn, cnn_dim = self._create_cnn()
        super().__init__(cnn, cnn_dim=cnn_dim, out_dim=out_dim, img_size=img_size)
    
    def _create_cnn(self):
        """Create OSNet. Uses torchreid if available, else simplified version."""
        try:
            import torchreid
            from torchreid.models import osnet_x1_0
            model = osnet_x1_0(pretrained=False)
            # Remove classifier, keep feature extractor
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            # Hack: replace forward to return feature map before pooling
            original_forward = model.forward
            def forward_with_feat_map(x):
                # OSNet forward (simplified) — depends on exact implementation
                # For torchreid: feature map before gap is accessible
                x = model.conv1(x)
                x = model.maxpool(x)
                x = model.conv2(x)
                x = model.conv3(x)
                x = model.conv4(x)
                x = model.conv5(x)
                return x  # Feature map before global pooling
            model.forward = forward_with_feat_map
            return model, 512
        except ImportError:
            # Simplified OSNet-lite based on depthwise separable convs
            return self._build_osnet_lite(), 512
    
    def _build_osnet_lite(self):
        """
        Simplified OSNet-lite: lightweight CNN with depthwise separable blocks.
        ~2M params, output 512-dim feature maps at 16×8 (for 256×128 input).
        """
        class OSBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.conv3 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
                self.bn3 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = None
                if in_ch != out_ch or stride != 1:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )
            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                return self.relu(out + identity)
        
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            OSBlock(64, 256, 2), OSBlock(256, 256),
            OSBlock(256, 384, 2), OSBlock(384, 384),
            OSBlock(384, 512, 2), OSBlock(512, 512),
        )