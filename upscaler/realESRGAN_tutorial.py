import torch 
import numpy as np 
from PIL import Image 
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ---------------------

model_path = 'RealESRGAN x4plus.pth'

state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)

upsampler = RealESRGANer(
    scale=4, 
    model_path=model_path, 
    model=model,
    tile=256, 
    pre_pad=0, 
    half=False
    )

img = Image.open('S__183361588_0.png').convert('RGB')
img = np.array(img)

output, _ = upsampler.enhance(img, outscale=4)
output_img = Image.fromarray(output)
output_img.save('S__183361588_0_upscaled.png')