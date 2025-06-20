import os
import re
import time
import io
import numpy as np
from PIL import Image
import torch
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 🔐 Authenticate once
gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")
if gauth.credentials is None:
    # First time authentication
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh expired token
    gauth.Refresh()
else:
    # Use existing valid credentials
    gauth.Authorize()

# Save credentials for next time
gauth.SaveCredentialsFile("credentials.json")
drive = GoogleDrive(gauth)

# 🧠 Load Real-ESRGAN model once
# 'RealESRGAN x4plus.pth'
model_path = 'RealESRGAN x4plus.pth'

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# Load model
state_dict = torch.load(model_path, map_location=device)['params_ema']
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)
model.eval()
model = model.to(device)

# Create upsampler WITHOUT model_path parameter
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=512 if device == 'cuda' else 256,
    pre_pad=0,
    half=True if device == 'cuda' else False
)

# Clean up
del state_dict

# Track already-processed file IDs
processed_files = set()

def get_or_create_folder(folder_name, parent_id='root'):
    query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folders = drive.ListFile({'q': query}).GetList()
    if folders:
        return folders[0]['id']
    folder = drive.CreateFile({
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': parent_id}]
    })
    folder.Upload()
    return folder['id']

def find_non_watermarked_folders():
    print("🔎 Scanning for non_watermarked_* folders...")
    all_folders = drive.ListFile({
        'q': "mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    return [f for f in all_folders if f['title'].startswith('non_watermarked_')]

def upscale_with_realesrgan(input_path, output_path):
    img = Image.open(input_path).convert('RGB')
    img = np.array(img)

    output, _ = upsampler.enhance(img, outscale=4)
    Image.fromarray(output).save(output_path)

def process_folder(folder):
    folder_name = folder['title']
    folder_id = folder['id']
    match = re.search(r'non_watermarked_(\w+)', folder_name)
    supply_id = match.group(1) if match else "unknown"
    upscaled_folder_name = f"upscaled_{supply_id}"
    
    # 🔧 Get parent folder info directly
    folder_metadata = drive.CreateFile({'id': folder_id})
    folder_metadata.FetchMetadata()
    parent_folder_id = folder_metadata['parents'][0]['id']
    
    # Create upscaled folder in the same parent as non_watermarked folder
    upscaled_folder_id = get_or_create_folder(upscaled_folder_name, parent_id=parent_folder_id)

    files = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for file in files:
        file_id = file['id']
        title = file['title']
        if file_id in processed_files:
            continue

        print(f"⬇️ Downloading {title}...")
        file.GetContentFile(title)

        # 🧠 RealESRGAN upscaling
        try:
            upscaled_title = f"upscaled_{title}"
            upscale_with_realesrgan(title, upscaled_title)
            print(f"✅ Upscaled: {upscaled_title}")
        except Exception as e:
            print(f"❌ Error while upscaling {title}: {e}")
            os.remove(title)
            continue

        # Upload result
        output_file = drive.CreateFile({
            'title': upscaled_title,
            'parents': [{'id': upscaled_folder_id}]
        })
        output_file.SetContentFile(upscaled_title)
        output_file.Upload()
        print(f"⬆️ Uploaded: {upscaled_title}")

        # Clean up
        os.remove(title)
        os.remove(upscaled_title)
        processed_files.add(file_id)

# 🔁 Loop forever
while True:
    try:
        folders = find_non_watermarked_folders()
        for folder in folders:
            process_folder(folder)
    except Exception as e:
        print(f"❌ Global error: {e}")
    print("⏳ Sleeping 60 seconds...")
    time.sleep(60)
