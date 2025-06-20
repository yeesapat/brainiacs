import discord
from discord import ui, ButtonStyle, File
from discord.ext import commands
import asyncio
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload
import io
import os
import tempfile
from dotenv import load_dotenv
import re
import sys

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE") 
UPLOAD_FOLDER_ID = os.getenv("GOOGLE_FOLDER_ID")
WATERMARK_PATH = os.getenv("WATERMARK_PATH", "Water_Mark.png")

if CREDENTIALS_FILE is None:
    print("WARNING: GOOGLE_CREDENTIALS_FILE environment variable is not set.")
    print("Google Drive upload functionality will be disabled.")
elif not os.path.exists(CREDENTIALS_FILE):
    print(f"WARNING: Google credentials file not found at {CREDENTIALS_FILE}")
    print("Google Drive upload functionality will be disabled.")

SCOPES = ['https://www.googleapis.com/auth/drive.file']

active_sessions = {}
class ImageQCSession:
    def __init__(self, message_id, supply_id, original_images, user_id):
        self.message_id = message_id
        self.supply_id = supply_id
        self.original_images = original_images 
        self.processed_images = [] 
        self.processed_images_no_watermark = []
        self.current_index = 0
        self.user_id = user_id
        self.qc_status = [] 
        self.folder_id = None
        self.folder_link = None
        self.feedback = {} 
        self.passed_images = [] 
    
    def is_complete(self):
        return all(status is not None for status in self.qc_status)
    
    def all_passed(self):
        return all(status is True for status in self.qc_status)

# --- Image Processing Functions ---
def apply_gray_world(image):
    """Apply Gray World color correction algorithm to an image"""
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Split the image into its BGR components
    b, g, r = cv2.split(cv_image)
    
    # Calculate the average of each channel
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)
    
    # Calculate the average gray value
    avg = (r_avg + g_avg + b_avg) / 3
    
    # Calculate the scaling factors
    r_scale = avg / r_avg if r_avg > 0 else 1
    g_scale = avg / g_avg if g_avg > 0 else 1
    b_scale = avg / b_avg if b_avg > 0 else 1
    
    # Apply the scaling factors
    r = cv2.convertScaleAbs(r, alpha=r_scale)
    g = cv2.convertScaleAbs(g, alpha=g_scale)
    b = cv2.convertScaleAbs(b, alpha=b_scale)
    
    # Merge the channels back
    balanced_image = cv2.merge([b, g, r])
    
    # Convert back to PIL image
    return Image.fromarray(cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB))

def component_stretching(image):
    """Apply contrast stretching to each color channel"""
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Split the image into its BGR components
    b, g, r = cv2.split(cv_image)
    
    # Apply histogram stretching to each channel
    channels = []
    for channel in [b, g, r]:
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        # Avoid division by zero
        if max_val > min_val:
            stretched = np.uint8(255 * ((channel - min_val) / (max_val - min_val)))
        else:
            stretched = channel
            
        channels.append(stretched)
        
    # Merge the channels back
    stretched_image = cv2.merge([channels[0], channels[1], channels[2]])
    
    # Convert back to PIL image
    return Image.fromarray(cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB))

def retouch_image(pil_image):
    # Apply Gray World assumption for color balance
    balanced_image = apply_gray_world(pil_image)

    # Convert to OpenCV format
    cv_image = cv2.cvtColor(np.array(balanced_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # Dynamically adjust contrast/brightness based on brightness score
    # Histogram Equalization -> for improving contrast and brightness
    if brightness < 60:
        alpha = 1.25  # higher contrast
        beta = 25    # brighten
    elif brightness > 180:
        alpha = 0.95  # reduce contrast a bit
        beta = -25   # darken
    else:
        alpha = 1.2
        beta = 10

    cv_image = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)

    # Back to PIL
    processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # # Sharpen
    # enhancer = ImageEnhance.Sharpness(processed_image)
    # processed_image = enhancer.enhance(1.3)

    # Component stretch
    processed_image = component_stretching(processed_image)

    return processed_image

def add_watermark(image, watermark_path=WATERMARK_PATH, position="top-right", margin=5, opacity=0.8):
    try:
        # Check if watermark file exists
        if not os.path.exists(watermark_path):
            print(f"Warning: Watermark file not found at {watermark_path}")
            return image.convert("RGB") if image.mode != "RGB" else image
        
        # Open watermark image with alpha
        watermark = Image.open(watermark_path).convert("RGBA")
        image = image.convert("RGBA")
        img_w, img_h = image.size
        wm_w, wm_h = watermark.size
        fixed_wm_width = 400
        aspect_ratio = watermark.height / watermark.width
        wm_w = fixed_wm_width
        wm_h = int(fixed_wm_width * aspect_ratio)
        watermark = watermark.resize((wm_w, wm_h))

        wm_w, wm_h = watermark.size
        alpha = watermark.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        watermark.putalpha(alpha)
        
        # Position watermark
        if position == "top-right":
            x = img_w - wm_w - margin
            y = margin
        elif position == "bottom-right":
            x = img_w - wm_w - margin
            y = img_h - wm_h - margin
        elif position == "bottom-left":
            x = margin
            y = img_h - wm_h - margin
        elif position == "top-left":
            x = margin
            y = margin
        else:  # center
            x = (img_w - wm_w) // 2
            y = (img_h - wm_h) // 2

        # Paste with transparency
        image.paste(watermark, (x, y), watermark)
        return image.convert("RGB")
    except Exception as e:
        print(f"Error applying watermark: {e}")
        return image.convert("RGB") if image.mode != "RGB" else image

# --- Google Drive Functions ---
def is_gdrive_enabled():
    """Check if Google Drive functionality is properly configured"""
    return CREDENTIALS_FILE is not None and os.path.exists(CREDENTIALS_FILE)

def create_drive_folder(folder_name, parent_id=None):
    if not is_gdrive_enabled():
        print("Cannot create Google Drive folder: credentials not configured")
        return None, None
    
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            folder_metadata['parents'] = [parent_id]

        folder = service.files().create(body=folder_metadata, fields='id, webViewLink').execute()

        service.permissions().create(
            fileId=folder['id'],
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()

        return folder['id'], folder['webViewLink']
    except Exception as error:
        print(f'Error creating folder: {error}')
        return None, None

def upload_to_google_drive(image_data, filename='processed_image.png', folder_id=None):
    if not is_gdrive_enabled():
        print("Cannot upload to Google Drive: credentials not configured")
        return None, None
    
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': filename}
        if folder_id:
            file_metadata['parents'] = [folder_id]

        media = MediaIoBaseUpload(io.BytesIO(image_data), mimetype='image/png')

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()

        # Make file public
        service.permissions().create(
            fileId=file['id'],
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        
        return file['id'], file['webViewLink']
    
    except Exception as error:
        print(f'An error occurred during upload: {error}')
        return None, None

# --- UI Components ---
class QCButtons(ui.View):
    def __init__(self, session):
        super().__init__(timeout=None)
        self.session = session
    
    # @ui.button(label="◀ Previous", style=ButtonStyle.secondary)
    # async def previous_button(self, interaction: discord.Interaction, button: ui.Button):
    #     if self.session.current_index > 0:
    #         self.session.current_index -= 1
    #         await update_qc_message(interaction, self.session)
    #     else:
    #         await interaction.response.send_message("Already at the first image.", ephemeral=True)
    
    # @ui.button(label="Next ▶", style=ButtonStyle.secondary)
    # async def next_button(self, interaction: discord.Interaction, button: ui.Button):
    #     if self.session.current_index < len(self.session.processed_images) - 1:
    #         self.session.current_index += 1
    #         await update_qc_message(interaction, self.session)
    #     else:
    #         await interaction.response.send_message("Already at the last image.", ephemeral=True)
    
    @ui.button(label="🟥", style=ButtonStyle.danger)
    async def cancel_button(self, interaction: discord.Interaction, button: ui.Button):
        # Cancel the entire process
        await interaction.response.send_message("QC process cancelled.", ephemeral=False)
        
        # Delete session
        if self.session.message_id in active_sessions:
            del active_sessions[self.session.message_id]
            
        # Clean up the message
        await interaction.message.delete()
    
    @ui.button(label="❌ failed", style=ButtonStyle.danger)
    async def not_pass_button(self, interaction: discord.Interaction, button: ui.Button):
        # Mark current image as not passed
        self.session.qc_status[self.session.current_index] = False
        
        await interaction.response.send_message(
        "Choose a retouch option for this image:",
        ephemeral=False,
        view=RetouchReasonButtons(self.session, self.session.current_index)
        )
    
    @ui.button(label="✅ Pass", style=ButtonStyle.success)
    async def pass_button(self, interaction: discord.Interaction, button: ui.Button):
        # Mark current image as passed
        self.session.qc_status[self.session.current_index] = True
        
        # Add to passed images list
        if self.session.current_index not in self.session.passed_images:
            self.session.passed_images.append(self.session.current_index)
        
        # Format the passed images message
        passed_nums = [str(i+1) for i in self.session.passed_images]
        if len(passed_nums) <= 2:
            passed_str = " and ".join(passed_nums)
        else:
            passed_str = ", ".join(passed_nums[:-1]) + ", and " + passed_nums[-1]
        
        # await interaction.response.send_message(
        #     f"Image{' ' if len(passed_nums) == 1 else 's '}{passed_str} marked as PASSED.", 
        #     ephemeral=False
        # )
        
        # Move to next image if available
        if self.session.current_index < len(self.session.processed_images) - 1:
            self.session.current_index += 1
            await update_qc_message(interaction, self.session)
        else:
            # Check if all images have been reviewed
            if self.session.is_complete():
                await finalize_qc_process(interaction, self.session)


class RetouchReasonButtons(ui.View):
    def __init__(self, session, image_index):
        super().__init__(timeout=None)
        self.session = session
        self.image_index = image_index

    @ui.button(label="🔍 Too Blurry", style=ButtonStyle.primary)
    async def too_blurry(self, interaction, button):
        await self.retouch(interaction, "Sharpen")

    @ui.button(label="🌡 Too Yellow", style=ButtonStyle.primary)
    async def too_yellow(self, interaction, button):
        await self.retouch(interaction, "Reduce Warm Tones")

    @ui.button(label="❄ Too Blue", style=ButtonStyle.primary)
    async def too_blue(self, interaction, button):
        await self.retouch(interaction, "Reduce Cool Tones")

    @ui.button(label="💡 Too Bright", style=ButtonStyle.primary)
    async def too_bright(self, interaction, button):
        await self.retouch(interaction, "⛔️ Adjust Brightness")

    @ui.button(label="🌑 Too Dark", style=ButtonStyle.primary)
    async def too_dark(self, interaction, button):
        await self.retouch(interaction, "💡 Adjust Brightness")

    async def retouch(self, interaction, reason):
        try:
            original = self.session.original_images[self.image_index]
            cv_image = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

            # Apply targeted retouch
            if reason == "Sharpen":
                # Sharpen
                kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                cv_image = cv2.filter2D(cv_image, -1, kernel)
            elif reason == "Reduce Warm Tones":
                # Reduce warm tones (shift towards blue)
                cv_image[:, :, 2] = cv2.subtract(cv_image[:, :, 2], 10)  # Red down
                cv_image[:, :, 1] = cv2.subtract(cv_image[:, :, 1], 5)   # Green down
            elif reason == "Reduce Cool Tones":
                # Reduce cool tones (shift toward warmth)
                cv_image[:, :, 0] = cv2.subtract(cv_image[:, :, 0], 10)  # Blue down
                cv_image[:, :, 1] = cv2.add(cv_image[:, :, 1], 5)        # Green up
            elif reason == "⛔️ Adjust Brightness":
                cv_image = cv2.convertScaleAbs(cv_image, alpha=0.95, beta=-20)
            elif reason == "💡 Adjust Brightness":
                cv_image = cv2.convertScaleAbs(cv_image, alpha=1.05, beta=20)

            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Save retouched image without watermark
            self.session.processed_images_no_watermark[self.image_index] = pil_image

            # Add watermark
            watermarked = add_watermark(pil_image)
            self.session.processed_images[self.image_index] = watermarked

            # Reset QC status
            self.session.qc_status[self.image_index] = None
            if self.image_index in self.session.passed_images:
                self.session.passed_images.remove(self.image_index)

            self.session.current_index = self.image_index

            await interaction.response.send_message(
                f" **{reason}**  to image {self.image_index + 1}.", 
                ephemeral=False
            )

            await update_qc_message(interaction, self.session)

        except Exception as e:
            await interaction.channel.send(f"❌ Error applying retouch: {str(e)}")

# --- Helper Functions ---
async def update_qc_message(interaction, session):
    # Create a temporary file to send the current image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        current_image = session.processed_images[session.current_index]
        current_image.save(temp_file, format="PNG")
    
    # Build status message
    status_markers = []
    for i, status in enumerate(session.qc_status):
        if i == session.current_index:
            marker = "🔍"  # Current image
        elif status is True:
            marker = "✅"  # Passed
        elif status is False:
            marker = "❌"  # Not passed
        else:
            marker = "⬜"  # Pending
        status_markers.append(marker)
    
    status_line = " ".join(status_markers)
    file = File(temp_file.name, filename="preview.png")
    
    embed = discord.Embed(
        title=f"Review of Supply ID: {session.supply_id}",
        description=f"Image {session.current_index + 1} of {len(session.processed_images)}\n{status_line}",
        color=0x3498db
    )
    
    embed.set_image(url="attachment://preview.png")
    
    try:
        await interaction.response.edit_message(embed=embed, attachments=[file], view=QCButtons(session))
    except discord.errors.InteractionResponded:
        await interaction.message.edit(embed=embed, attachments=[file], view=QCButtons(session))
    
    # Delete the temporary file after sending
    os.unlink(temp_file.name)

async def finalize_qc_process(interaction, session):
    approved_images = []
    passed_count = sum(1 for status in session.qc_status if status is True)
    failed_count = sum(1 for status in session.qc_status if status is False)
    
    # Upload passed images to Google Drive
    if passed_count > 0:
        # Upload all images that passed QC
        for i, (img, img_no_watermark, status) in enumerate(zip(session.processed_images, session.processed_images_no_watermark, session.qc_status)):
            if status:  # True means passed
                # Add to approved images list
                approved_images.append((img, img_no_watermark, f"processed_image_{i+1}.png"))
        
        # Check if Google Drive functionality is available
        if is_gdrive_enabled():
            # Create main folder
            main_folder_name = f"Approved_{session.supply_id}"
            main_folder_id, main_folder_link = create_drive_folder(main_folder_name, parent_id=UPLOAD_FOLDER_ID)
            
            if main_folder_id:
                # Create two subfolders
                watermark_folder_id, _ = create_drive_folder("Watermarked", parent_id=main_folder_id)
                no_watermark_folder_id, _ = create_drive_folder("No_Watermark", parent_id=main_folder_id)
                
                # Upload all approved images to both folders
                upload_results = []
                
                for img, img_no_watermark, filename in approved_images:
                    # Upload watermarked version
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    file_id, file_link = upload_to_google_drive(
                        img_bytes, 
                        filename=filename, 
                        folder_id=watermark_folder_id
                    )
                    
                    # Upload non-watermarked version
                    img_byte_arr_no_wm = io.BytesIO()
                    img_no_watermark.save(img_byte_arr_no_wm, format='PNG')
                    img_bytes_no_wm = img_byte_arr_no_wm.getvalue()
                    
                    file_id_no_wm, _ = upload_to_google_drive(
                        img_bytes_no_wm, 
                        filename=filename, 
                        folder_id=no_watermark_folder_id
                    )
                    
                    if file_id:
                        upload_results.append((filename, file_link))
                
                # Final success message
                if upload_results:
                    await interaction.channel.send(
                        f"✅ Retouched images of Supply ID: {session.supply_id}\n"
                        f"📁 Folder Link: {main_folder_link}\n"
                        # f"Both watermarked and non-watermarked versions are available in separate subfolders."
                    )
                else:
                    await interaction.channel.send(
                        f"⚠️ QC Complete for Supply ID: {session.supply_id}, but no images were uploaded successfully."
                    )
            else:
                await interaction.channel.send(
                    f"❌ QC Complete for Supply ID: {session.supply_id}, but failed to create Google Drive folders."
                )
        else:
            # Google Drive functionality not available - save locally
            # Create directories to save passed images
            local_dir = f"approved_images_{session.supply_id}"
            watermarked_dir = os.path.join(local_dir, "watermarked")
            no_watermark_dir = os.path.join(local_dir, "no_watermark")
            
            os.makedirs(watermarked_dir, exist_ok=True)
            os.makedirs(no_watermark_dir, exist_ok=True)
            
            # Save all approved images locally
            saved_count = 0
            for img, img_no_watermark, filename in approved_images:
                try:
                    # Save watermarked version
                    filepath_wm = os.path.join(watermarked_dir, filename)
                    img.save(filepath_wm, format='PNG')
                    
                    # Save non-watermarked version
                    filepath_no_wm = os.path.join(no_watermark_dir, filename)
                    img_no_watermark.save(filepath_no_wm, format='PNG')
                    
                    saved_count += 1
                except Exception as e:
                    print(f"Error saving image {filename}: {e}")
            
            await interaction.channel.send(
                f"✅ QC Complete Supply ID: {session.supply_id}\n"
                f"{passed_count} images passed QC.\n"
                f"⚠️ Google Drive is not configured, so {saved_count} images were saved locally in folder: {local_dir}\n"
                f"Both watermarked and non-watermarked versions are available in separate subfolders."
            )
    
    # Handle failed images
    if failed_count > 0:
        failed_indices = [i for i, status in enumerate(session.qc_status) if status is False]
        failed_nums = [str(i+1) for i in failed_indices]
        
        if not session.all_passed():
            await interaction.channel.send(
                f"⚠️ QC Status for Supply ID: {session.supply_id}\n"
                f"Results: {passed_count} passed, {failed_count} failed.\n"
                f"Failed images: {', '.join(failed_nums)}\n"
                f"Use the 'Retouch Again' button on the failed images to retry."
            )
    
    # Clean up the session if all images passed
    if session.all_passed() and session.message_id in active_sessions:
        del active_sessions[session.message_id]
        
        # Clean up the QC message
        try:
            await interaction.message.delete()
        except Exception as e:
            print(f"Error deleting message: {e}")
            pass

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'✅ Log in as: {bot.user}')
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="for images to process"))

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Process commands first
    await bot.process_commands(message)
    
    # Check if message has attachments and is in the correct channel
    if (message.channel.id == CHANNEL_ID or CHANNEL_ID == 0) and message.attachments:
        # Extract Supply ID from message content using regex
        supply_id_match = re.search(r'(?i)supply\s*id:?\s*(\w+)', message.content)
        supply_id = supply_id_match.group(1) if supply_id_match else f"Unknown_{message.id}"
        
        # Filter only image attachments
        image_attachments = [
            attachment for attachment in message.attachments 
            if attachment.content_type and attachment.content_type.startswith('image/')
        ]
        
        if not image_attachments:
            await message.reply("❌ No valid image attachments found.")
            return
        
        # Create a processing status message
        status_message = await message.reply(f"⏳ Processing {len(image_attachments)} images for Supply ID: {supply_id}...")
        
        # Prepare the session
        session = ImageQCSession(
            message_id=status_message.id,
            supply_id=supply_id,
            original_images=[],
            user_id=message.author.id
        )
        
        try:
            # Process each image
            for attachment in image_attachments:
                try:
                    # Download the image
                    image_bytes = await attachment.read()
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # Store original image
                    session.original_images.append(image)
                    
                    # Process the image
                    retouched_image = retouch_image(image)
                    
                    # Store processed image without watermark
                    session.processed_images_no_watermark.append(retouched_image)
                    
                    # Add watermark and store
                    watermarked_image = add_watermark(retouched_image)
                    session.processed_images.append(watermarked_image)
                    
                    # Initialize QC status as None (pending)
                    session.qc_status.append(None)
                except Exception as e:
                    print(f"Error processing attachment {attachment.filename}: {e}")
            
            # If no images were processed successfully
            if not session.processed_images:
                await status_message.edit(content="❌ Failed to process any of the attached images.")
                return
                
            # Save session
            active_sessions[status_message.id] = session
            
            # Create a temporary file to send the first processed image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                session.processed_images[0].save(temp_file, format="PNG")
            
            # Create an embed for the QC interface
            file = File(temp_file.name, filename="preview.png")
            
            embed = discord.Embed(
                title=f"QC Review - Supply ID: {supply_id}",
                description=f"Image 1 of {len(session.processed_images)}\n" + 
                            "🔍 " + "⬜ " * (len(session.processed_images) - 1),
                color=0x3498db
            )
            
            embed.set_image(url="attachment://preview.png")
            
            # Replace the status message with the QC interface
            try:
                await status_message.delete()
            except Exception as e:
                print(f"Error deleting status message: {e}")
                
            try:
                qc_message = await message.reply(
                    embed=embed, 
                    file=file,
                    view=QCButtons(session)
                )
                
                # Update the message ID in the session
                session.message_id = qc_message.id
                active_sessions[qc_message.id] = session
            except Exception as e:
                print(f"Error sending QC message: {e}")
                await message.reply(f"❌ Error creating QC interface: {str(e)}")
            
            # Delete the temporary file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Error deleting temporary file: {e}")
            
        except Exception as e:
            await status_message.edit(content=f"❌ Error processing images: {str(e)}")
            print(f"Error processing images: {e}")

# --- Run the bot ---
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
    