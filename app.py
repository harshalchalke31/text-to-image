# importing dependancies
import os
import customtkinter as ctk
# this is going to help to render our image from stable diffusion back to our application
from PIL import ImageTk
from authtoken import auth_token
from datetime import datetime
import torch
from torch import autocast
from diffusers import DiffusionPipeline

# Step 1: Initializing the application
app = ctk.CTk()
app.geometry('600x600')
app.title("Text-to-Image 1")
ctk.set_appearance_mode('dark')

# Application title
title_label = ctk.CTkLabel(app, text="Text-to-Image Generation with Stable Diffusion", font=('Helvetica', 20, 'bold'), text_color='cyan')
title_label.place(x=10, y=10)

# Application description
description_text = (
    "Welcome to the Text-to-Image Generation application using the Stable Diffusion model. "
    "Experience the power of AI by transforming your prompts into stunning visuals. "
    "Simply enter a description, and let the model craft high-quality images for you. "
    "Dive into the world of creative possibilities and explore the boundaries of imagination!"
)
description_label = ctk.CTkLabel(app, text=description_text, font=('Helvetica', 12), wraplength=780, justify='left', text_color='white')
description_label.place(x=10, y=50)

# create a text box on the application to input the prompt to the model
textbox = ctk.CTkEntry(app, width=580,corner_radius=10,font=('Helvetica',20),placeholder_text='Enter prompt here',fg_color='grey10',text_color='white')
textbox.place(x=10,y=80)

# create a place holder for image
lmain = ctk.CTkLabel(app,width=512 ,height=512)  # height and width are corresponding to what our model is going to return back to us
lmain.place(x=44,y=170)

# initialize model and pipeline
# modelid = 'CompVis/Stable-diffusion-v1-4'
modelid= 'stabilityai/stable-diffusion-xl-base-1.0'
device='cuda'

# if we use the revison we can load the model into a GPU with 4GB of VRAM
pipeline = DiffusionPipeline.from_pretrained(modelid,variant='fp16', torch_dtype=torch.float16,use_safetensors=True, use_auth_token=auth_token)
pipeline.to(device)

def generate():
    '''get the prompt from the textbox, set the guidance scale according to 
        how close you want your model to follow the promt, then extract the image'''
    with autocast(device): 
        image = pipeline(textbox.get(), guidance_scale=8.5).images[0]
        
    now = datetime.now()
    formatted_now = now.strftime('%m-%d-%H-%M')

    # directory for saving image
    directory= 'generated_images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory,f'gen_img({formatted_now}).png')

    # save the generated image
    image.save(filepath)
    final_image = ImageTk.PhotoImage(image)
    lmain.configure(image=final_image)

# create a button to trigger a response
button = ctk.CTkButton(app,text='Generate Image',font=('Helvetica',16),text_color='white',fg_color='blue',command=generate)
button.place(x=220,y=130)
# to run the application
app.mainloop()
