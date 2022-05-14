"""
This script stitches the agent movements together to form a step sequence
"""

# Importing python libraries for required processing
from PIL import Image
import glob

# Input and Output directory for processing
input_images_directory = "/export2/scratch/cv_proj_team1/Attention_and_Move/output/agent_movements/*.png"
output_gif_directory = "/export2/scratch/cv_proj_team1/Attention_and_Move/output/agent_movements/multi_agent_movements.gif"

# Processing the images to create the sequence
imgs = (Image.open(f) for f in sorted(glob.glob(input_images_directory)))
img = next(imgs)
img.save(fp=output_gif_directory,
         format='GIF',
         append_images=imgs,
         save_all=True,
         duration=300,
         loop=0)
