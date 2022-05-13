"""
This script stitches the agent movements together to form a step sequence
"""

# Importing python libraries for required processing
from PIL import Image
import glob

# Input and Output directory for processing
input_images_directory = "*.png"
output_gif_directory = "multi_agent_movements.gif"

# Processing the images to create the sequence
step_images = (Image.open(f) for f in sorted(glob.glob(input_images_directory)))
step = next(step_images)
step.save(fp=output_gif_directory,
          format='GIF',
          append_images=step,
          save_all=True,
          duration=1000,
          loop=0)
