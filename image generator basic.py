import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"  # You can use a different model if you prefer
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")  # Move the model to GPU if available also use cuda in cpu both are correct 

def generate_image(prompt):
    # Generate images
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Example usage
if __name__ == "__main__":
    prompt = "A beautiful sunset over a mountain range"
    image = generate_image(prompt)

    # Save and display the generated image
    image.save("generated_image.png")
    image.show()
