import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def message_to_binary(message):
    return ''.join(format(ord(c), '08b') for c in message)

def embed_message_in_image(image, message):
    binary_message = message_to_binary(message) + '10111110'  
    pixels = np.array(image)  
    height, width = image.size
    
    message_index = 0
    for i in range(height):
        for j in range(width):
            if message_index < len(binary_message):
                pixel_val = pixels[i, j]
                pixels[i, j] = (pixel_val & ~1) | int(binary_message[message_index])
                message_index += 1
    
    return Image.fromarray(pixels, mode='L')

def calculate_mse(img1, img2):
    pixels1, pixels2 = np.array(img1), np.array(img2)
    return np.mean((pixels1 - pixels2) ** 2)

def calculate_psnr(mse, max_pixel_value=255):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(np.array(img1), np.array(img2))

def process_image(image_path, secret_message):
    original_image = Image.open(image_path).convert('L')
    encoded_image = embed_message_in_image(original_image, secret_message)
    mse = calculate_mse(original_image, encoded_image)
    psnr = calculate_psnr(mse)
    similarity_index = calculate_ssim(original_image, encoded_image)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f"Original Image\nSecret message: {secret_message}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(encoded_image, cmap='gray')
    plt.title(f"Encoded Image\nMSE: {mse:.4f}, PSNR: {psnr:.2f} dB, SSIM: {similarity_index:.4f} \n {secret_message}")
    plt.axis('off')
    
    plt.show()
    
    # Save encoded image
    encoded_image.save("encoded_lena_grayscale.png")

process_image("lena.png", "Hi I am Zaid 'Keep it secret'")
