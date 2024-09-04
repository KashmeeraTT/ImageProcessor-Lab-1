import cv2
import numpy as np


index_number = "210279A"
filename = f"road{index_number[-3:-1]}.png"

# Load the image
image = cv2.imread(filename)
if image is None:
    raise FileNotFoundError("Source image not found.")


# Convert to grayscale (8-bpp)
def get_image_size(image):
    return image.shape[:2]


def convert_to_grayscale(image):
    height, width = get_image_size(image)
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            (b, g, r) = image[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_image[i, j] = gray_value
    return gray_image


gray_image = convert_to_grayscale(image)


# Enhance contrast linearly
def enhance_contrast(image):
    min_val, max_val = np.min(image), np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)


enhanced_image = enhance_contrast(gray_image)
cv2.imwrite("original.jpeg", enhanced_image)

# Define the filters
filters = {
    "FilterA": np.array([[0, -1, -1, -1, 0],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 8, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [0, -1, -1, -1, 0]]),
    "FilterB": np.array([[1, 4, 6, 4, 1],
                         [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6],
                         [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]]),
    "FilterC": np.full((5, 5), 5),
    "FilterD": np.array([[0, -1, -1, -1, 0],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 16, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [0, -1, -1, -1, 0]])
}

# Normalize filters
filters = {name: f / np.sum(f) if np.sum(f) !=
           0 else f for name, f in filters.items()}


# Apply filters and save images
def apply_filter(image, filter_matrix):
    image_height, image_width = image.shape
    filter_size = filter_matrix.shape[0]
    pad_size = filter_size // 2

    # Padding the image with zeros
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

    # Create an output image
    output_image = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+filter_size, j:j+filter_size]
            output_image[i, j] = np.sum(region * filter_matrix)

    # Clip values to stay within valid range
    output_image = np.clip(output_image, 0, 255)

    return output_image.astype(np.uint8)


rms_values = {}
for name, filter_matrix in filters.items():
    filtered_image = apply_filter(enhanced_image, filter_matrix)
    cv2.imwrite(f"{name}.jpeg", filtered_image)

    # Compute RMS difference
    rms_diff = np.sqrt(np.mean((enhanced_image - filtered_image) ** 2))
    rms_values[name] = rms_diff


# Print RMS values
for name, rms in rms_values.items():
    print(f"RMS difference for {name}: {rms}")
