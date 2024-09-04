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