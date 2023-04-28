import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[start_row:start_row + num_rows, start_col:start_col + num_cols]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    row_ratio = input_rows / output_rows
    col_ratio = input_cols / output_cols
    for i in range(output_rows):
        for j in range(output_cols):
            # calculate the corresponding pixel in the input image
            input_i = int(i * row_ratio)
            input_j = int(j * col_ratio)

            # copy the pixel value from the input image to the output image
            output_image[i, j, :] = input_image[input_i, input_j, :]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2, )
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    return rotation_matrix @ point
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    theta=-theta
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3
    # Define the rotation matrix
    pivot_point = np.array([input_cols / 2, input_rows / 2])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Define a function for rotating 2D coordinates around a pivot point
    def rotate2d(point, pivot_point, theta):
        offset_point = point - pivot_point
        rotated_point = np.dot(rotation_matrix, offset_point)
        final_point = rotated_point + pivot_point
        return final_point.astype(int)

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    for row_idx in range(output_image.shape[0]):
        for col_idx in range(output_image.shape[1]):
            # Rotate the pixel's 2D coordinates
            output_coords = np.array([col_idx, row_idx])
            input_coords = rotate2d(output_coords, pivot_point, theta)

            # Check if the input coordinates are within the bounds of the input image
            if input_coords[0] >= 0 and input_coords[
                    0] < input_cols and input_coords[1] >= 0 and input_coords[
                        1] < input_rows:
                # Interpolate the pixel's value using the four surrounding pixels
                top_left = np.floor(input_coords).astype(int)
                bottom_right = np.ceil(input_coords).astype(int)
                top_right = np.array([bottom_right[0], top_left[1]])
                bottom_left = np.array([top_left[0], bottom_right[1]])

                # Compute the interpolation weights
                alpha = input_coords[0] - top_left[0]
                beta = input_coords[1] - top_left[1]

                # Interpolate the pixel's value
                top_left_val = input_image[top_left[1], top_left[0], :]
                top_right_val = input_image[top_right[1], top_right[0], :]
                bottom_left_val = input_image[bottom_left[1],
                                              bottom_left[0], :]
                bottom_right_val = input_image[bottom_right[1],
                                               bottom_right[0], :]
                output_image[row_idx, col_idx, :] = (
                    (1 - alpha) * (1 - beta) * top_left_val + alpha *
                    (1 - beta) * top_right_val +
                    (1 - alpha) * beta * bottom_left_val +
                    alpha * beta * bottom_right_val)
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
