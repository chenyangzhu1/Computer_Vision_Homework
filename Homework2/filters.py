import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    height = int((Hk - 1) / 2)
    width = int((Wk - 1) / 2)
    for i in range(Hi):
        for j in range(Wi):
            for ki in range(Hk):
                for kj in range(Wk):
                    a = image[
                        i - height + ki, j - width +
                        kj] if i - height + ki >= 0 and j - width + kj >= 0 and i - height + ki < Hi and j - width + kj < Wi else 0
                    out[i, j] += a * kernel[Hk - ki - 1, Wk - kj - 1]
    ### END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height:H + pad_height, pad_width:W + pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    padded_image = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded_image[i:i + Hk, j:j + Wk] * kernel)
    ### END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))

    out = conv_fast(f, np.flip(np.flip(g, axis=0), axis=1))
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g -= np.mean(g)

    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    g_mean = np.mean(g)
    g_std = np.std(g)
    if Hg % 2 == 0:
        new_g = np.pad(g, ((1, 0), (0, 0)), "constant", constant_values=0)
    if Wg % 2 == 0:
        new_g = np.pad(new_g, ((0, 0), (1, 0)), "constant", constant_values=0)
    Hg, Wg = new_g.shape
    f_pad = zero_pad(f, int((Hg - 1) / 2), int((Wg - 1) / 2))
    for i in range(Hf):
        for j in range(Wf):
            f_area = f_pad[i:i + Hg, j:j + Wg]
            f_mean = np.mean(f_area)
            f_std = np.std(f_area)
            out[i, j] = np.dot(((f_area - f_mean) / f_std).flatten(),
                               ((new_g - g_mean) / g_std).flatten())

    ### END YOUR CODE

    return out
