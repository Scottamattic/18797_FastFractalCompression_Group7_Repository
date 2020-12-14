"""
NOTE THIS FILE HAS DEPRECATED ROUTINES. I AM KEEPING IT AROUND AS A REFERENCE.
"""


from context import fractal
from fractal.dataloader import Loader
from fractal import utils
from fractal.plotting import plot_image
from fractal.coding import encode, decode
import copy
from skimage.transform import rescale
import pandas as pd
import numpy as np


if __name__ == '__main__':
    ############################################################################
    # load the first face image
    ############################################################################
    images = Loader("")
    range_image = images.get_image(0, scale_factor=0.25)

    # plot it so we know what it looks like
    plot_image(range_image, 
        title=f"Range Image {range_image.shape[0]}x{range_image.shape[1]}",
        cmap='gray')

    ############################################################################
    # divide up the first image into chunks
    ############################################################################
    # domain image is a 50% downsampled range image
    # domain_image = images.get_image(0, scale_factor=0.125)
    domain_image = np.zeros([128,128])

    plot_image(domain_image, 
        title=f"Domain Image {domain_image.shape[0]}x{domain_image.shape[1]}",
        cmap='gray')

    # each block is 4x4
    domain_chunks = utils.image_to_chunks(domain_image, 2)
    range_chunks = utils.image_to_chunks(range_image, 2)

    ############################################################################
    # encode the range image
    ############################################################################
    # encode the range image
    # codebook = encode(domain_chunks, range_chunks, verbose=True)

    # pd.DataFrame(codebook).to_csv("codebook.csv")
    codebook = pd.read_csv("codebook.csv", index_col=0).values
    print(codebook)
    print(codebook.shape)

    # decode the encoding -- should be the same as the original range image
    reconstructed_chunks = decode(codebook, domain_chunks)
    reconstructed_chunks_1iter = copy.deepcopy(reconstructed_chunks)
    
    for i in range(9):
        rec_rim = utils.chunks_to_image(reconstructed_chunks)
        # TODO try this without anti aliasing
        rec_dim = rescale(rec_rim, 0.5)#, anti_aliasing=True)
        domain_chunks = utils.image_to_chunks(rec_dim, 2)
        reconstructed_chunks = decode(codebook, domain_chunks)
    

    reconstructed_chunks_10iter = copy.deepcopy(reconstructed_chunks)


    for i in range(90):
        rec_rim = utils.chunks_to_image(reconstructed_chunks)
        # TODO try this without anti aliasing
        rec_dim = rescale(rec_rim, 0.5)#, anti_aliasing=True)
        domain_chunks = utils.image_to_chunks(rec_dim, 2)
        reconstructed_chunks = decode(codebook, domain_chunks)

    reconstructed_chunks_100iter = copy.deepcopy(reconstructed_chunks)


    # cast the chunks back into image dimensions
    reconstructed_image_1iter = \
        utils.chunks_to_image(reconstructed_chunks_1iter)
    reconstructed_image_10iter = \
        utils.chunks_to_image(reconstructed_chunks_10iter)
    reconstructed_image_100iter = \
        utils.chunks_to_image(reconstructed_chunks_100iter)

    # plot the result
    plot_image(reconstructed_image_1iter, 
        title="Reconstructed Image 1 iteration 64x64", 
        cmap='gray')
    plot_image(reconstructed_image_10iter, 
        title="Reconstructed Image 10 iterations 64x64",
        cmap='gray')
    plot_image(reconstructed_image_100iter, 
        title="Reconstructed Image 100 iterations 64x64",
        cmap='gray')








