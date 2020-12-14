from context import fractal
from fractal.dataloader import Loader
from fractal import utils
from fractal.plotting import plot_image
from fractal.coding import encode, decode, encode_svd, decode_svd
import copy
from skimage.transform import rescale
import pandas as pd
import numpy as np
import skimage.metrics as metrics
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

MODE = 'equipartition4'
SCALE = 1/4

def normalize_img(im):
    max_val = np.amax(np.abs(im))
    im = im / max_val
    im = np.clip(im,0,1) * 255
    return im


def load_aaron_face():
    facedir = Path(__file__).resolve().parent.parent / "data" / "faces"
    aaronface = Image.open(list(facedir.glob("*.pgm"))[0])
    aaronface = np.asarray(aaronface.getdata()).reshape(64,64)
    return aaronface

if __name__ == '__main__':
    print(MODE)
    print("Standard")
    ############################################################################
    # load the first face image
    ############################################################################
    images = Loader("")
    range_image = images.get_image(0, scale_factor=SCALE)
    orig = range_image

    # plot it so we know what it looks like
    plot_image(range_image, 
        title=f"Range Image {range_image.shape[0]}x{range_image.shape[1]}",
        cmap='gray')

    ############################################################################
    # divide up the first image into chunks
    ############################################################################
    # domain image is a 50% downsampled range image
    domain_image = images.get_image(0, scale_factor=SCALE/2)

    plot_image(domain_image, 
        title=f"Domain Image {domain_image.shape[0]}x{domain_image.shape[1]}",
        cmap='gray')

    # each block is 4x4
    domain_chunks = utils.Partition(domain_image, mode=MODE)
    range_chunks = utils.Partition(range_image, mode=MODE)

    ############################################################################
    # Standard Encoding
    ############################################################################
    
    ############################################################################
    # encode the range image
    ############################################################################
    # encode the range image
    start = timer()
    codebook = encode(domain_chunks, range_chunks, verbose=False)
    pd.DataFrame(codebook).to_csv("codebook.csv")
    end = timer()
    print("Time to encode is " + str(end - start))

    # load aaron's face to use as input for the reconstruction
    aaronface = load_aaron_face()
    plot_image(aaronface, 
        title=f"Domain Image {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray')
    # domain_chunks = utils.Partition(aaronface, mode=MODE)

    domain_chunks = utils.Partition(aaronface, mode=MODE)

    # save the psnr for plotting later
    psnr = np.zeros(10)
    err = np.zeros(10)
    ssi = np.zeros(10)

    # decode the encoding -- should be the same as the original range image
    # domain_chunks = utils.Partition(np.zeros([64, 64]), mode=MODE)
    reconstructed_chunks = decode(codebook, domain_chunks)
    reconstructed_chunks_1iter = copy.deepcopy(reconstructed_chunks.image)
    domain_chunks_original = copy.deepcopy(domain_chunks)
    psnr[0] = metrics.peak_signal_noise_ratio(orig,normalize_img(reconstructed_chunks.image), data_range=255)
    ssi[0] = metrics.structural_similarity(orig,reconstructed_chunks.image, data_range=255)
    err[0] = metrics.mean_squared_error(orig,reconstructed_chunks.image)
    
    for i in range(4):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode(codebook, domain_chunks)
        psnr[i+1] = metrics.peak_signal_noise_ratio(orig,normalize_img(reconstructed_chunks.image), data_range=255)
        ssi[i+1] = metrics.structural_similarity(orig,reconstructed_chunks.image, data_range=255)
        err[i+1] = metrics.mean_squared_error(orig,reconstructed_chunks.image)
    

    reconstructed_chunks_10iter = copy.deepcopy(reconstructed_chunks.image)


    for i in range(5):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode(codebook, domain_chunks)
        psnr[i+5] = metrics.peak_signal_noise_ratio(orig,normalize_img(reconstructed_chunks.image), data_range=255)
        ssi[i+5] = metrics.structural_similarity(orig,reconstructed_chunks.image, data_range=255)
        err[i+5] = metrics.mean_squared_error(orig,reconstructed_chunks.image)

    reconstructed_chunks_100iter = copy.deepcopy(reconstructed_chunks.image)


    # Get image and metrics after 10 iterations
    domain_chunks = copy.deepcopy(domain_chunks_original)
    start = timer()
    reconstructed_chunks = decode(codebook, domain_chunks)
    for i in range(9):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode(codebook, domain_chunks)
    end = timer()
    print("Time to decode 10 iterations is " + str(end - start))



    # plot the result
    plot_image(reconstructed_chunks_1iter, 
        title="Reconstructed Image 1 iteration 64x64", 
        cmap='gray')
    plot_image(reconstructed_chunks_10iter, 
        title="Reconstructed Image 5 iterations 64x64",
        cmap='gray')
    plot_image(reconstructed_chunks_100iter, 
        title="Reconstructed Image 10 iterations 64x64",
        cmap='gray')


    print("PSNR: ",psnr[0],psnr[5],psnr[9])
    print("MSE: ",err[0],err[5],err[9])
    print("SSI: ",ssi[0],ssi[5],ssi[9])
    print(reconstructed_chunks_100iter.shape)

    # Plot PSNR and MSE
    x = np.arange(1,11)
    plt.figure(figsize=(10,7))
    plt.plot(x,psnr)
    plt.suptitle("PSNR", y=0.93, fontsize=23)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("dB", fontsize=16)
    plt.show()

    plt.figure(figsize=(10,7))
    plt.suptitle("Mean Squared Error", y=0.93, fontsize=23)
    plt.plot(x,err)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("Mean Squared Error", fontsize=16)
    plt.show()

    plt.figure(figsize=(10,7))
    plt.suptitle("Structural Similarity", y=0.93, fontsize=23)
    plt.plot(x,ssi)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("Structural Similarity Index", fontsize=16)
    plt.show()


    ############################################################################

    print("SVD")
    ############################################################################
    # load the first face image
    ############################################################################
    images = Loader("")
    range_image = images.get_image(0, scale_factor=SCALE)
    orig = range_image

    # plot it so we know what it looks like
    plot_image(range_image, 
        title=f"Range Image {range_image.shape[0]}x{range_image.shape[1]}",
        cmap='gray')

    ############################################################################
    # divide up the first image into chunks
    ############################################################################
    # domain image is a 50% downsampled range image
    domain_image = images.get_image(0, scale_factor=SCALE/2)

    plot_image(domain_image, 
        title=f"Domain Image {domain_image.shape[0]}x{domain_image.shape[1]}",
        cmap='gray')

    # each block is 4x4
    domain_chunks = utils.Partition(domain_image, mode=MODE)
    range_chunks = utils.Partition(range_image, mode=MODE)

    ############################################################################
    # SVD Encoding
    ############################################################################

    ############################################################################
    # encode the range image
    ############################################################################
    # encode the range image
    start = timer()
    codebook = encode_svd(domain_chunks, range_chunks, verbose=False)
    end = timer()
    print("Time to encode is " + str(end - start))

    # save the psnr for plotting later
    psnr = np.zeros(10)
    err = np.zeros(10)
    ssi = np.zeros(10)

    num_weights = len(domain_chunks[0].ravel())
    X = np.zeros([num_weights, len(domain_chunks)])

    for i in range(len(domain_chunks)):
        X[:, i] = np.ravel(domain_chunks[i])

    # X = cp.array(X)
    # u, s, vh = cp.linalg.svd(X)
    # u = u.get()
    # s = s.get()
    # vh = vh.get()
    u, s, vh = np.linalg.svd(X)

    # print(u.shape, np.diag(s).shape, vh.shape)
    R = np.dot(np.diag(s), vh[:len(s), :len(s)])

    # print(R.shape)
    for idx, weights in enumerate(codebook):
        contractiveness = np.dot(np.linalg.inv(R), weights.reshape(num_weights,1))
        # print(f"contractiveness {np.sum(contractiveness)} for block {idx}")

    # load aaron's face to use as input for the reconstruction
    aaronface = load_aaron_face()
    plot_image(aaronface, 
        title=f"Domain Image {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray')

    domain_chunks = utils.Partition(aaronface, mode=MODE)
    # domain_chunks = utils.Partition(np.zeros([domain_image.shape[0], 
    #                                         domain_image.shape[1]]), mode=MODE)

    # Get image and metrics after 1 iterations
    start = timer()
    reconstructed_chunks = decode_svd(codebook, domain_chunks)
    end = timer()
    print("Time to decode 1 iterations is " + str(end - start))
    reconstructed_chunks_1iter = copy.deepcopy(reconstructed_chunks.image)
    reconstructed_chunks1 = copy.deepcopy(reconstructed_chunks)
    domain_chunks_original = copy.deepcopy(domain_chunks)
    
    # Get image and metrics after 2 iterations
    for i in range(4):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
    reconstructed_chunks_10iter = copy.deepcopy(reconstructed_chunks.image)

    # Get image and metrics after 10 iterations
    domain_chunks = copy.deepcopy(domain_chunks_original)
    reconstructed_chunks = decode_svd(codebook, domain_chunks)
    psnr[0] = metrics.peak_signal_noise_ratio(orig,reconstructed_chunks.image, data_range=255)
    ssi[0] = metrics.structural_similarity(orig,reconstructed_chunks.image, data_range=255)
    err[0] = metrics.mean_squared_error(orig,reconstructed_chunks.image)
    for i in range(9):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
        psnr[i+1] = metrics.peak_signal_noise_ratio(orig,reconstructed_chunks.image, data_range=255)
        ssi[i+1] = metrics.structural_similarity(orig,reconstructed_chunks.image, data_range=255)
        err[i+1] = metrics.mean_squared_error(orig,reconstructed_chunks.image)

    # Time 10 iterations
    domain_chunks = copy.deepcopy(domain_chunks_original)
    start = timer()
    reconstructed_chunks = decode_svd(codebook, domain_chunks)
    for i in range(9):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
    end = timer()
    print("Time to decode 10 iterations is " + str(end - start))

    reconstructed_chunks_100iter = copy.deepcopy(reconstructed_chunks.image)

    # plot the result
    plot_image(reconstructed_chunks_1iter, 
        title=f"Reconstructed Image 1 iteration 64x64 ({MODE})", 
        cmap='gray')
    plot_image(reconstructed_chunks_10iter, 
        title=f"Reconstructed Image 5 iterations 64x64 ({MODE})",
        cmap='gray')
    plot_image(reconstructed_chunks_100iter, 
        title=f"Reconstructed Image 10 iterations 64x64 ({MODE})",
        cmap='gray')
    print("PSNR: ",psnr[0],psnr[5],psnr[9])
    print("MSE: ",err[0],err[5],err[9])
    print("SSI: ",ssi[0],ssi[5],ssi[9])
    print(reconstructed_chunks_100iter.shape)

    # Plot PSNR and MSE
    x = np.arange(1,11)
    plt.figure(figsize=(10,7))
    plt.plot(x,psnr)
    plt.suptitle("PSNR", y=0.93, fontsize=23)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("dB", fontsize=16)
    plt.show()

    plt.figure(figsize=(10,7))
    plt.suptitle("Mean Squared Error", y=0.93, fontsize=23)
    plt.plot(x,err)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("Mean Squared Error", fontsize=16)
    plt.show()

    plt.figure(figsize=(10,7))
    plt.suptitle("Structural Similarity", y=0.93, fontsize=23)
    plt.plot(x,ssi)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.ylabel("Structural Similarity Index", fontsize=16)
    plt.show()

 ###############################################################################

