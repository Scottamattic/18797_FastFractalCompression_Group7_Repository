from context import fractal
from fractal.dataloader import Loader
from fractal import utils
from fractal.plotting import plot_image
from fractal.coding import encode_svd, decode_svd
from fractal.coding import encode, decode
import copy
from skimage.transform import rescale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import matplotlib as mpl

MODE = 'equipartition4'
SCALE = 1
BLOCK_SIZE = 16
# percentage*block_size determines how many coefficients to retain
# this is where the compression happens. For example, 0.75*BLOCK_SIZE
# will drop 25% of coefficients. Similarly, 0.33*BLOCK_SIZE will zero 
# out 67% (rounded to the nearest integer) coefficients
COMPRESSION_FACTOR = int(0.25*BLOCK_SIZE) 

def PSNR(original, compressed): 
    from math import log10, sqrt 

    # sometimes recovered signal has out-of-bound values
    compressed = np.clip(compressed, 0, 255)
    
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def load_reconstruction_input():
    facedir = Path(__file__).resolve().parent.parent / "data"
    face = Image.open(list(facedir.glob("mandrill.jpg"))[0])
    face = np.asarray(face.getdata()).reshape(512,512)
    face = rescale(face, 0.25)
    return face

def load_mandrill():
    facedir = Path(__file__).resolve().parent.parent / "data"
    face = Image.open(list(facedir.glob("mandrill.jpg"))[0])
    face = np.asarray(face.getdata()).reshape(512,512)
    face = rescale(face, 0.25)
    return face

def load_texture():
    facedir = Path(__file__).resolve().parent.parent / "data"
    face = Image.open(list(facedir.glob("texture1.png"))[0])
    face = np.asarray(face.getdata()).reshape(240,240)
    face = rescale(face, 0.25)
    return face

def load_aaron():
    facedir = Path(__file__).resolve().parent.parent / "data" / "faces"
    face = Image.open(list(facedir.glob("Aaron_Eckhart_0001.pgm"))[0])
    face = np.asarray(face.getdata()).reshape(64,64)
    face = rescale(face, 0.25)
    return face


def FFIC(domain_chunks, range_chunks, inpface):
    ############################################################################
    # encode the range image using svd encoding 
    ############################################################################
    
    # encode image
    start = time.time()
    codebook = encode_svd(domain_chunks, range_chunks, verbose=False)
    svd_encoding_time = time.time() - start

    # for svd only we implement compression by 
    # dropping coefficients in the codebook
    codebook[:, COMPRESSION_FACTOR:] = 0

    # Use an input face to use as input for the reconstruction
    domain_chunks = utils.Partition(inpface, mode=MODE)

    # initialize psnr
    psnr = np.zeros(10);

    # decode image 100 times
    start = time.time()
    reconstructed_chunks = decode_svd(codebook, domain_chunks)
    reconstructed_chunks_1 = copy.deepcopy(reconstructed_chunks)
    psnr[0] = PSNR(range_image, reconstructed_chunks.image)
    
    for i in range(9):
        rec_dim = rescale(reconstructed_chunks.image, 0.5) 
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
        psnr[i+1] = PSNR(range_image, reconstructed_chunks.image)
    
    svd_decoding_time = time.time()-start

    # plot_image(reconstructed_chunks.image, 
    #     title=f"Reconstructed Image (SVD Encoding) \n {reconstructed_chunks.image.shape[0]}x{reconstructed_chunks.image.shape[0]}, {COMPRESSION_FACTOR}/{BLOCK_SIZE} Compression", 
    #     cmap='gray', y=0.97)

    ############################################################################
    # encoding and decoding time results
    ############################################################################

    print(f"svd mode encoding: {svd_encoding_time}")
    print(f"svd mode decoding: {svd_decoding_time}")

    ############################################################################
    # psnr results
    ############################################################################
    # psnr = PSNR(range_image, reconstructed_chunks.image)
    print(f"PSNR: {psnr} \t Coefficients Retained: {COMPRESSION_FACTOR}/{BLOCK_SIZE}")
    return psnr

if __name__ == '__main__':

    ############################################################################
    # load images
    ############################################################################
    images = Loader(optpath="", regex="lena.jpg")
    range_image = images.get_image(0, scale_factor=SCALE)

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

    compression_ratios = np.arange(1/16,16/16,1/16)
    num_ratios = len(compression_ratios)


    ############################################################################
    # Use Mandrill for reconstruction
    ############################################################################
    # Initialize psnr matrix
    psnr_all_mandrill = np.zeros((num_ratios,10))

    # load an input face to use as input for the reconstruction
    inpface = load_mandrill()
    plot_image(inpface, 
        title=f"Reconstruction Input {inpface.shape[0]}x{inpface.shape[1]}",
        cmap='gray', y=0.97)

    for i in range(num_ratios):
        ratio = compression_ratios[i]
        COMPRESSION_FACTOR = int(ratio*BLOCK_SIZE) 
        psnr_all_mandrill[i,:] = FFIC(domain_chunks, range_chunks, inpface)


    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

    # Plot PSNR vs Compression
    plt.figure(figsize=(12,7))
    num_iter = np.arange(1,11)
    plt.plot(num_iter,np.transpose(psnr_all_mandrill))
    plt.suptitle("Reconstruction with Mandrill: PSNR vs Compression Ratio", y=0.93, fontsize=16)
    plt.xlabel("Number of Iterations", fontsize=16)
    plt.xticks(np.arange(0, 11, step=2), fontsize=16)
    plt.ylabel("PSNR (dB)", fontsize=16)
    # plt.yscale("log")
    plt.legend(["16:1","8:1","16:3","4:1","16:5","8:3","16:7","2:1","16:9",
        "8:5","16:11","4:3","16:13","8:7","16:15"], fontsize=10,
        loc='center left', bbox_to_anchor=(1, 0.5), title="Compression\n Ratios")
    plt.savefig("../results/mandrill_psnr.png")
    plt.show()


    ############################################################################
    # Use Aaron for reconstruction
    ############################################################################
    # Initialize psnr matrix
    psnr_all_aaron = np.zeros((num_ratios,10))

    # load an input face to use as input for the reconstruction
    inpface = load_aaron()
    plot_image(inpface, 
        title=f"Reconstruction Input {inpface.shape[0]}x{inpface.shape[1]}",
        cmap='gray', y=0.97)

    for i in range(num_ratios):
        ratio = compression_ratios[i]
        COMPRESSION_FACTOR = int(ratio*BLOCK_SIZE) 
        psnr_all_aaron[i,:] = FFIC(domain_chunks, range_chunks, inpface)

    # Plot PSNR vs Compression
    plt.figure(figsize=(12,7))
    num_iter = np.arange(1,11)
    plt.plot(num_iter,np.transpose(psnr_all_aaron))
    plt.suptitle("Reconstruction with Aaron: PSNR vs Compression Ratio", y=0.93, fontsize=16)
    plt.xlabel("Number of Iterations")
    plt.xticks(np.arange(0, 11, step=2))
    plt.ylabel("PSNR (dB)")
    # plt.yscale("log")
    plt.legend(["16:1","8:1","16:3","4:1","16:5","8:3","16:7","2:1","16:9",
        "8:5","16:11","4:3","16:13","8:7","16:15"],
        loc='center left', bbox_to_anchor=(1, 0.5), title="Compression\n Ratios")
    plt.savefig("../results/aaron_psnr.png")
    plt.show()


    # ############################################################################
    # # Use Texture for reconstruction
    # ############################################################################
    # Initialize psnr matrix
    psnr_all_texture = np.zeros((num_ratios,10))

    # load an input face to use as input for the reconstruction
    inpface = load_texture()
    plot_image(inpface, 
        title=f"Reconstruction Input {inpface.shape[0]}x{inpface.shape[1]}",
        cmap='gray', y=0.97)

    for i in range(num_ratios):
        ratio = compression_ratios[i]
        COMPRESSION_FACTOR = int(ratio*BLOCK_SIZE) 
        psnr_all_texture[i,:] = FFIC(domain_chunks, range_chunks, inpface)

    # Plot PSNR vs Compression
    plt.figure(figsize=(12,7))
    num_iter = np.arange(1,11)
    plt.plot(num_iter,np.transpose(psnr_all_texture))
    plt.suptitle("Reconstruction with Texture: PSNR vs Compression Ratio", y=0.93, fontsize=16)
    plt.xlabel("Number of Iterations")
    plt.xticks(np.arange(0, 11, step=2))
    plt.ylabel("PSNR (dB)")
    # plt.yscale("log")
    plt.legend(["16:1","8:1","16:3","4:1","16:5","8:3","16:7","2:1","16:9",
        "8:5","16:11","4:3","16:13","8:7","16:15"],
        loc='center left', bbox_to_anchor=(1, 0.5), title="Compression\n Ratios")
    plt.savefig("../results/texture_psnr.png")
    plt.show()