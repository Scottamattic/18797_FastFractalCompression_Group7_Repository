import numpy as np
from fractal import utils
from sklearn.decomposition import NMF
from numba import njit

@njit
def distortion(range_block, domain_block, alpha, t0):
    return np.sum((range_block-((alpha * domain_block) + t0))**2)

@njit
def compute_alpha(domain_block, range_block):
    nrows, ncols = domain_block.shape
    if nrows != ncols:
        raise Exception("Domain block must be square")
        
    N = nrows*ncols
        
    # estimate the value of alpha
    alpha = N*np.sum(domain_block * range_block) - \
               np.sum(domain_block)*np.sum(range_block)
    
    # normalization
    alpha = alpha/(N*np.sum(domain_block**2) - np.sum(domain_block)**2)
    return alpha

@njit
def compute_t0(domain_block, range_block, alpha):
    nrows, ncols = domain_block.shape
    if nrows != ncols:
        raise Exception("Domain block must be square")
    
    N = nrows*ncols

    t0 = (1/N) * (np.sum(range_block) - alpha*np.sum(domain_block))
    return t0


def permute(domain_block):
    transformations = []

    # try all rotations of the block
    for i in range(4):
        domain_block = np.rot90(domain_block)
        transformations.append(domain_block)

    # flip the block
    domain_block = np.fliplr(domain_block)

    # try all rotations of the flipped block
    for i in range(4):
        domain_block = np.rot90(domain_block)
        transformations.append(domain_block)

    return transformations


def encode_block(didx, domain_block_ref, range_block):
    # import os
    # print(os.getpid())
    best = []
    lowest_err = np.infty
    for permtype, domain_block in enumerate(permute(domain_block_ref)):
        # compute alpha and t0; the transform values
        alpha = compute_alpha(domain_block, range_block)
        t0 = compute_t0(domain_block, range_block, alpha)

        # compute the error for this choice of domain block
        # if this one is not suitabile, we will try another
        dist = distortion(range_block, domain_block, alpha, t0)

        if dist < lowest_err:
            lowest_err = dist
            best = [lowest_err, didx, permtype, alpha, t0]
    
    return best


def encode(domain_blocks, range_blocks, multiproc=True, verbose=False):
    num_words = 4
    N = len(range_blocks)
    codebook = np.zeros([N, num_words])

    encoder = utils.Parallelize(encode_block, multiproc)

    didxs = list(range(len(domain_blocks)))

    for codeidx, rb in enumerate(range_blocks):
        candidate_codes = encoder(didxs, domain_blocks, [rb]*len(domain_blocks))

        codebook[codeidx] = sorted(candidate_codes)[0][1:]

        if verbose:
            print(f"processed block {codeidx}/{N}")
    
    return codebook

@njit
def svd_gpu(X):
    u, s, vh = np.linalg.svd(X)
    return u

def encode_svd(domain_blocks, range_blocks, verbose=False):
    # TODO reinstate error checks
    # if domain_blocks.mode != 'equipartition4' or \
    #    range_blocks.mode != 'equipartition4':
    #    raise Exception("encode_svd only supports 'equipartition4' partitioning")

    num_weights = len(domain_blocks[0].ravel())

    X = np.zeros([num_weights, len(domain_blocks)])

    for i in range(len(domain_blocks)):
        X[:, i] = np.ravel(domain_blocks[i])

    u = svd_gpu(X)
    
    N = len(range_blocks)
    codebook = np.zeros([N, num_weights])

    for codeidx, rb in enumerate(range_blocks):
        if verbose:
            print(f'processed {codeidx}/{N} blocks')
        codebook[codeidx, :] = np.dot(u.T, rb.reshape(num_weights,1)).ravel()
    
    return codebook

def encode_nmf(domain_blocks, range_blocks, verbose=False):
    # TODO reinstate error checks
    # if domain_blocks.mode != 'equipartition4' or \
    #    range_blocks.mode != 'equipartition4':
    #    raise Exception("encode_svd only supports 'equipartition4' partitioning")

    num_weights = len(domain_blocks[0].ravel())

    X = np.zeros([num_weights, len(domain_blocks)])

    for i in range(len(domain_blocks)):
        X[:, i] = np.ravel(domain_blocks[i])

    model = NMF(n_components=16, init='random', random_state=0)
    _ = model.fit_transform(X)
    u = model.components_[:,:16]
    ut = np.linalg.pinv(u)
    print(ut.shape)

    N = len(range_blocks)
    codebook = np.zeros([N, num_weights])

    for codeidx, rb in enumerate(range_blocks):
        if verbose:
            print(f'processed {codeidx}/{N} blocks')
        codebook[codeidx, :] = np.dot(ut, rb.reshape(num_weights,1)).ravel()
    
    return codebook


def decode_nmf(codebook, domain_blocks):
    # if domain_blocks.mode != 'equipartition4' or\
    #    domain_blocks.mode !=:
    #    raise Exception("encode_svd only supports 'equipartition4' partitioning")
    num_weights = len(domain_blocks[0].ravel())
    X = np.zeros([num_weights, len(domain_blocks)])

    for i in range(len(domain_blocks)):
        X[:, i] = np.ravel(domain_blocks[i])
    X = np.abs(X)
    model = NMF(n_components=16, init='random', random_state=0)
    _ = model.fit_transform(X)
    u = model.components_[:,:16]

    N = int(np.sqrt(len(codebook)))
    img = utils.init_range_image(N, domain_blocks)
    range_blocks = utils.Partition(img, mode=domain_blocks.mode)

    nrows = int(np.sqrt(num_weights))
    ncols = nrows
    for ridx, weights in enumerate(codebook):
        range_blocks[ridx] = np.dot(u, weights[:, np.newaxis])\
                             .reshape(nrows, ncols)

    return range_blocks

def decode_svd(codebook, domain_blocks):
    # if domain_blocks.mode != 'equipartition4' or\
    #    domain_blocks.mode !=:
    #    raise Exception("encode_svd only supports 'equipartition4' partitioning")

    num_weights = len(domain_blocks[0].ravel())
    X = np.zeros([num_weights, len(domain_blocks)])

    for i in range(len(domain_blocks)):
        X[:, i] = np.ravel(domain_blocks[i])

    u = svd_gpu(X)

    N = int(np.sqrt(len(codebook)))
    img = utils.init_range_image(N, domain_blocks)
    range_blocks = utils.Partition(img, mode=domain_blocks.mode)

    nrows = int(np.sqrt(num_weights))
    ncols = nrows
    for ridx, weights in enumerate(codebook):
        range_blocks[ridx] = np.dot(u, weights[:, np.newaxis])\
                             .reshape(nrows, ncols)

    return range_blocks

def decode(codebook, domain_blocks):
    """Decodes the codebook into estimates of the range blocks"""

    N = int(np.sqrt(len(codebook)))
    if N != np.sqrt(len(codebook)):
        raise Exception("Codebook size must correspond to a square range image")
    
    # create a range block partition to populate
    img = utils.init_range_image(N, domain_blocks)

    range_blocks = utils.Partition(img, mode=domain_blocks.mode)

    for ridx, (didx, permtype, alpha, t0) in enumerate(codebook):
        permtype = int(permtype)
        dref = domain_blocks[didx]
        db = permute(dref)[permtype]
        range_blocks[ridx] = (alpha * db) + t0
        
    return range_blocks

