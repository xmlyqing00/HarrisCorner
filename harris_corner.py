import cv2
import argparse
import numpy as np
import time
from scipy.ndimage import maximum_filter

def get_args():
    parser = argparse.ArgumentParser(description='Harris Corner Detector')
    parser.add_argument('--img', type=str, default='test/test0.jpg',
        help='Path to the input image.')
    parser.add_argument('--k', type=float, default=0.04,
        help='Harris cornet detector balance factor.')
    parser.add_argument('--sigma', type=float, default=0.8,
        help='Sigma of Gaussian of derivatives.')
    parser.add_argument('--nms_radius', type=int, default=5,
        help='Radius of the non- size of Gaussian of derivatives.')
    return parser.parse_args()

def simple_nms(scores, nms_radius: int):
    """ 
        Fast Non-maximum suppression to remove nearby points 
        maximum_filter: Similar to the max_pool, return the 
            maximum value of each pixel's neighborhood 
            (blocksize = nms_radius)
    """
    
    assert(nms_radius >= 0)

    nms_radius = nms_radius * 2 + 1
    zeros = np.zeros_like(scores)
    max_mask = scores == maximum_filter(scores, nms_radius)

    # for _ in range(2):
    #     supp_mask = maximum_filter(max_mask.astype(float), nms_radius) > 0
    #     supp_scores = np.where(supp_mask, zeros, scores)
    #     new_max_mask = supp_scores == maximum_filter(supp_scores, nms_radius)
    #     max_mask = max_mask | (new_max_mask & (~supp_mask))
        
    return np.where(max_mask, scores, zeros)

# def get_ksize_by_simga(sigma):
#     """ By OpenCV Doc https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#Mat%20getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype) """

#     return int(((sigma - 0.8) / 0.3 + 1) * 2 + 1)

def viz(img, corner_list, title):

    img2 = img.copy()
    for y, x in zip(*corner_list):
        cv2.circle(img2, (x, y), 2, (255, 0, 0), 2)

    print(f'{title} numbers:', len(corner_list[0]))
    cv2.imshow(title, img2)


def compute_harris_by_DoG(img):

    print('\nCompute Harris Corner by Derivative of Gaussian.')
    time_st = time.time()

    # Step 1, color to grayscale conversion.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2, compute spatial derivative of gaussian.
    img_dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    img_dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

    # Step 3, construct structure tensor.
    img_dx2 = img_dx * img_dx
    img_dy2 = img_dy * img_dy
    img_dxdy = img_dx * img_dy
    img_dx2 = cv2.GaussianBlur(img_dx2, (0, 0), args.sigma)
    img_dy2 = cv2.GaussianBlur(img_dy2, (0, 0), args.sigma)
    img_dxdy = cv2.GaussianBlur(img_dxdy, (0, 0), args.sigma)

    # Step 4, calc Harris response
    det = img_dx2 * img_dy2 - img_dxdy * img_dxdy
    trace = img_dx2 + img_dy2
    R = det - args.k * trace ** 2

    # Dynamic threshold followed by OpenCV
    valid_pos = R > 0.01 * R.max()
    corner_list_raw = valid_pos.nonzero()

    # Step 5, non-maximum suppression
    R = simple_nms(R, args.nms_radius)
    valid_pos = R > 0.01 * R.max()
    corner_list_after_nms = valid_pos.nonzero()
    print(f'Elapsed time: {time.time() - time_st:.6f}s')

    # Compare with OpenCV implementation
    R_opencv = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    valid_pos = R_opencv > 0.01 * R_opencv.max()
    corner_list_opencv = valid_pos.nonzero()

    # Visualization
    viz(img, corner_list_raw, 'Harris corner raw')
    viz(img, corner_list_after_nms, 'Harris corner after nms')
    viz(img, corner_list_opencv, 'Harris corner by OpenCV')
    cv2.waitKey()


if __name__ == '__main__':

    args = get_args()

    # Step 0, read img
    img = cv2.imread(args.img)

    # Compute Harris Corner
    compute_harris_by_DoG(img)

    

