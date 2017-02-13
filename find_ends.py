# 03-24-14 Creation by RW

import matplotlib, copy, skimage, mahotas
from skimage import io, filter, morphology, draw

import matplotlib.pyplot as plt
import numpy as np

# 8 potential structuring elements that mark the end of a connected skeletal segment
end1 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])

end2 = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]])

end3 = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0]])

end4 = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]])

end5 = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0]])

end6 = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0]])

end7 = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]])

end8 = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

xbranch0 = np.array([[1,0,1],[0,1,0],[1,0,1]])
xbranch1 = np.array([[0,1,0],[1,1,1],[0,1,0]])
tbranch0 = np.array([[0,0,0],[1,1,1],[0,1,0]])
tbranch1 = np.flipud(tbranch0)
tbranch2 = tbranch0.T
tbranch3 = np.fliplr(tbranch2)
tbranch4 = np.array([[1,0,1],[0,1,0],[1,0,0]])
tbranch5 = np.flipud(tbranch4)
tbranch6 = np.fliplr(tbranch4)
tbranch7 = np.fliplr(tbranch5)  
ybranch0 = np.array([[1,0,1],[0,1,0],[2,1,2]])
ybranch1 = np.flipud(ybranch0)
ybranch2 = ybranch0.T
ybranch3 = np.fliplr(ybranch2)
ybranch4 = np.array([[0,1,2],[1,1,2],[2,2,1]])
ybranch5 = np.flipud(ybranch4)
ybranch6 = np.fliplr(ybranch4)
ybranch7 = np.fliplr(ybranch5)

# Returns the chromosome skeletons of an image after thresholding and cleaning
def skel_chromosomes(img):
    # first, create a copy of the redchannel and apply threshold
    redchannel = copy.copy(img)
    redchannel = img[:,:,0]

    thresh = skimage.filter.threshold_otsu(redchannel)
    binary = redchannel >= thresh

    # clean binarized image with morphological opening and then skeletonize
    square = skimage.morphology.square(4)
    cleaned = skimage.morphology.binary_opening(binary, square)

    skeleton = skimage.morphology.skeletonize(cleaned)

    return skeleton
    
# Takes a skeletonized image and returns the end points in a list of coordinate tuples 
def find_ends(skeleton):
    # look for the different ending structures
    # hitmiss returns 1 at positions in skel that match the end structuring elements
    ends_matrix = mahotas.hitmiss(skeleton, end1) + \
                  mahotas.hitmiss(skeleton, end2) + \
                  mahotas.hitmiss(skeleton, end3) + \
                  mahotas.hitmiss(skeleton, end4) + \
                  mahotas.hitmiss(skeleton, end5) + \
                  mahotas.hitmiss(skeleton, end6) + \
                  mahotas.hitmiss(skeleton, end7) + \
                  mahotas.hitmiss(skeleton, end8)    

    end_coordinates = np.nonzero(ends_matrix)

    # zip the arrays up into a list of coordinate tuples and return
    return list(zip(end_coordinates[0], end_coordinates[1]))

def find_branches(skeleton):
    branch_matrix = mahotas.hitmiss(skeleton, xbranch0) + \
                    mahotas.hitmiss(skeleton, xbranch1) + \
                    mahotas.hitmiss(skeleton, tbranch0) + \
                    mahotas.hitmiss(skeleton, tbranch1) + \
                    mahotas.hitmiss(skeleton, tbranch2) + \
                    mahotas.hitmiss(skeleton, tbranch3) + \
                    mahotas.hitmiss(skeleton, tbranch4) + \
                    mahotas.hitmiss(skeleton, tbranch5) + \
                    mahotas.hitmiss(skeleton, tbranch6) + \
                    mahotas.hitmiss(skeleton, tbranch7) + \
                    mahotas.hitmiss(skeleton, ybranch0) + \
                    mahotas.hitmiss(skeleton, ybranch1) + \
                    mahotas.hitmiss(skeleton, ybranch2) + \
                    mahotas.hitmiss(skeleton, ybranch3) + \
                    mahotas.hitmiss(skeleton, ybranch4) + \
                    mahotas.hitmiss(skeleton, ybranch5) + \
                    mahotas.hitmiss(skeleton, ybranch6) + \
                    mahotas.hitmiss(skeleton, ybranch7)

    branch_coordinates = np.nonzero(branch_matrix)

    return branch_matrix
#    return list(zip(branch_coordinates[0], branch_coordinates[1]))

# Plots an image with yellow circles at the coordinates, expecting list of tuples
# in the form of (row, column) for coords
def yellow_highlight(img, coords):
    # create a copy of the image and draw yellow circles at the coordinates
    highlight = copy.copy(img)

    for position in coords:
        rr, cc = skimage.draw.circle(position[0], position[1], 5)
        highlight[rr, cc, 0] = 255
        highlight[rr, cc, 1] = 255

    plt.imshow(highlight)


img = io.imread("Z:/RecombinationImageAnalysis/testimages/2432/2432_1_1_REV.tif")
skeleton_img = skel_chromosomes(img)
coordinates = find_ends(skeleton_img)

yellow_highlight(img, coordinates)
