import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import skimage, mahotas
from skimage import io, filter, morphology, draw, graph, feature

from scipy import ndimage
from skimage.morphology import disk
from skimage.morphology import square
   
class Blobject:
    def __init__(self, blob_mask, binary_blue, ends_matrix):
        self.blob_mask = blob_mask

        self.ends_matrix = ma.masked_where(blob_mask == 0, ends_matrix)
        self.ends_matrix = self.ends_matrix.filled(0)

        self.binary_blue = ma.masked_where(blob_mask == 0, binary_blue)
        self.binary_blue = self.binary_blue.filled(0)

        self.end_positions = []
        self.centromere_positions = []
        
        self.assign_ends()
        self.assign_centromeres()
                
    def assign_ends(self):
        end_coordinates = np.nonzero(self.ends_matrix)
        self.end_positions = list(zip(end_coordinates[0], end_coordinates[1]))

    def assign_centromeres(self):
        blue_labels = ndimage.label(self.binary_blue)
        
        dot_list = [np.nonzero(blue_labels[0] == dot_number) for dot_number in range(1, blue_labels[1] + 1)]
        centers = [(int(np.mean(dot[0])), int(np.mean(dot[1]))) for dot in dot_list]
        
        for dot in centers:
            distance = [((dot[0] - end[0])**2 + (dot[1] - end[1])**2)**(0.5) for end in self.end_positions]
            val, idx = min((val, idx) for (idx, val) in enumerate(distance))

            centro_coord = self.end_positions[idx]
            del self.end_positions[idx]
            self.centromere_positions.append(centro_coord)

    def chromosome_conversion(self, red_mask):
        if(len(self.centromere_positions) < 1):
            return []
        if(len(self.end_positions) < 1):
            return []
        else:
            red_masked = ma.masked_where(self.blob_mask == 0, red_mask)
            red_masked = red_masked.filled(0)
            
            ends = list(self.end_positions)
            chromosomes = []

            weight_array = np.array(red_masked).astype(np.int32)
            max_weight = np.amax(weight_array)
            
            weight_array = max_weight - weight_array
            weight_array[weight_array == max_weight] = 30 * max_weight
            
            last_centromere = None
            for last_centromere in self.centromere_positions: pass

            for centromere in self.centromere_positions:
                if(centromere == last_centromere):
                    max_distance = 0
                else:
                    min_curvature = np.inf

                chosen_end = ()
                chosen_idx = np.array([])

                if not ends:
                    break
                
                for end in ends:
                    idx, wt = skimage.graph.route_through_array(weight_array, centromere, end)
                    idx = np.array(idx).T

                    dx = np.convolve(idx[0], [1/60.0, -3/20.0, 3/4.0, 0, -3/4.0, 3/20.0, -1/60.0], 'valid')
                    dy = np.convolve(idx[1], [1/60.0, -3/20.0, 3/4.0, 0, -3/4.0, 3/20.0, -1/60.0], 'valid')

                    raw_curvature = sum([(x**2 + y**2)**(0.5) for x,y in zip(dx,dy)])
                    euclidean_distance = np.linalg.norm(np.array(centromere) - np.array(end))
                    curvature = raw_curvature / euclidean_distance

                    if(centromere == last_centromere):
                        if(euclidean_distance > max_distance):
                            max_distance = euclidean_distance
                            chosen_end = end
                            chosen_idx = idx
                    elif curvature < min_curvature:
                        min_curvature = curvature
                        chosen_end = end
                        chosen_idx = idx
                
                path = np.zeros_like(self.blob_mask)
                path[chosen_idx[0], chosen_idx[1]] = 1
                new_mask = skimage.morphology.binary_dilation(path, disk(5))
                new_mask = ma.masked_where(new_mask == 0, self.blob_mask)
                new_mask = new_mask.filled(0)

                ends.remove(chosen_end)
                chromosomes.append(Chromosome(new_mask, centromere, chosen_end, self))

            return chromosomes
   
class Chromosome:
    def __init__(self, mask, centromere_position, end_position, parent):
        self.mask = mask
        self.centromere_position = centromere_position
        self.end_position = end_position
        self.parent = parent

        # Skeletonizing taking some time, let's take it out for now 4-16-14
        #self.skeleton = skimage.morphology.skeletonize(mask)
        #self.skeleton_size = sum(sum(self.skeleton))
        
        self.binary_size = sum(sum(self.mask))


class CellImage():
    # Public variables: redchannel, greenchannel, blue channel, binary_red, binary_blue, blobjects, chromosomes, interferance
    def __init__(self, im_path):
        self.img = skimage.io.imread(im_path)
        
        self.redchannel   = self.img[:,:,0]
        self.greenchannel = self.img[:,:,1]
        self.bluechannel  = self.img[:,:,2]

        r_thresh = skimage.filter.threshold_otsu(self.redchannel)
        self.binary_red = self.redchannel > r_thresh
        self.binary_red = skimage.morphology.binary_opening(self.binary_red, square(4))

        green_grad = skimage.filter.rank.gradient(self.greenchannel, disk(7))
        g_thresh = skimage.filter.threshold_otsu(green_grad)
        self.binary_green = green_grad > g_thresh
        
        blue_grad = skimage.filter.rank.gradient(self.bluechannel, disk(7))
        b_thresh = skimage.filter.threshold_otsu(blue_grad)
        self.binary_blue = blue_grad > b_thresh
        self.binary_blue = skimage.morphology.binary_opening(self.binary_blue, square(4))
        self.binary_blue = skimage.morphology.binary_closing(self.binary_blue, square(4))

        b_distance = ndimage.distance_transform_edt(self.binary_blue)
        b_local_max = skimage.feature.peak_local_max(b_distance, indices = False, footprint = np.ones((10,10)), labels = self.binary_blue)
        b_markers = ndimage.label(b_local_max)[0]
        self.binary_blue = skimage.morphology.watershed(-b_distance, b_markers, mask = self.binary_blue)

        e_matrix = self.find_ends(self.binary_red)

        blob_labels = self.label_blobs(self.redchannel)
        blob_masks = [blob_labels[0] == blob_number for blob_number in range(1, blob_labels[1]+1)]

        self.blobjects = [Blobject(x_mask, self.binary_blue, e_matrix) for x_mask in blob_masks]

        self.chromosomes = []      
        for blobject in self.blobjects:
            self.chromosomes += blobject.chromosome_conversion(self.redchannel)

    #
    #
    #
    #
    def reset_chromosomes(self):
        for idx, chromosome in reversed(list(enumerate(self.chromosomes))):
            if chromosome.parent:
                del self.chromosomes[idx]
                                
        for blobject in self.blobjects:
            self.chromosomes += blobject.chromosome_conversion(self.redchannel)
    
    def label_blobs(self, redchannel):
        thresh = skimage.filter.threshold_otsu(redchannel)
        binary = redchannel >= thresh

        cleaned = skimage.morphology.binary_opening(binary, square(4))
        labels = ndimage.label(cleaned)

        return labels

    def find_ends(self, binary_red):
    # 8 potential structuring elements that mark the end of a connected skeletal segment
        end1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        end2 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        end3 = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        end4 = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        end5 = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        end6 = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        end7 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        end8 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])

        skeleton = skimage.morphology.skeletonize(binary_red)

        ends_matrix = mahotas.hitmiss(skeleton, end1) + \
                      mahotas.hitmiss(skeleton, end2) + \
                      mahotas.hitmiss(skeleton, end3) + \
                      mahotas.hitmiss(skeleton, end4) + \
                      mahotas.hitmiss(skeleton, end5) + \
                      mahotas.hitmiss(skeleton, end6) + \
                      mahotas.hitmiss(skeleton, end7) + \
                      mahotas.hitmiss(skeleton, end8)

        return ends_matrix
