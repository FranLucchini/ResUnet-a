import numpy as np
from utils import get_boundary_label, get_distance_label, binarize_matrix, split_pair_names, normalize_rgb
import tensorflow as tf
import cv2

class DataParser():

    def __init__(self, img_path, label_path, label_dict, val_img_path=None, val_label_path=None, batch_size=8, image_size=256, num_classes = 2, Test=False):
        self.img_path = img_path
        self.label_path = label_path
        # Get sorted filenames of folder   
        self.samples = split_pair_names(self.img_path, self.label_path)
        # Number of samples
        self.n_samples = len(self.samples)
        # List of IDs
        self.all_ids = list(range(self.n_samples))
        np.random.shuffle(self.all_ids)
        
        if not Test:
            # Throw error if validation is not included
            assert (val_img_path is not None) and (val_label_path is not None)
            self.val_img_path = val_img_path
            self.val_label_path = val_label_path
            self.val_samples = split_pair_names(self.val_img_path, self.val_label_path)
            self.validation_steps = len(self.val_samples)/(batch_size*2)
            self.validation_ids = list(range(len(self.val_samples)))

        self.training_ids = list(range(self.n_samples))
        self.batch_size = batch_size
        self.steps_per_epoch = self.n_samples/batch_size
        self.image_size = image_size
        self.label_dict = label_dict
        self.num_classes = num_classes
        self.is_test = Test

    def get_batch(self, batch):
        # batch (list of ids)

        images = []
        seg = []
        bound = []
        dist = []

        for b in batch:
    
            im = cv2.imread(self.samples[b][0])
            im = cv2.resize(im,(self.image_size,self.image_size))

            em = cv2.imread(self.samples[b][1], 0)
            em = cv2.resize(em,(self.image_size,self.image_size))
            em[em > 0] = 255
            em = np.stack([em, em, em], axis=-1)

            em = binarize_matrix(em, self.label_dict)

            em = tf.keras.utils.to_categorical(em, self.num_classes)

            im = im.astype(np.float32)
            im = normalize_rgb(im)
            images.append(im)

            # All multitasking labels are saved in one-hot
            # Segmentation
            seg.append(em.astype(np.float32))
                      
            # Boundary
            bound_label_h = get_boundary_label(em).astype(np.float32)
            bound.append(bound_label_h)
                      
            # Distance
            dist_label_h = get_distance_label(em).astype(np.float32)
            dist.append(dist_label_h)

        images = np.asarray(images)

        labels = {'segmentation': np.asarray(seg)}
        labels['boundary'] = np.asarray(bound)
        labels['distance'] = np.asarray(dist)

        return images, labels
