
import cv2
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass
from matplotlib.axes import Axes
from tqdm import tqdm
import h5py

import re
from pathlib import Path

@dataclass
class ImagePair:

    img1: np.ndarray
    img2: np.ndarray

    H: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray

    i: int = None
    j: int = None


def orb_flann_factory(nfeatures=5000):
    FLANN_INDEX_LSH = 6
    detector = cv2.ORB_create(nfeatures=nfeatures)
    flann = cv2.FlannBasedMatcher(
        indexParams={
            'algorithm': FLANN_INDEX_LSH,
            'table_number': 6,
            'key_size': 12,
            'multi_probe_level': 1},
        searchParams={'checks': 50},
    )
    return detector, flann


def sift_flann_factory(nfeatures=5000):
    FLANN_INDEX_KDTREE = 1
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    flann = cv2.FlannBasedMatcher(
        indexParams={
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5},
        searchParams={'checks': 50},
    )
    return detector, flann


class ImageMatcher:

    def __init__(self, images: list[np.ndarray],
                 human_masks: list[np.ndarray],
                 keyframe_interval: int = 30):

        self.images = images
        if human_masks is None:
            print('Warning: no human segmentation masks provided. '
                  'Loop closures may be unreliable.')
            self.human_masks = [np.full_like(images[0], 0, dtype=np.uint8)] * len(images)
        else:
            self.human_masks = human_masks.astype(np.uint8)
        self.keyframe_interval = keyframe_interval

        self.orb_detector, self.orb_flann = orb_flann_factory(1000)
        self.sift_detector, self.sift_flann = sift_flann_factory(1000)

        self.orb_reliable = []
        self.orb_kds = []

        for i, img in enumerate(self.images):
            #kds = self.orb_detector.detectAndCompute(img, 1 - self.human_masks[i])
            #reliable = True
            #if len(kds[0]) < 100:
            #    kds = self.orb_detector.detectAndCompute(img, None)
            #    reliable = False
            kds = self.orb_detector.detectAndCompute(img, None)
            reliable = True

            self.orb_kds.append(kds)
            self.orb_reliable.append(reliable)

        self.sift_kds = []
        for i, img in enumerate(self.images):
            if i % keyframe_interval == 0:
                kds = self.sift_detector.detectAndCompute(img, 1 - self.human_masks[i])
                if len(kds[0]) < 40:
                    print(f'Warning: only {len(kds[0])} SIFT keypoints detected in keyframe {i}.')
                    self.sift_kds.append(None)
                else:
                    self.sift_kds.append(kds)
            else:
                self.sift_kds.append(None)

    def match(self, i: int, j: int,
              method: str='orb',
              min_match_count: int = 400,
              keep_percent: float = 1.0,
              ransac_reproj_thresh: float = 2.0,
              ransac_max_iters: int = 2000,
              verbose=False) -> ImagePair | None:

        flann = getattr(self, method + '_flann')
        kds = getattr(self, method + '_kds')

        kp1, des1 = kds[i]
        kp2, des2 = kds[j]

        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m in matches:
            # Sometimes OpenCV will return just 1 nearest neighbour,
            # so we cannot apply the ratio test. Skip such cases.
            if len(m) < 2:
                if verbose:
                    #print('Warning: insufficient neighbours for ratio test. Skipping.')
                    pass
                continue
            a, b = m
            if a.distance < 0.7 * b.distance:
                good.append(a)

        if len(good) < min_match_count:
            if verbose:
                print(f'Warning: {len(good)} matches after ratio test is below threshold.')
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_thresh,
            maxIters=ransac_max_iters,
            confidence=0.99,
        )

        if H is None:
            if verbose:
                print(f'Warning: failed to find homography.')
            return None

        mask = mask.astype(bool).ravel()
        src_pts = src_pts[mask][::int(1 / keep_percent)]
        dst_pts = dst_pts[mask][::int(1 / keep_percent)]

        assert len(src_pts) == len(dst_pts)

        if len(src_pts) < min_match_count:
            if verbose:
                print(f'Warning: {len(src_pts)} matches after homography RANSAC is below threshold.')
            return None

        return ImagePair(self.images[i], self.images[j], H, src_pts, dst_pts, i, j)

    def find_H(self, p0, p1, good):
        src_pts, dst_pts = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)

        assert len(src_pts) == len(dst_pts)

        if len(src_pts) < 50:
            print(f'Warning: {len(src_pts)} matches after optical flow is below threshold.')
            return None, None, None

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99,
        )

        if H is None:
            print(f'Warning: failed to find homography.')
            return None, None, None

        mask = mask.astype(bool).ravel()
        src_pts = src_pts[mask]
        dst_pts = dst_pts[mask]

        assert len(src_pts) == len(dst_pts)

        if len(src_pts) < 50:
            print(f'Warning: {len(src_pts)} matches after homography RANSAC is below threshold.')
            return None, None, None

        return H, src_pts, dst_pts

