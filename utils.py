
import cv2
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass
from matplotlib.axes import Axes
from tqdm import tqdm
import h5py

from scipy.spatial.transform import Rotation as Rot

import copy
import re
import os
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


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 2000,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


class ImageMatcher:

    def __init__(self, images: list[np.ndarray],
                 human_masks: list[np.ndarray],
                 keyframe_interval: int = 30):

        self.images = images
        self.human_masks = human_masks.astype(np.uint8)
        self.keyframe_interval = keyframe_interval

        self.orb_detector, self.orb_flann = sift_flann_factory(2000)#orb_flann_factory(1000)
        self.sift_detector, self.sift_flann = sift_flann_factory(2000)

        self.orb_reliable = []
        self.orb_kds = []

        for i, img in enumerate(self.images):
            kds = self.orb_detector.detectAndCompute(img, 1 - self.human_masks[i])
            reliable = True
            if len(kds[0]) < 100:
                kds = self.orb_detector.detectAndCompute(img, None)
                reliable = False

            self.orb_kds.append(kds)
            self.orb_reliable.append(reliable)

        self.sift_kds = []
        for i, img in enumerate(self.images):
            if i % keyframe_interval == 0:
                kds = self.sift_detector.detectAndCompute(img, 1 - self.human_masks[i])
                if len(kds[0]) < 50:
                    self.sift_kds.append(None)
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

    def lk_track(self) -> list[ImagePair]:
        image_pairs = []
        tracks = []
        prev_frame = None
        track_len = 4
        detect_interval = 2
        for i in tqdm(range(0, len(self.images))):
            frame = self.images[i]
            if len(tracks) > 0:
                img0, img1 = prev_frame, frame
                p0 = np.float32([t[-1] for t in tracks]).reshape(-1, 1, 2)
                p1, *_ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, *_ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []

                H, sp, dp = self.find_H(p0, p1, good)

                if H is None:
                    print(f'Warning: failed to find homography between frame {i-1} to {i}.')
                else:
                    image_pairs.append(ImagePair(img0, img1, H, sp, dp, i-1, i))

                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                tracks = new_tracks

            if i % detect_interval == 0:
                mask = np.zeros_like(frame)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                #p = cv2.goodFeaturesToTrack(frame, mask=mask, **feature_params)
                p = self.orb_detector.detect(frame, mask=mask)
                p = np.array([kp.pt for kp in p])
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
            prev_frame = frame
        return image_pairs


def load_video(path: str, grayscale: bool = True,
               frame_segment: tuple[int, int] = (0, 99999)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []

    for i in range(0, frame_segment[0] + frame_segment[1]):

        ret, frame = cap.read()
        if not ret:
            break

        if i < frame_segment[0]:
            #frames.append(None)
            continue

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale else frame)

    cap.release()
    return frames


def project_points(H, pts, keep_z=False):

    assert H.shape == (3, 3)
    assert len(pts.shape) == 2 and pts.shape[1] in [2, 3]
    if pts.shape[1] == 2:
        o = np.ones([pts.shape[0], 1])
        pts_h = np.concatenate([pts, o], axis=1)
    else:
        pts_h = pts
    pts_p = pts_h @ H.T
    if keep_z:
        return pts_p
    return pts_p[:, :2] / pts_p[:, 2:3]


def reprojection_error(H, src_pts, dst_pts, plot=False):

    proj_pts = project_points(H, src_pts)

    if plot:
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], marker='x', c='r', lw=0.5)
        plt.scatter(dst_pts[:, 0], dst_pts[:, 1], marker='+', c='g', lw=1)
        plt.show()

    return np.sqrt(np.square(proj_pts - dst_pts).sum(axis=1)).mean()


def visualize_matches(pair: ImagePair):

    axs: tuple[Axes]
    fig, axs = plt.subplots(1, 2, dpi=100)
    a1, a2 = axs

    for a in axs:
        a.set_axis_off()

    a1.imshow(pair.img1, 'gray')
    a1.scatter(*pair.src_pts.T, c='lime', marker='.', s=1, lw=1)

    a2.imshow(pair.img2, 'gray')
    a2.scatter(*pair.dst_pts.T, c='lime', marker='.', s=1, lw=1)

    fig.tight_layout()
    plt.show()


def export_image_pairs(path: str, ps: list[ImagePair]):
    with h5py.File(path, 'w') as f:
        for idx, p in enumerate(ps):
            g = f.create_group(f'pair_{idx}')
            g.create_dataset('H', data=p.H)
            g.create_dataset('src_pts', data=p.src_pts)
            g.create_dataset('dst_pts', data=p.dst_pts)
            g.attrs['i'] = p.i
            g.attrs['j'] = p.j

        indices = {}
        for p in ps:
            if p.i not in indices:
                indices[p.i] = 1
            if p.j not in indices:
                indices[p.j] = 1

        f.attrs['n_pairs'] = len(ps)
        f.create_dataset('cam_indices', data=sorted(indices.keys()))

def import_cam_params(path: str) -> dict[np.ndarray]:
    cam_params = {}
    with h5py.File(path, 'r') as f:
        for d in f.values():
            cam_idx = d.attrs['cam_idx']
            cam_params[cam_idx] = np.array(d)
    return cam_params

def load_est_gt_poses(vid_path, vid_w):

    vid_path = Path(vid_path)
    data_dir = vid_path.parent
    vid_stem = vid_path.stem

    cam_params_raw = import_cam_params(
        os.path.join(data_dir, vid_stem) + '-opt-poses.h5')

    cam_params_est = {}
    for i, p in sorted(cam_params_raw.items()):
        cam_params_est[i] = {
            'R': Rot.from_rotvec(p[:3], degrees=False).as_matrix(),
            'hfov': 2 * np.degrees(np.arctan(vid_w * 0.5 / p[3])),
        }

    gt_path = os.path.join(data_dir, f'{vid_stem[:4]}.npy')

    if not os.path.isfile(gt_path):
        return cam_params_est, None

    gt = np.load(gt_path)
    cam_params_gt = {}
    for i in sorted(cam_params_est.keys()):
        p = gt[i]
        cam_params_gt[i] = {
            'R': Rot.from_euler('YXZ', p[:3], degrees=True).as_matrix(),
            'hfov': p[3],
        }

    return cam_params_est, cam_params_gt

def set_ref_cam(ref_cam_idx, cam_params_est, cam_params_gt):

    cpe = copy.deepcopy(cam_params_est)
    cpg = copy.deepcopy(cam_params_gt)

    R = cpe[ref_cam_idx]['R'].T.copy()
    for p in cpe.values():
        p['R'] = R @ p['R']

    R = cpg[ref_cam_idx]['R'].T.copy()
    for p in cpg.values():
        p['R'] = R @ p['R']

    return cpe, cpg

def hfov_to_K(hfov_degrees, w, h):
    f = 0.5 * w / np.tan(np.radians(hfov_degrees / 2))
    K = np.array([[-f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    return K

def H_between_frames(p0, p1, w, h):
    """ Compute transformation H from image plane of p0 to p1"""
    R0, R1 = p0['R'], p1['R']
    K0 = hfov_to_K(p0['hfov'], w, h)
    K1 = hfov_to_K(p1['hfov'], w, h)
    H = K1 @ R1.T @ R0 @ np.linalg.inv(K0)
    return H
