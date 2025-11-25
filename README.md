# 3d-perception-from-scratch

A curated, from-scratch implementation of classical and modern 3D perception algorithms used in robotics, autonomous systems, and computer vision.

This repository is my personal “3D Perception Learning Lab,” where I implement every major algorithm by hand, without relying on PCL, Open3D, or deep learning libraries.

## 🚀 Goals
- Implement all essential point cloud algorithms from scratch
- Build a deep understanding of geometric perception
- Replicate algorithms used by PCL, Open3D, and robotics labs

## 📚 Contents
### Core Math & Geometry
- PCA normals
- KD-Tree
- SE(3) transforms
- SVD / eigen

### Features
- PFH
- FPFH
- SHOT
- ISS keypoints

### Registration
- ICP (point-to-point, point-to-plane)
- GICP
- RANSAC alignment
- Fast Global Registration

### Segmentation
- RANSAC plane
- Euclidean clustering
- Region growing

### Deep 3D Models
- PointNet
- PointNet++
- Point Transformer

## Proposed Structure
```
3d-vision-system/
│
├── README.md
├── requirements.txt
├── docs/
│   ├── architecture-diagrams/
│   ├── notes/
│   └── images/
│
├── data/
│   ├── sample_pointclouds/
│   └── test_sets/
│
├── core/
│   ├── geometry/
│   │   ├── transforms.py
│   │   ├── se3.py
│   │   └── utils.py
│   ├── math/
│   │   ├── pca.py
│   │   ├── svd.py
│   │   ├── kdtree.py
│   │   └── nearest_neighbors.py
│   └── visualization/
│       ├── draw_pointcloud.py
│       └── draw_normals.py
│
├── features/
│   ├── pfh.py
│   ├── fpfh.py
│   ├── shot.py
│   └── keypoints/
│       ├── iss.py
│       └── sift3d.py
│
├── registration/
│   ├── icp_point2point.py
│   ├── icp_point2plane.py
│   ├── gicp.py
│   ├── ransac_alignment.py
│   └── fgr.py
│
├── segmentation/
│   ├── ransac_plane.py
│   ├── euclidean_cluster.py
│   └── region_growing.py
│
└── deep/
    ├── pointnet/
    ├── pointnet2/
    └── point_transformer/

```
