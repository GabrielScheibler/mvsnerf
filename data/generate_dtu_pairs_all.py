import sys
import os
import numpy as np

def build_proj_mats():
    id_list = range(49)
    elements = []
    for vid in id_list:
        proj_mat_filename = os.path.join(root_dir,
                                            f'Cameras/train/{vid:08d}_cam.txt')
        intrinsic, extrinsic = read_cam_file(
            proj_mat_filename)
        
        R = extrinsic[:3,:3]
        T = extrinsic[:3,-1]
        C = (-np.transpose(R)) @ T

        element = (vid, C)
        elements += [element]

    indices = np.ones([49,48])
    dists = np.ones([49,48])
    for i in range(49):
        x, d = sort_by_distance(i,elements)
        indices[i,:] = x
        dists[i,:] = d

    return indices, dists

def write_test_file(indices, dists):

    string = ""
    string += str(indices.shape[0]) + '\n'

    for r in range(indices.shape[0]):
        string += str(r) + '\n'
        line = str(indices[r].shape[0])
        for c in range(indices[r].shape[0]):
            line += ' ' + str(int(indices[r,c])) + " " + '0.0'
        line += '\n'
        string += line

    with open('dtu_pairs_all.txt', 'w') as f:
        f.write(string)

def sort_by_distance(idx,array):
     _, C = array[idx]
     l = array
     l = [(x[0],distance(x[1],C)) for x in l]
     l.sort(key=lambda elem: elem[1])
     d = [x[1] for x in l]
     l = [x[0] for x in l]
     #l = [x[i] for x in l for i in (0,1)]
     del l[0]
     del d[0]
     return l , d

def distance(v1,v2):
     return np.sum((v1-v2)**2)

        
def read_cam_file(filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(
            ' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(
            ' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        return intrinsics, extrinsics


root_dir = sys.argv[1]
i,d = build_proj_mats()
write_test_file(i,d)