import numpy as np
import os

from termcolor import colored

from libs.utils import voxels


class Evaluator:
    def __init__(self, result_path, mesh_th):
        self.mesh_th = mesh_th
        self.vis_result_dir = os.path.join(result_path, 'mesh')
        print(colored('the mesh visual results are saved at {}'.format(self.vis_result_dir), 'yellow'))

        self.pts_result_dir = os.path.join(result_path, 'pts')
        print(colored('the pts infos are saved at {}'.format(self.pts_result_dir), 'yellow'))

    def evaluate(self, output, batch):
        cube = output['cube']
        cube = cube[10:-10, 10:-10, 10:-10]

        pts = batch['pts'][0].detach().cpu().numpy()
        pts = pts[cube > self.mesh_th]

        i = batch['frame_index'].item()
        
        result_dir = self.pts_result_dir
        os.system('mkdir -p {}'.format(result_dir))
        result_path = os.path.join(result_dir, '{}.npy'.format(i))
        np.save(result_path, pts)

    # TODO evaluate mesh
    def summarize(self):
        return {}

    def visualize_voxel(self, output, batch):
        cube = output['cube']
        cube = cube[10:-10, 10:-10, 10:-10]
        cube[cube < self.mesh_th] = 0
        cube[cube > self.mesh_th] = 1

        sh = cube.shape
        square_cube = np.zeros((max(sh), ) * 3)
        square_cube[:sh[0], :sh[1], :sh[2]] = cube
        voxel_grid = voxels.VoxelGrid(square_cube)
        mesh = voxel_grid.to_mesh()
        mesh.show()

    def visualize(self, output, batch):
        mesh = output['mesh']
        # mesh.show()

        result_dir = self.vis_result_dir
        os.system('mkdir -p {}'.format(result_dir))
        i = batch['frame_index'][0].item()
        if 'cam_ind' in batch:
            result_path = os.path.join(result_dir, '{}_cam{}.ply'.format(i, batch['cam_ind'].item()))
        else:
            result_path = os.path.join(result_dir, '{}.ply'.format(i))
        mesh.export(result_path)