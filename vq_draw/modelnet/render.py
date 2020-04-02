import math

from PIL import Image
import numpy as np


class VoxelRenderer:
    """
    Render fixed-sized voxel grids to images very
    efficiently.

    This renderer caches ray collisions so that voxel
    grids can be rendered without casting rays.
    """

    def __init__(self, grid_size, image_size=200):
        if grid_size != 2 ** int(round(math.log2(grid_size))):
            raise ValueError('grid_size must be a power of 2')
        self.grid_size = grid_size
        self.image_size = image_size
        self._pixel_voxels = []
        self._pixel_radiances = []
        for ray in self._camera_rays():
            collisions = self._bounding_volume_collisions((0, 0, 0), self.grid_size, ray)
            collisions = sorted(collisions, key=lambda x: x.ray_time)
            self._pixel_voxels.append(
                tuple(zip(*[c.voxel_coord for c in collisions])),
            )
            self._pixel_radiances.append(
                np.array([c.incident_cos for c in collisions], dtype='float32'),
            )

    def render(self, voxel_data):
        arr = np.zeros((self.image_size**2,), dtype='float32')
        for i, (voxels, radiances) in enumerate(zip(self._pixel_voxels, self._pixel_radiances)):
            if not len(voxels):
                continue
            values = voxel_data[voxels]
            if np.any(values):
                arr[i] = radiances[np.argmax(values)]
        return arr.reshape([self.image_size, self.image_size])

    def render_grid_to_file(self, out_path, voxel_grid):
        renders = np.array([self.render(x) for y in voxel_grid for x in y])
        renders = renders.reshape([*voxel_grid.shape[:2], *renders[0].shape])
        full_img = np.concatenate(np.concatenate(renders, axis=-2), axis=-1)
        full_img = np.repeat(full_img[..., None], 3, axis=-1)
        int_img = (full_img * 255).astype('uint8')
        Image.fromarray(int_img).save(out_path)

    def _camera_rays(self):
        origin = np.array([2, 2, 2], dtype='float64')

        z_direction = -origin
        x_direction = np.array([-origin[1], origin[0], 0])
        y_direction = np.cross(z_direction, x_direction)

        # Smaller field of view by adding a factor of 2.
        x_direction /= np.linalg.norm(x_direction) * 2
        y_direction /= np.linalg.norm(y_direction) * 2
        z_direction /= np.linalg.norm(z_direction)

        for y in range(self.image_size):
            rel_y = y_direction * (y / (self.image_size / 2) - 1)
            for x in range(self.image_size):
                rel_x = x_direction * (x / (self.image_size / 2) - 1)
                ray_dir = rel_x + rel_y + z_direction

                # Normalize to get proper lighting.
                ray_dir /= np.linalg.norm(ray_dir)

                yield Ray(tuple(origin), tuple(ray_dir))

    def _bounding_volume_collisions(self, vox_origin, vox_size, ray):
        scale = 1 / self.grid_size
        box_size = vox_size * scale
        min_coord = (vox_origin[0] * scale - ray.origin[0],
                     vox_origin[1] * scale - ray.origin[1],
                     vox_origin[2] * scale - ray.origin[2])
        max_coord = (min_coord[0] + box_size,
                     min_coord[1] + box_size,
                     min_coord[2] + box_size)

        t_min, t_max = -100000.0, 100000.0
        for c1, c2, ray_dir in zip(min_coord, max_coord, ray.direction):
            t1 = c1 / ray_dir
            t2 = c2 / ray_dir
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min >= t_max or t_min < 0:
                return []

        if vox_size > 1:
            half_size = vox_size // 2
            collisions = []
            for x in range(2):
                new_x = vox_origin[0] + half_size * x
                for y in range(2):
                    new_y = vox_origin[1] + half_size * y
                    for z in range(2):
                        new_z = vox_origin[2] + half_size * z
                        sub_origin = (new_x, new_y, new_z)
                        sub_coll = self._bounding_volume_collisions(sub_origin, half_size, ray)
                        collisions.extend(sub_coll)
            return collisions

        # Figure out the cosine for the side of the box
        # the ray collided with.
        # Assume the camera is the light source.
        max_incidence = 0
        max_diff = 0
        for ray_dir, min_val, max_val in zip(ray.direction, min_coord, max_coord):
            p = ray_dir * t_min
            c = (min_val + max_val) / 2
            diff = abs(p - c)
            if diff > max_diff:
                max_diff = diff
                max_incidence = abs(ray_dir)

        return [RayCollision(t_min, vox_origin, max_incidence)]


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        # Prevent epsilon directions.
        self.direction = tuple(x if abs(x) > 1e-8 else 1e-8 for x in direction)


class RayCollision:
    def __init__(self, ray_time, voxel_coord, incident_cos):
        self.ray_time = ray_time
        self.voxel_coord = voxel_coord
        self.incident_cos = incident_cos
