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

    def __init__(self, grid_size, image_size=100):
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
        origin = np.array([3, 3, 3], dtype='float64')

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
                yield Ray(origin, rel_x + rel_y + z_direction)

    def _bounding_volume_collisions(self, vox_origin, vox_size, ray):
        min_coord = (np.array(vox_origin, dtype='float64') / self.grid_size) - ray.origin
        max_coord = min_coord + vox_size / self.grid_size
        t1 = min_coord / ray.direction
        t2 = max_coord / ray.direction
        t_mins, t_maxes = np.min([t1, t2], axis=0), np.max([t1, t2], axis=0)
        t_min = np.max(t_mins)
        t_max = np.min(t_maxes)

        # Most of the time, we will not collide with the box.
        # Also theoretically possible to be inside the box.
        if t_min >= t_max or t_min < 0:
            return []

        if vox_size > 1:
            half_size = vox_size // 2
            collisions = []
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        sub_origin = tuple(o + half_size*d for o, d in zip(vox_origin, (x, y, z)))
                        sub_coll = self._bounding_volume_collisions(sub_origin, half_size, ray)
                        collisions.extend(sub_coll)
            return collisions

        point = ray.direction * t_min
        approx_normal = point - (min_coord + max_coord) / 2
        max_component = np.argmax(np.abs(approx_normal))
        return [RayCollision(t_min, vox_origin, abs(ray.direction[max_component]))]


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.copy()

        # Avoid divisions by zero.
        for i, x in enumerate(direction):
            if abs(x) < 1e-8:
                direction[i] = 1e-8


class RayCollision:
    def __init__(self, ray_time, voxel_coord, incident_cos):
        self.ray_time = ray_time
        self.voxel_coord = voxel_coord
        self.incident_cos = incident_cos
