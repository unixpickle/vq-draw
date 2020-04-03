package main

import (
	"math"

	"github.com/unixpickle/model3d"
)

type VoxelCoord [3]int

// A VoxelConnector creates voxel grids based on direct
// connectivity between points in space relative to some
// mesh surface.
type VoxelConnector struct {
	Space    *VoxelSpace
	Collider model3d.Collider
}

// NewVoxelConnector creates a new VoxelConnector for a
// given 3D model and voxel grid size.
func NewVoxelConnector(m *model3d.Mesh, gridSize int) *VoxelConnector {
	collider := model3d.MeshToCollider(m)

	sizes := collider.Max().Sub(collider.Min())
	size := math.Max(math.Max(sizes.X, sizes.Y), sizes.Z)

	unit := model3d.Coord3D{X: 1, Y: 1, Z: 1}
	origin := sizes.Sub(unit.Scale(size)).Scale(0.5).Add(collider.Min())

	return &VoxelConnector{
		Space: &VoxelSpace{
			Origin:   origin,
			Size:     size,
			GridSize: gridSize,
		},
		Collider: collider,
	}
}

// Voxels creates a voxel grid of the model using a simple
// search algorithm.
//
// A voxel is 1 if it is unreachable from outside the mesh
// or if it is close to a boundary of the mesh.
// Otherwise, it is 0.
func (v *VoxelConnector) Voxels() []byte {
	reachable := NewBorderVoxels(v.Space.GridSize)
	edges := NewBorderVoxels(v.Space.GridSize)

	queue := []VoxelCoord{{-1, -1, -1}}
	*reachable.At(queue[0]) = true

	for len(queue) > 0 {
		coord := queue[0]
		queue = queue[1:]
		reachable.Neighbors(coord, func(neighbor VoxelCoord) {
			connected, onEdge := v.Connect(coord, neighbor)
			if connected {
				r := reachable.At(neighbor)
				if !*r {
					*r = true
					queue = append(queue, neighbor)
				}
			} else if onEdge {
				*edges.At(coord) = true
			}
		})
	}

	result := edges.Unbordered()
	for i, r := range reachable.Unbordered() {
		result[i] |= r ^ 1
	}

	return result
}

// Connect attempts to make a connection from v1 to v2.
//
// If a connection can be made, the first return value is
// true. Otherwise, the second return value indicates
// whether or not v1 is closer to the surface standing in
// the way of v1 and v2.
func (v *VoxelConnector) Connect(v1, v2 VoxelCoord) (connected, sourceBorder bool) {
	c1 := v.Space.Coord(v1)
	c2 := v.Space.Coord(v2)

	// If the sphere containing the line segment does
	// not contain anything, no surface can be in the
	// way.
	//
	// This is faster than a ray collision, since it
	// only has to check a local neighborhood.
	if !v.Collider.SphereCollision(c1.Mid(c2), c1.Dist(c2)/(2-1e-8)) {
		return true, false
	}

	ray := &model3d.Ray{
		Origin:    c1,
		Direction: c2.Sub(c1),
	}
	coll, ok := v.Collider.FirstRayCollision(ray)
	if !ok || coll.Scale > 1 {
		return true, false
	}
	return false, coll.Scale < 0.5
}

type BorderVoxels struct {
	GridSize int
	Data     []bool
}

func NewBorderVoxels(gridSize int) *BorderVoxels {
	g := gridSize + 2
	return &BorderVoxels{
		GridSize: gridSize,
		Data:     make([]bool, g*g*g),
	}
}

func (b *BorderVoxels) At(coord VoxelCoord) *bool {
	size := b.GridSize + 2
	return &b.Data[(coord[2]+1)+((coord[1]+1)+(coord[0]+1)*size)*size]
}

func (b *BorderVoxels) Neighbors(coord VoxelCoord, f func(VoxelCoord)) {
	for x := -1; x <= 1; x++ {
		for y := -1; y <= 1; y++ {
			for z := -1; z <= 1; z++ {
				if x == 0 && y == 0 && z == 0 {
					continue
				}
				newCoord := VoxelCoord{coord[0] + x, coord[1] + y, coord[2] + z}
				if b.InBounds(newCoord) {
					f(newCoord)
				}
			}
		}
	}
}

func (b *BorderVoxels) InBounds(c VoxelCoord) bool {
	for _, x := range c {
		if x < -1 || x > b.GridSize {
			return false
		}
	}
	return true
}

func (b *BorderVoxels) Unbordered() []byte {
	res := make([]byte, 0, b.GridSize*b.GridSize*b.GridSize)
	for x := 0; x < b.GridSize; x++ {
		for y := 0; y < b.GridSize; y++ {
			for z := 0; z < b.GridSize; z++ {
				if *b.At(VoxelCoord{x, y, z}) {
					res = append(res, 1)
				} else {
					res = append(res, 0)
				}
			}
		}
	}
	return res
}

type VoxelSpace struct {
	Origin   model3d.Coord3D
	Size     float64
	GridSize int
}

func (v *VoxelSpace) Coord(vc VoxelCoord) model3d.Coord3D {
	unit := model3d.Coord3D{X: 1, Y: 1, Z: 1}
	cellSize := v.Size / float64(v.GridSize)
	idxCoord := model3d.Coord3D{X: float64(vc[0]), Y: float64(vc[1]), Z: float64(vc[2])}
	return v.Origin.Add(idxCoord.Add(unit.Scale(0.5)).Scale(cellSize))
}
