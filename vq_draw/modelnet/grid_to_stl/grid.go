package main

import (
	"encoding/json"
	"io"
	"math"

	"github.com/pkg/errors"
	"github.com/unixpickle/model3d/model3d"
)

type VoxelGrid struct {
	Size int

	// Threshold can be set to change the behavior of the
	// solid containment check.
	Threshold float64

	values []float64
}

// ReadVoxelGrid reads a VoxelGrid as a JSON object.
func ReadVoxelGrid(r io.Reader) (*VoxelGrid, error) {
	var object [][][]float64
	dec := json.NewDecoder(r)
	if err := dec.Decode(&object); err != nil {
		return nil, errors.Wrap(err, "read voxel grid")
	}
	size := len(object)
	result := make([]float64, 0, size*size*size)
	for _, yPlane := range object {
		if len(yPlane) != size {
			return nil, errors.New("read voxel grid: invalid dimensions")
		}
		for _, xLine := range yPlane {
			if len(xLine) != size {
				return nil, errors.New("read voxel grid: invalid dimensions")
			}
			result = append(result, xLine...)
		}
	}
	return &VoxelGrid{
		Size:   size,
		values: result,
	}, nil
}

// Min gets the minimum of the bounding box.
func (v *VoxelGrid) Min() model3d.Coord3D {
	return model3d.Coord3D{}
}

// Max gets the maximum of the bounding box.
func (v *VoxelGrid) Max() model3d.Coord3D {
	return model3d.Coord3D{X: 1, Y: 1, Z: 1}
}

// Contains checks if the value at the point is greater
// than the threshold.
func (v *VoxelGrid) Contains(c model3d.Coord3D) bool {
	return v.Interp(c) >= v.Threshold
}

// Interp gets a trilinear interpolated value for the grid
// at the given point.
func (v *VoxelGrid) Interp(c model3d.Coord3D) float64 {
	// Put the grid inside the unit cube.
	c = c.Scale(float64(v.Size))

	xs, xFracs := roundedCoords(c.X)
	ys, yFracs := roundedCoords(c.Y)
	zs, zFracs := roundedCoords(c.Z)
	var value float64
	for i, x := range xs {
		xFrac := xFracs[i]
		for j, y := range ys {
			yFrac := yFracs[j]
			for k, z := range zs {
				zFrac := zFracs[k]
				value += xFrac * yFrac * zFrac * v.Get(x, y, z)
			}
		}
	}
	return value
}

// Get gets the exact value at integer coordinates.
// If a coordinate is out of bounds, 0 is returned.
func (v *VoxelGrid) Get(x, y, z int) float64 {
	if x < 0 || y < 0 || z < 0 || x >= v.Size || y >= v.Size || z >= v.Size {
		return 0
	}
	return v.values[x+v.Size*(y+z*v.Size)]
}

func roundedCoords(c float64) (vals [2]int, fracs [2]float64) {
	min := int(math.Floor(c))
	max := min + 1
	minFrac := float64(max) - c
	maxFrac := 1 - minFrac
	return [2]int{min, max}, [2]float64{minFrac, maxFrac}
}
