// Command grid_to_stl converts a JSON-encoded grid of
// voxel probabilities into a triangle mesh and saves it
// as an STL file.
//
// The JSON input is read from stdin and decoded as a 3D
// array with z on the outer  dimension, then y, then x.
// The array should be NxNxN, i.e. a perfect cube.
package main

import (
	"flag"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
)

func main() {
	var threshold float64
	var outputPath string
	flag.Float64Var(&threshold, "threshold", 0.5, "minimum value for containment")
	flag.StringVar(&outputPath, "output", "output.stl", "output STL file")
	flag.Parse()

	grid, err := ReadVoxelGrid(os.Stdin)
	grid.Threshold = threshold
	essentials.Must(err)

	mesh := model3d.MarchingCubesSearch(grid, 0.5/float64(grid.Size), 8)
	mesh.SaveGroupedSTL(outputPath)
}
