// Command export_voxels exports ModelNet models as binary
// voxel grids.
package main

import (
	"archive/zip"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d"
)

func main() {
	var numVariations int
	var gridSize int

	flag.IntVar(&numVariations, "variations", 1, "number of random perturbations to produce")
	flag.IntVar(&gridSize, "grid-size", 64, "number of voxels along each dimension")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "[flags] <input_dir> <output_dir>")
		flag.PrintDefaults()
		os.Exit(1)
	}
	flag.Parse()
	if len(flag.Args()) != 2 {
		flag.Usage()
	}

	inDir := flag.Args()[0]
	outDir := flag.Args()[1]

	err := filepath.Walk(inDir, func(inPath string, info os.FileInfo, err error) error {
		relPath, err := filepath.Rel(inDir, inPath)
		essentials.Must(err)
		outPath := filepath.Join(outDir, relPath)

		if info.IsDir() {
			if _, err := os.Stat(outPath); os.IsNotExist(err) {
				essentials.Must(os.Mkdir(outPath, 0755))
			}
			return nil
		}

		if filepath.Ext(inPath) == ".off" {
			return ConvertModel(inPath, outPath, numVariations, gridSize)
		}
		return nil
	})
	essentials.Must(err)
}

func ConvertModel(inPath, outPath string, numVariations, gridSize int) error {
	log.Println("Converting", inPath, "...")

	r, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer r.Close()
	triangles, err := model3d.ReadOFF(r)
	if err != nil {
		return err
	}
	mesh := model3d.NewMeshTriangles(triangles)

	outBase := outPath[:len(outPath)-len(filepath.Ext(outPath))]
	for i := 0; i < numVariations; i++ {
		outPath := fmt.Sprintf("%s-%d.npz", outBase, i)
		saveMesh := mesh
		if i != 0 {
			saveMesh = TransformMesh(saveMesh)
		}
		voxels := NewVoxelConnector(saveMesh, gridSize).Voxels()
		if err := SaveNumpy(outPath, gridSize, voxels); err != nil {
			return err
		}
	}

	return nil
}

func TransformMesh(mesh *model3d.Mesh) *model3d.Mesh {
	v1 := model3d.NewCoord3DRandUnit()
	v2 := model3d.NewCoord3DRandUnit().ProjectOut(v1).Normalize()
	v3 := model3d.NewCoord3DRandUnit().ProjectOut(v1).ProjectOut(v2).Normalize()
	transform := &model3d.Matrix3Transform{
		Matrix: model3d.NewMatrix3Columns(v1, v2, v3),
	}

	// Only use rotations, not mirrors.
	if transform.Matrix.Det() < 0 {
		for i := 0; i < 3; i++ {
			transform.Matrix[i] *= -1
		}
	}

	return mesh.MapCoords(transform.Apply)
}

func SaveNumpy(path string, gridSize int, data []byte) error {
	w, err := os.Create(path)
	if err != nil {
		return err
	}
	defer w.Close()
	zipWriter := zip.NewWriter(w)
	fileWriter, err := zipWriter.Create("voxels.npy")
	if err != nil {
		return err
	}
	if _, err := fileWriter.Write(EncodeNumpy(gridSize, data)); err != nil {
		return err
	}
	if err := zipWriter.Close(); err != nil {
		return err
	}
	return nil
}

func EncodeNumpy(gridSize int, data []byte) []byte {
	header := "\x93NUMPY\x01\x00\x76\x00{'descr': '|b1', 'fortran_order': False, 'shape': ("
	header += fmt.Sprintf("%d, %d, %d)}", gridSize, gridSize, gridSize)
	for len(header) < 0x80-1 {
		header += " "
	}
	header += "\n"
	return append([]byte(header), data...)
}
