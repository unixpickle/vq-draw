package main

import (
	"sort"

	"github.com/unixpickle/model3d"
)

// NonManifoldSolid creates a Solid out of a mesh that has
// (near-)duplicate triangles.
type NonManifoldSolid struct {
	model3d.Collider
}

func (n *NonManifoldSolid) Contains(c model3d.Coord3D) bool {
	if !model3d.InBounds(n, c) {
		return false
	}

	// For more accuracy, more directions could be checked.
	directions := []model3d.Coord3D{
		model3d.Coord3D{X: -0.40475415, Y: 0.86174632, Z: -0.30588783},
		model3d.Coord3D{X: -0.81025101, Y: 0.38452447, Z: -0.44230559},
		model3d.Coord3D{X: -0.09226702, Y: -0.74875317, Z: -0.65639584},
		model3d.Coord3D{X: -0.99668947, Y: 0.08087344, Z: 0.00834144},
		model3d.Coord3D{X: 0.67074042, Y: -0.60098173, Z: 0.43465877},
	}
	for _, d := range directions {
		if n.numIntersections(c, d)%2 == 0 {
			return false
		}
	}

	return true
}

func (n *NonManifoldSolid) numIntersections(coord, direction model3d.Coord3D) int {
	var collisions []model3d.RayCollision
	n.Collider.RayCollisions(&model3d.Ray{
		Origin:    coord,
		Direction: direction,
	}, func(r model3d.RayCollision) {
		collisions = append(collisions, r)
	})
	if len(collisions) == 0 {
		return 0
	}

	sort.Slice(collisions, func(i, j int) bool {
		return collisions[i].Scale < collisions[j].Scale
	})

	// Ignore collisions with nearly duplicate surfaces,
	// treating duplicate triangles as one boundary.
	epsilon := n.Max().Sub(n.Min()).Norm() * 1e-8
	var lastScale float64
	var numUnique int
	for _, c := range collisions {
		if c.Scale-lastScale > epsilon {
			numUnique++
		}
		lastScale = c.Scale
	}
	return numUnique
}
