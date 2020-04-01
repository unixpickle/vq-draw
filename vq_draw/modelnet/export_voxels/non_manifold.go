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

	var collisions []model3d.RayCollision
	n.Collider.RayCollisions(&model3d.Ray{
		Origin:    c,
		Direction: model3d.Coord3D{X: -0.40475415, Y: 0.86174632, Z: -0.30588783},
	}, func(r model3d.RayCollision) {
		collisions = append(collisions, r)
	})
	if len(collisions) == 0 {
		return false
	}

	sort.Slice(collisions, func(i, j int) bool {
		return collisions[i].Scale < collisions[j].Scale
	})

	// Ignore collisions with nearly duplicate surfaces,
	// treating duplicate triangles as one boundary.
	epsilon := n.Max().Sub(n.Min()).Norm() * 1e-5
	var lastScale float64
	var numUnique int
	for _, c := range collisions {
		if c.Scale-lastScale > epsilon {
			numUnique++
		}
		lastScale = c.Scale
	}
	return numUnique%2 == 1
}
