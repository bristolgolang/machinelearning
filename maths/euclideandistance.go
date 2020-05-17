package maths

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// EuclideanDistance of two vectors
//
// Usable as a distance function for
func EuclideanDistance(a, b mat.Vector) float64 {
	var v mat.VecDense
	v.SubVec(a, b)
	return math.Sqrt(mat.Dot(&v, &v))
}
