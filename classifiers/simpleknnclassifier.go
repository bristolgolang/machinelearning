package classifiers

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// A SimpleKNN implementation
type SimpleKNN struct {
	// K is the number of neighbours we're looking at
	K int
	// The distance function to find out how close a neighbour is
	Distance   func(a, b mat.Vector) float64
	datapoints *mat.Dense
	classes    []string
}

// Fit the data to the classifier
func (k *SimpleKNN) Fit(trainingData *mat.Dense, classes []string) {
	k.datapoints = trainingData
	k.classes = classes
}

// Predict the input based on the classifier
func (k *SimpleKNN) Predict(X *mat.Dense) []string {
	r, _ := X.Dims()
	targets := make([]string, r)
	distances := make([]float64, len(k.classes))
	inds := make([]int, len(k.classes))

	// For every entry we are predicting
	for i := 0; i < r; i++ {
		// Calculate distance to nearest neighbour
		for j := 0; j < len(k.classes); j++ {
			distances[j] = k.Distance(k.datapoints.RowView(j), X.RowView(i))
		}

		// sort the distances
		floats.Argsort(distances, inds)
		votes := make(map[string]float64)
		// for the nearest K neighbours tally up their class count
		for n := 0; n < k.K; n++ {
			votes[k.classes[inds[n]]]++
		}
		// figure out which class has the most neighbours
		var winningCount float64
		for k, v := range votes {
			if v > winningCount {
				targets[i] = k
				winningCount = v
			}
		}
	}
	return targets
}
