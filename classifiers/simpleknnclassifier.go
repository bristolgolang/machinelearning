package classifiers

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// A SimpleKNN implementation
type SimpleKNN struct {
	K          int
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

		// HERE BE DRAGONS
		floats.Argsort(distances, inds)
		votes := make(map[string]float64)
		for n := 0; n < k.K; n++ {
			votes[k.classes[inds[n]]]++
		}
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
