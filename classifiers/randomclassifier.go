package classifiers

import (
	"math/rand"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Random is a baseline binary classifier that predicts results at random.
type Random struct{}

// Fit the data to the classifier
//
// Nothing to fit, as we will be predicting at random
func (b *Random) Fit(trainingData *mat.Dense, labels []string) {}

// Predict on the test data based upon the training data
//
// Uses rand to randomly predict "0" or "1"
func (b *Random) Predict(testData *mat.Dense) []string {
	// move seed to once initialisation
	rand.Seed(time.Now().Unix())
	rowCount, _ := testData.Dims()
	predictions := make([]string, rowCount)
	for i := 0; i < rowCount; i++ {
		predictions[i] = strconv.Itoa(rand.Intn(2))
	}

	return predictions
}
