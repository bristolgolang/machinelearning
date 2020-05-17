package harness

import (
	"encoding/csv"
	"os"
	"sort"
	"strconv"
	"time"

	rnd "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/sampleuv"
)

// Predictor implementations define specific machine learning algorithms
// to classify data. The interface contains Fit and Predict methods to
// Fit the model to training data and Predict classes of previously
// unseen observations respectively.
type Predictor interface {
	// Fit the model to the training data. This trains the model learning
	// associations between each feature vector (row vector) within matrix
	// X and it's associated ground truth class label in slice Y.
	Fit(X *mat.Dense, Y []string)
	// Predict will classify the feature vectors (row vectors) within
	// matrix X, predicting the correct class for each based upon what
	// what the model learned during training.
	Predict(X *mat.Dense) []string
}

// Metrics represents a collection of performance metrics to evaluate a
// machine learning classifier.
type Metrics struct {
	Accuracy  float64
	Recall    float64
	Precision float64
	F1        float64
}

// Evaluate takes a path to the dataset CSV file and an algorithm that
// implements the Predictor interface.  The function returns performance
// scores measuring the skill of the algorithm at correctly predicting the
// class of observations or an error if one occurs.
// This function assumes that labels are the last column in the dataset.
func Evaluate(datasetPath string, algo Predictor) (Metrics, error) {
	hasHeader := true
	records, err := loadFile(datasetPath, hasHeader)
	if err != nil {
		return Metrics{}, err
	}

	trainData, trainLabels, testData, testLabels := split(true, records, 0.7)

	algo.Fit(trainData, trainLabels)

	predictions := algo.Predict(testData)

	return evaluate(predictions, testLabels), nil
}

func loadFile(path string, header bool) ([][]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if header {
		records = records[1:]
	}
	return records, nil
}

// split the dataset into training and test sets for training and evaluation
// respectively.
func split(random bool, records [][]string, trainProportion float64) (*mat.Dense, []string, *mat.Dense, []string) {
	datasetLength := len(records)
	indx := make([]int, int(float64(datasetLength)*trainProportion))
	if random {
		// randomly sample k indices for training set and sort
		// TODO allow seed to be replicatable
		r := rnd.New(rnd.NewSource(uint64(time.Now().Unix())))
		sampleuv.WithoutReplacement(indx, datasetLength, r)
		sort.Ints(indx)
	} else {
		// take the first k indices for the training set
		for i := range indx {
			indx[i] = i
		}
	}

	trainData := mat.NewDense(len(indx), len(records[0])-1, nil)
	trainLabels := make([]string, len(indx))
	testData := mat.NewDense(len(records)-len(indx), len(records[0])-1, nil)
	testLabels := make([]string, len(records)-len(indx))

	var trainind, testind int
	for i, v := range records {
		if trainind < len(indx) && i == indx[trainind] {
			// training set
			readRecord(trainLabels, trainData, trainind, v)
			trainind++
		} else {
			// test set
			readRecord(testLabels, testData, testind, v)
			testind++
		}
	}
	return trainData, trainLabels, testData, testLabels
}

func readRecord(labels []string, data mat.Mutable, recordNum int, record []string) {
	labels[recordNum] = record[len(record)-1]
	for i, v := range record[:len(record)-1] {
		s, err := strconv.ParseFloat(v, 64)
		if err != nil {
			// replace invalid numbers with 0
			s = 0
		}
		data.Set(recordNum, i, s)
	}
}

func evaluate(predictions, labels []string) Metrics {
	var tp, fn, fp, tn int

	for i, v := range labels {
		if v == "1" {
			if predictions[i] == "1" {
				tp++
			} else {
				fn++
			}
		} else {
			if predictions[i] == "1" {
				fp++
			} else {
				tn++
			}
		}
	}

	m := Metrics{
		Accuracy:  float64(tn+tp) / float64(len(labels)),
		Recall:    float64(tp) / float64(tp+fn),
		Precision: float64(tp) / float64(tp+fp),
	}

	m.F1 = 2 * ((m.Precision * m.Recall) / (m.Precision + m.Recall))

	return m
}
