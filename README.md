# Machine Learning excercise

This is the excercise to go alongside the machine learning talks our [meetup group](https://www.meetup.com/golang-bristol/) held in April. It is a _mildly_ ammended version of [fresh8's mlworkshop](https://github.com/fresh8/mlworkshop), thank you!

We will be exploring concepts used when creating classifiers in Go, building our own from scratch using the [slides from James' talk](https://snip.ly/3d66pb)

## Goal

To explore, understand and implement our own classifier algorithm to detect authentic bank notes, using this [data set from UCI](http://archive.ics.uci.edu/ml/datasets/banknote+authentication).

## How to partake

This exercise is available for everyone, we will be covering it in the Bristol meetup.

Either clone the repo and implement the `main.go` file in the root, you can then make a PR back to into this repo, sharing what you learnt with others, or create your own package and import the harness package.

If you fancy sharing your work, make a PR into this repo, they will not be merged but it will allow others to see what other solutions you have all come up with.

## Understanding the harness package

### The Predictor interface

The biggest concept in the harness package is the `Predictor` interface. If you're unfamiliar with go interfaces, [go by example](https://gobyexample.com/interfaces) has simple examples. If you prefer podcasts, [go time](https://changelog.com/gotime) has an excellent episode on [interfaces](https://changelog.com/gotime/118)

```go
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
```

To implement the interface we would create a `struct`, and implement the `Fit` and `Predict` methods. We don't need to explicitly state we're implementing the interface (we don't even need to import it).

### Measuring success

It's important to get a good measure of how well your classifier works. There are an enourmous number of ways to measure how we your implementation works. Different metrics work for different types of models, as we're using classification, accuracy, recall, precision, and f1-score are what we're mostly interested in.

Further reading of a different metrics for [classification and regression models](https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce).

```go
// Metrics represents a collection of performance metrics to evaluate a
// machine learning classifier.
type Metrics struct {
	Accuracy  float64
	Recall    float64
	Precision float64
	F1        float64
}
```

### The Evaluation function

The `Evaluate` function takes in our `Predictor` interface from earlier, and uses the `Fit` and `Predict` methods we discussed easlier.

```go
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
```
