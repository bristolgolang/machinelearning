# Machine Learning excercise

This is the excercise to go alongside the machine learning talks our [meetup group](https://www.meetup.com/golang-bristol/) held in April. It is a _mildly_ ammended version of [fresh8's mlworkshop](https://github.com/fresh8/mlworkshop), thank you for sharing!

We will be exploring concepts used when creating classifiers in Go, building our own from scratch using the [slides from James' talk](https://snip.ly/3d66pb)

## Goal

To explore, understand and implement our own KNN classifier algorithm that performs better than a `random` implementation.
We'll be looking at a titanic dataset that has already been feature engineered, such that all categorical data has been encoded into numbers for us.

To run this exercise as it stands, run the following

```
$ go run main.go

random              | {Accuracy:0.503731343283582 Recall:0.4723618090452261 Precision:0.7704918032786885 F1:0.5856697819314641}
```

### Create a KNN classifier from scratch

There's a PR for a simple KNN classifier, so if you wish to avoid spoilers (or want to look for guidance) check out the PRs to this repo.

### Improve a simple KNN classifier

Either branch off a classifier in a PR, or take the classifier you built, and try some of these [techniques for improving your classifier](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/#improvements)

### Bonus Goals

- To implement some feature engineering on a more [challenging titanic dataset](https://www.kaggle.com/c/titanic/data).

## How to partake

This exercise is available for everyone, however we will be covering explicitly it in the June 2020 Bristol (UK) Go meetup.
It is recommended to read through the README while exploring the codebase.

Clone the repo and implement your own KNN classifier in the classifiers package, you can then make a PR back to into this repo, sharing what you learnt with others (recommended)

## Understanding the codebase

### Understanding the harness package

#### The Predictor interface

The biggest concept in the harness package is the `Predictor` interface.
If you're unfamiliar with go interfaces, [go by example](https://gobyexample.com/interfaces) has simple examples.
If you prefer podcasts, [go time](https://changelog.com/gotime) has an excellent episode on [interfaces](https://changelog.com/gotime/118).
One of the beautiful things about interfaces in Go is in our example, the harness package and classifier package do not know about eachother, nor need to, simply the calling code needs to ensure the implementation matches the interface.

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

To implement the interface we would create a `struct`, and implement the `Fit` and `Predict` methods.
As stated earlier, we don't need to explicitly state we're implementing the interface (as we don't even need to import it).

#### Measuring success

It's important to get a good measure of how well your classifier works, and there are an enourmous number of ways to measure this.
Different metrics work for different types of models, as we're using classification, accuracy, recall, precision, and f1-score are what we're mostly interested in.

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

#### The Evaluation function

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

	seed := uint64(time.Now().Unix())
	trainData, trainLabels, testData, testLabels := split(records, 0.7, seed)

	algo.Fit(trainData, trainLabels)

	predictions := algo.Predict(testData)

	return evaluate(predictions, testLabels), nil
}
```

We'll omit a few concepts here for the sake of brevity, however, if you want to read further, the godocs are a great place to learn;
- [file io](https://golang.org/pkg/os/)
- [reading csv](https://golang.org/pkg/encoding/csv/)
- [random](https://pkg.go.dev/golang.org/x/exp/rand?tab=doc)

One thing to note is we are passing a seed into splitter function, this would enable us to test this function dependably.

### Understanding the random classifier

The important thing here is we implement both `Fit` and `Predict`, this allows us to use the random classifier with the harness discussed earlier.

#### The struct

To implement the interface, we need a struct.
All we intend to do is randomly choose either 1 or 0 for our predictions, so we don't need to store any data, do any precomputations or anything.
What we do do, is provide a seed, similar to above, this just allows some determinism should we choose to test the classifier works correctly.

```go
// Random is a baseline binary classifier that predicts results at random.
type Random struct {
	Seed int64
}
```

#### Fitting our random classifier

As mentioned earlier, we don't need to do anything here, but we still need to implement the function to make use of `Evaluate` discussed earlier.

```go
// Fit the data to the classifier
//
// Nothing to fit, as we will be predicting at random
func (b *Random) Fit(_ *mat.Dense, _ []string) {}
```

#### Predicting with our random classifier

Here we wish to arbitrarily choose whether to predict 1 or 0.

```go
// Predict on the test data based upon the training data
//
// Uses rand to randomly predict "0" or "1"
func (b *Random) Predict(testData *mat.Dense) []string {
	// move seed to once initialisation
	rand.Seed(b.Seed)
	rowCount, _ := testData.Dims()
	predictions := make([]string, rowCount)
	for i := 0; i < rowCount; i++ {
		predictions[i] = strconv.Itoa(rand.Intn(2))
	}

	return predictions
}
```

### Implementing the KNN classifier

#### The struct

Building off of our random classifier, we not only store the datapoints, but also the classes we're trying to predict.
On top of this, we store `K`, which is the number of neighbours to look at for comparison.
We also use a `Distance` function, this allows us to compute how close two vectors are.

```go
// A SimpleKNN implementation
type SimpleKNN struct {
	// K is the number of neighbours we're looking at
	K int
	// The distance function to find out how close a neighbour is
	Distance   func(a, b mat.Vector) float64
	datapoints *mat.Dense
	classes    []string
}
```

#### Fitting our KNN classifier

We won't do anything too complicated to fit our model, we'll simply assign the datapoints and classes to our struct.

```go
// Fit the data to the classifier
func (k *SimpleKNN) Fit(trainingData *mat.Dense, classes []string) {
	k.datapoints = trainingData
	k.classes = classes
}
```

#### Predicting with our KNN classifier

It's best to break this up into a couple of chunks as a lot is going on here.
To get set up we make slices for everything we're going to need, this includes the targets (our predictions), and the distances from our vector to the other vectors.

After thiswe want to create a loop to iterate through all the entries we need to predict for.

```go
// Predict the input based on the classifier
func (k *SimpleKNN) Predict(X *mat.Dense) []string {
	r, _ := X.Dims()
	targets := make([]string, r)
	distances := make([]float64, len(k.classes))
	inds := make([]int, len(k.classes))

	// For every entry we are predicting
	for i := 0; i < r; i++ {
		// ...
	}
	return targets
}
```

Within the for loop, we need to determine which neighbours are the closest, we do this by iterating through each datapoint and use our `Distance` function from earlier. We then use the floats package to sort the distances from smallest to largest.

```go
	// for loop ...

		// Calculate distance to nearest neighbour
		for j := 0; j < len(k.classes); j++ {
			distances[j] = k.Distance(k.datapoints.RowView(j), X.RowView(i))
		}

		// sort the distances
		floats.Argsort(distances, inds)

		// ...
	// end for loop
```

After we learn the nearest neighbours, we then need to iterate through to our value of `K` (the number of neighbours to compare against). For each neighbour, we add one to its class, we then iterate through the votes to determine which class our prediction should be.

```go
	// for loop
		// ...

		// for the nearest K neighbours tally up their class count
		votes := make(map[string]float64)
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
	// end for loop
```
