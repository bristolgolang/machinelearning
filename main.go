package main

import (
	"fmt"
	"log"
	"os"

	"github.com/bristolgolang/machinelearning/classifiers"
	"github.com/bristolgolang/machinelearning/harness"
	"github.com/bristolgolang/machinelearning/maths"
)

func main() {
	if err := compareClassifiers(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func compareClassifiers() error {
	log.Println("Evaluating models")

	classifierResults := make(map[string]harness.Metrics)
	// Add your classifiers here and to classifierResults
	randomResult, err := computeClassifierResults(&classifiers.Random{})
	if err != nil {
		return err
	}
	classifierResults["random"] = randomResult

	simpleKNNResult, err := computeClassifierResults(&classifiers.SimpleKNN{K: 2, Distance: maths.EuclideanDistance})
	if err != nil {
		return err
	}
	classifierResults["simpleKNN"] = simpleKNNResult

	harness.PrintResults(classifierResults)
	return nil
}

func computeClassifierResults(predictor harness.Predictor) (harness.Metrics, error) {
	randomResult, err := harness.Evaluate("data_banknote_authentication.csv", predictor)
	if err != nil {
		return harness.Metrics{}, err
	}
	return randomResult, nil
}
