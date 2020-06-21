package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/bristolgolang/machinelearning/classifiers"
	"github.com/bristolgolang/machinelearning/harness"
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
	randomResult, err := computeClassifierResults(&classifiers.Random{Seed: time.Now().Unix()})
	if err != nil {
		return err
	}
	classifierResults["random"] = randomResult

	harness.PrintResults(classifierResults)
	return nil
}

func computeClassifierResults(predictor harness.Predictor) (harness.Metrics, error) {
	randomResult, err := harness.Evaluate("titanic.csv", predictor)
	if err != nil {
		return harness.Metrics{}, err
	}
	return randomResult, nil
}
