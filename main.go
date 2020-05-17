package main

import (
	"log"

	"github.com/bristolgolang/machinelearning/classifiers"
	"github.com/bristolgolang/machinelearning/harness"
	"github.com/bristolgolang/machinelearning/maths"
)

func main() {
	log.Println("Evaluating models")
	randomClassifier := classifiers.Random{}
	result, err := harness.Evaluate("data_banknote_authentication.csv", &randomClassifier)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Random result = %+v", result)

	knnClassifier := classifiers.KNN{K: 2, Distance: maths.EuclideanDistance}
	result, err = harness.Evaluate("data_banknote_authentication.csv", &knnClassifier)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("KNN result =    %+v", result)
}
