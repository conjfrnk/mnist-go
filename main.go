package main

import (
	"fmt"
	"math/rand"
	"time"
)

const (
	steps           = 250
	pInterval       = 50
	batchSize       = 100
	trainImagesFile = "data/train-images-idx3-ubyte"
	trainLabelsFile = "data/train-labels-idx1-ubyte"
	testImagesFile  = "data/t10k-images-idx3-ubyte"
	testLabelsFile  = "data/t10k-labels-idx1-ubyte"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	trainDataset, err := mnistGetDataset(trainImagesFile, trainLabelsFile)
	if err != nil {
		fmt.Printf("Error loading train dataset: %v\n", err)
		return
	}

	testDataset, err := mnistGetDataset(testImagesFile, testLabelsFile)
	if err != nil {
		fmt.Printf("Error loading test dataset: %v\n", err)
		return
	}

	network := newNeuralNetwork()
	batches := len(trainDataset.Images) / batchSize

	for i := 0; i <= steps; i++ {
		batch := mnistBatch(trainDataset, batchSize, i%batches)
		loss := neuralNetworkTrainingStep(batch, network, 0.5)
		accuracy := calculateAccuracy(testDataset, network)

		if (i % pInterval) == 0 {
			fmt.Printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss/float64(len(batch.Images)), accuracy)
		} else {
			fmt.Printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\r", i, loss/float64(len(batch.Images)), accuracy)
		}
	}
}

func calculateAccuracy(dataset *mnistDataset, network *neuralNetwork) float64 {
	var maxActivation float64
	correct := 0

	for i, image := range dataset.Images {
		activations := neuralNetworkHypothesis(&image, network)

		predict := 0
		maxActivation = activations[0]
		for j, activation := range activations {
			if activation > maxActivation {
				maxActivation = activation
				predict = j
			}
		}

		if predict == int(dataset.Labels[i]) {
			correct++
		}
	}

	return float64(correct) / float64(len(dataset.Images))
}
