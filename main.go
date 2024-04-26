package main

import (
	"fmt"
)

const (
	steps       = 500
	pInterval   = 50
	batchSize   = 100
	trainImages = "data/train-images-idx3-ubyte"
	trainLabels = "data/train-labels-idx1-ubyte"
	testImages  = "data/t10k-images-idx3-ubyte"
	testLabels  = "data/t10k-labels-idx1-ubyte"
)

func main() {
	train, err := getDataset(trainImages, trainLabels)
	if err != nil {
		fmt.Printf("Error loading train dataset: %v\n", err)
		return
	}

	test, err := getDataset(testImages, testLabels)
	if err != nil {
		fmt.Printf("Error loading test dataset: %v\n", err)
		return
	}

	network := newNetwork()
	batches := len(train.Images) / batchSize

	for i := 0; i <= steps; i++ {
		batch := batch(train, batchSize, i%batches)
		loss := trainingStep(batch, network, 0.5)
		accuracy := calculateAccuracy(test, network)

		if (i % pInterval) == 0 {
			fmt.Printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss/float64(len(batch.Images)), accuracy)
		} else {
			fmt.Printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\r", i, loss/float64(len(batch.Images)), accuracy)
		}
	}
}

func calculateAccuracy(dataset *mnistDataset, network *network) float64 {
	var maxActivation float64
	correct := 0

	for i, image := range dataset.Images {
		activations := hypothesis(&image, network)

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
