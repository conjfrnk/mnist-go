package main

import (
	"math"
	"math/rand"
)

type neuralNetwork struct {
	B [mnistLabels]float64
	W [mnistLabels][mnistImageSize]float64
}

type neuralNetworkGradient struct {
	BGrad [mnistLabels]float64
	WGrad [mnistLabels][mnistImageSize]float64
}

func newNeuralNetwork() *neuralNetwork {
	n := &neuralNetwork{}
	for i := range n.B {
		n.B[i] = rand.Float64()
		for j := range n.W[i] {
			n.W[i][j] = rand.Float64()
		}
	}
	return n
}

func neuralNetworkSoftmax(activations []float64) {
	max := activations[0]
	for _, value := range activations {
		if value > max {
			max = value
		}
	}

	var sum float64
	for i := range activations {
		activations[i] = math.Exp(activations[i] - max)
		sum += activations[i]
	}

	for i := range activations {
		activations[i] /= sum
	}
}

func neuralNetworkHypothesis(image *mnistImage, network *neuralNetwork) []float64 {
	activations := make([]float64, mnistLabels)
	for i := range activations {
		activations[i] = network.B[i]
		for j := range image.Pixels {
			activations[i] += network.W[i][j] * float64(image.Pixels[j]) / 255.0
		}
	}
	neuralNetworkSoftmax(activations)
	return activations
}

func neuralNetworkGradientUpdate(image *mnistImage, network *neuralNetwork, gradient *neuralNetworkGradient, label uint8) float64 {
	activations := neuralNetworkHypothesis(image, network)
	loss := -math.Log(activations[label])

	for i := range activations {
		bGrad := activations[i]
		if i == int(label) {
			bGrad -= 1
		}
		gradient.BGrad[i] += bGrad
		for j := range image.Pixels {
			WGrad := bGrad * float64(image.Pixels[j]) / 255.0
			gradient.WGrad[i][j] += WGrad
		}
	}

	return loss
}

func neuralNetworkTrainingStep(dataset *mnistDataset, network *neuralNetwork, learningRate float64) float64 {
	gradient := &neuralNetworkGradient{}
	totalLoss := 0.0

	for i := range dataset.Images {
		totalLoss += neuralNetworkGradientUpdate(&dataset.Images[i], network, gradient, dataset.Labels[i])
	}

	for i := range network.B {
		network.B[i] -= learningRate * gradient.BGrad[i] / float64(dataset.Size)
		for j := range network.W[i] {
			network.W[i][j] -= learningRate * gradient.WGrad[i][j] / float64(dataset.Size)
		}
	}

	return totalLoss
}
