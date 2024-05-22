package main

import (
	"math"
	"math/rand"
)

type network struct {
	B [mnistLabels]float64
	W [mnistLabels][imageSize]float64
}

type networkGradient struct {
	BGrad [mnistLabels]float64
	WGrad [mnistLabels][imageSize]float64
}

func newNetwork() *network {
	n := &network{}
	for i := range n.B {
		n.B[i] = rand.Float64()
		for j := range n.W[i] {
			n.W[i][j] = rand.Float64()
		}
	}
	return n
}

func softmax(activations []float64) {
	currMax := activations[0]
	for _, value := range activations {
		if value > currMax {
			currMax = value
		}
	}

	var sum float64
	for i := range activations {
		activations[i] = math.Exp(activations[i] - currMax)
		sum += activations[i]
	}

	for i := range activations {
		activations[i] /= sum
	}
}

func hypothesis(image *mnistImage, network *network) []float64 {
	activations := make([]float64, mnistLabels)
	for i := range activations {
		activations[i] = network.B[i]
		for j := range image.Pixels {
			activations[i] += network.W[i][j] * float64(image.Pixels[j]) / 255.0
		}
	}
	softmax(activations)
	return activations
}

func gradientUpdate(image *mnistImage, network *network, gradient *networkGradient, label uint8) float64 {
	activations := hypothesis(image, network)
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

func trainingStep(dataset *mnistDataset, network *network, learningRate float64) float64 {
	gradient := &networkGradient{}
	totalLoss := 0.0

	for i := range dataset.Images {
		totalLoss += gradientUpdate(&dataset.Images[i], network, gradient, dataset.Labels[i])
	}

	for i := range network.B {
		network.B[i] -= learningRate * gradient.BGrad[i] / float64(dataset.Size)
		for j := range network.W[i] {
			network.W[i][j] -= learningRate * gradient.WGrad[i][j] / float64(dataset.Size)
		}
	}

	return totalLoss
}
