package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

type labelHeader struct {
	MagicNumber    uint32
	NumberOfLabels uint32
}

type imageHeader struct {
	MagicNumber     uint32
	NumberOfImages  uint32
	NumberOfRows    uint32
	NumberOfColumns uint32
}

type mnistImage struct {
	Pixels [imageSize]uint8
}

type mnistDataset struct {
	Images []mnistImage
	Labels []uint8
	Size   int
}

func getLabels(path string) ([]uint8, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open file: %s, %v", path, err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Fatalf("Failed to close file: %v", err)
		}
	}(file)

	var header labelHeader
	if err := binary.Read(file, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("could not read label file header from: %s, %v", path, err)
	}

	if header.MagicNumber != labelMagic {
		return nil, fmt.Errorf("invalid header read from label file: %s (%08X not %08X)", path, header.MagicNumber, labelMagic)
	}

	labels := make([]uint8, header.NumberOfLabels)
	if _, err := file.Read(labels); err != nil {
		return nil, fmt.Errorf("could not read labels from: %s, %v", path, err)
	}

	return labels, nil
}

func getImages(path string) ([]mnistImage, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open file: %s, %v", path, err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Fatalf("Failed to close file: %v", err)
		}
	}(file)

	var header imageHeader
	if err := binary.Read(file, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("could not read image file header from: %s, %v", path, err)
	}

	if header.MagicNumber != imageMagic {
		return nil, fmt.Errorf("invalid header read from image file: %s (%08X not %08X)", path, header.MagicNumber, imageMagic)
	}

	images := make([]mnistImage, header.NumberOfImages)
	if err := binary.Read(file, binary.BigEndian, &images); err != nil {
		return nil, fmt.Errorf("could not read images from: %s, %v", path, err)
	}

	return images, nil
}

func getDataset(imagePath, labelPath string) (*mnistDataset, error) {
	images, err := getImages(imagePath)
	if err != nil {
		return nil, err
	}

	labels, err := getLabels(labelPath)
	if err != nil {
		return nil, err
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("number of images does not match number of labels (%d != %d)", len(images), len(labels))
	}

	return &mnistDataset{Images: images, Labels: labels, Size: len(images)}, nil
}

func batch(dataset *mnistDataset, size, number int) *mnistDataset {
	start := size * number
	if start >= len(dataset.Images) {
		return &mnistDataset{}
	}

	end := start + size
	if end > len(dataset.Images) {
		end = len(dataset.Images)
	}

	return &mnistDataset{
		Images: dataset.Images[start:end],
		Labels: dataset.Labels[start:end],
		Size:   end - start,
	}
}
