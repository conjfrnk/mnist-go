package main

import (
	"encoding/binary"
	"fmt"
	"os"
)

type mnistLabelFileHeader struct {
	MagicNumber    uint32
	NumberOfLabels uint32
}

type mnistImageFileHeader struct {
	MagicNumber     uint32
	NumberOfImages  uint32
	NumberOfRows    uint32
	NumberOfColumns uint32
}

type mnistImage struct {
	Pixels [mnistImageSize]uint8
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
	defer file.Close()

	var header mnistLabelFileHeader
	if err := binary.Read(file, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("could not read label file header from: %s, %v", path, err)
	}

	if header.MagicNumber != mnistLabelMagic {
		return nil, fmt.Errorf("invalid header read from label file: %s (%08X not %08X)", path, header.MagicNumber, mnistLabelMagic)
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
	defer file.Close()

	var header mnistImageFileHeader
	if err := binary.Read(file, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("could not read image file header from: %s, %v", path, err)
	}

	if header.MagicNumber != mnistImageMagic {
		return nil, fmt.Errorf("invalid header read from image file: %s (%08X not %08X)", path, header.MagicNumber, mnistImageMagic)
	}

	images := make([]mnistImage, header.NumberOfImages)
	if err := binary.Read(file, binary.BigEndian, &images); err != nil {
		return nil, fmt.Errorf("could not read images from: %s, %v", path, err)
	}

	return images, nil
}

func mnistGetDataset(imagePath, labelPath string) (*mnistDataset, error) {
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

func mnistBatch(dataset *mnistDataset, size, number int) *mnistDataset {
	startOffset := size * number
	if startOffset >= len(dataset.Images) {
		return &mnistDataset{}
	}

	endOffset := startOffset + size
	if endOffset > len(dataset.Images) {
		endOffset = len(dataset.Images)
	}

	return &mnistDataset{
		Images: dataset.Images[startOffset:endOffset],
		Labels: dataset.Labels[startOffset:endOffset],
		Size:   endOffset - startOffset,
	}
}
