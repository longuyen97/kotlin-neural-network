package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray

interface SupervisedDataGenerator {
    fun getTrainingData() :  Pair<INDArray, INDArray>

    fun getTestingData() : INDArray
}