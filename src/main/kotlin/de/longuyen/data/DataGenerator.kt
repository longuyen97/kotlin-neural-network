package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray


interface DataGenerator {
    fun getTrainingData(): INDArray

    fun getTestingData(): INDArray
}