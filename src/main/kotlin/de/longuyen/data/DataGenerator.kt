package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray


/**
 * Interface for generating data
 */
interface DataGenerator {
    /**
     * Generating training pair.
     * First element of the pair is the training attributes
     * Second element of the pair is the training result
     */
    fun getTrainingData(): Pair<INDArray, INDArray>

    /**
     * Generating testing tensor without result
     */
    fun getTestingData(): INDArray
}