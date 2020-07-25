package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray


/**
 * Interface for generating data
 */
interface DataGenerator {
    /**
     * Generating training data
     */
    fun getTrainingData(): Map<Int, MutableList<String>>

    /**
     * Generating testing data
     */
    fun getTestingData(): Map<Int, MutableList<String>>

    /**
     * Generating  training data for unit testing purpose
     */
    fun getValidatingData(): Map<Int, MutableList<String>>
}