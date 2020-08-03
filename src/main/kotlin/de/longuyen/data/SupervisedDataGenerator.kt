package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray

/**
 * Interface to generating supervised data
 */
interface SupervisedDataGenerator {
    /**
     * Training data. First element of the return is the feature, second element of the pair is the target
     */
    fun getTrainingData() :  Pair<INDArray, INDArray>

    /**
     * Testing data without labels. The data doesn't have any target
     */
    fun getTestingData() : INDArray

    /**
     * Testing data with labels for model's validation.
     */
    fun getTestingDataWithLabels() :  Pair<INDArray, INDArray>
}