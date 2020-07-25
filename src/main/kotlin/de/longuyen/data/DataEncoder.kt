package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray


/**
 * Help class to memorize how the discrete attributes were encoded. Assuming the data in the column is discrete and can not be compared
 * @param index Indicates the original CSV column's index of the not encoded file. This original column will be removed after one hot encoding
 * @param oneHotEncoded Map each unique value in the column to a new index
 */
data class DiscreteAttributesCode(val index: Int, val oneHotEncoded: Map<String, Int>)

/**
 * Help class to memorize how the continuous attributes were encoded. Assuming the data in the column is continuous and can be compared
 * @param index Indicates the original CSV's column's index
 * @param min minimal value of the column
 * @param max maximal value of the column
 */
data class ContinuousAttributesCode(val index: Int, val min: Double, val max: Double)

/**
 * Interface for encoding data
 */
abstract class DataEncoder(val dataFrame: Map<Int, MutableList<String>>) {
    /**
     * Encode the input parameter of the constructor
     */
    abstract fun encode(): Pair<INDArray, INDArray>

    /**
     * Encode the input parameter @param dataFrame
     */
    abstract fun encode(dataFrame: Map<Int, MutableList<String>>): INDArray
}