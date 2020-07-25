package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray


/**
 * Help class to memorize how the discrete attributes were encoded
 */
data class DiscreteAttributesCode(val index: Int, val oneHotEncoded: Map<String, Int>)

/**
 * Help class to memorize how the continuous attributes were encoded
 */
data class ContinuousAttributesCode(val index: Int, val min: Double, val max: Double)

/**
 * Interface for encoding data
 */
abstract class DataEncoder(val dataFrame: Map<Int, MutableList<String>>) {
    abstract fun encode(): Pair<INDArray, INDArray>
    abstract fun encode(dataFrame: Map<Int, MutableList<String>>): INDArray
}