package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.nativeblas.Nd4jCpu


/**
 * Help class to memorize how the discrete attributes were encoded
 */
class DiscreteAttributesCode(val index: Int, val oneHotEncoded: Map<String, Int>)

/**
 * Help class to memorize how the continuous attributes were encoded
 */
class ContinuousAttributesCode(val index: Int, val min: Double, val max: Double)

/**
 * Interface for encoding data
 */
abstract class DataEncoder(val dataFrame: Map<Int, MutableList<String>>) {
    abstract fun encode(): Pair<INDArray, INDArray>
    abstract fun encode(dataFrame: Map<Int, MutableList<String>>): INDArray
}