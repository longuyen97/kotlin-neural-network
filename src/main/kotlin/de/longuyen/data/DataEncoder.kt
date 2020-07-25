package de.longuyen.data


/**
 * Help class to memorize how the discrete attributes were encoded
 */
class DiscreteAttributesCode(val index: Int, val oneHotEncoded: Map<Int, Int>)

/**
 * Help class to memorize how the continuous attributes were encoded
 */
class ContinuousAttributesCode(val index: Int, val min: Double, val max: Double)

/**
 * Interface for encoding data
 */
abstract class DataEncoder(val dataFrame: Map<Int, MutableList<String>>) {
    /**
     * Encoding discrete attributes of the data frame
     */
    abstract fun encodeDiscreteAttributes() : Map<Int, DiscreteAttributesCode>

    /**
     * Normalize the continuous attributes of the data frame
     */
    abstract fun normalizeContinuousAttributes()  : Map<Int, ContinuousAttributesCode>

    /**
     * Produce the final data
     */
    abstract fun produce() : Map<Int, MutableList<String>>
}