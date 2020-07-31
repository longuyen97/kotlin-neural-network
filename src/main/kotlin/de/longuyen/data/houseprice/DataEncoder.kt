package de.longuyen.data.houseprice

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable
import kotlin.collections.HashMap


/**
 * Help class to memorize how the discrete attributes were encoded. Assuming the data in the column is discrete and can not be compared
 * @param index Indicates the original CSV column's index of the not encoded file. This original column will be removed after one hot encoding
 * @param oneHotEncoded Map each unique value in the column to a new index
 */
data class DiscreteAttributesCode(val index: Int, val oneHotEncoded: Map<String, Int>) : Serializable

/**
 * Help class to memorize how the continuous attributes were encoded. Assuming the data in the column is continuous and can be compared
 * @param index Indicates the original CSV's column's index
 * @param min minimal value of the column
 * @param max maximal value of the column
 */
data class ContinuousAttributesCode(val index: Int, val min: Double, val max: Double) : Serializable

/**
 * Class for bringing the house price dataset in the correct form.
 * The dataset contains continuous and discrete data
 * The continuous data column can be kept, but the data itself needs to be normalized to between 0.0 and 1.0
 * The discrete data column must be deleted. Before deletion, the data must be one hot encoded.
 * Each column of the one hot encoding matrix will be inserted into the data frame.
 *
 * The object of this class will associated wiith one and only one dataset. The object can be serialized and deserialized
 * for serving the model. Incoming data must be processed for the inference to happen.
 *
 * @param dataFrame the orignal data. Will be edited inplace.
 */
class DataEncoder(private val dataFrame: Map<Int, MutableList<String>>) : Serializable {
    // The one hot encoding information for encoding future data
    val discreteMapping: MutableMap<Int, DiscreteAttributesCode>

    // The information of min / max data for future scaling
    val continuousMapping: MutableMap<Int, ContinuousAttributesCode>

    // The original dataset. Encoded
    val encoded: MutableMap<Int, MutableList<String>> = this.dataFrame.toMutableMap()

    init {
        if (this.encoded.containsKey(HousePriceDataAttributes.Id.index)) {
            this.encoded.remove(HousePriceDataAttributes.Id.index)
        }
        this.discreteMapping = encodeDiscreteAttributes(this.encoded)
        this.continuousMapping = normalizeContinuousAttributes(this.encoded)
    }

    fun encode(): Pair<INDArray, INDArray> {
        // Convert the value of the target column to a 2D double list
        val target = {
            val ret = mutableListOf<Double>()
            for (value in this.encoded[HousePriceDataAttributes.SalePrice.index]!!) {
                ret.add(value.toDouble())
            }
            mutableListOf(ret)
        }()

        // Convert the value of the 2D double list to a 2D double array
        val targetNdArray = {
            val ret = DoubleArray(target[0].size)
            for (i in target[0].indices) {
                ret[i] = target[0][i]
            }
            arrayOf(ret)
        }()

        // Convert the value of the features columns to a 2D double list
        val features = {
            val ret = mutableListOf<MutableList<Double>>()
            for (key in encoded.keys) {
                if (key != HousePriceDataAttributes.SalePrice.index) {
                    val column = mutableListOf<Double>()
                    for (value in encoded[key]!!) {
                        column.add(value.toDouble())
                    }
                    ret.add(column)
                }
            }
            ret
        }()

        // Convert the value of the target column to a 2D array
        val featuresNdarray = {
            val ret = arrayOfNulls<DoubleArray>(features.size)
            for (featureColumnIndex in features.indices) {
                val featureColumn = features[featureColumnIndex]
                val colArray = DoubleArray(featureColumn.size)
                for (cellIndex in featureColumn.indices) {
                    colArray[cellIndex] = featureColumn[cellIndex]
                }
                ret[featureColumnIndex] = colArray
            }
            ret
        }()

        return Pair(Nd4j.createFromArray(featuresNdarray), Nd4j.createFromArray(targetNdArray))
    }

    fun encodeFutureData(dataFrame: Map<Int, MutableList<String>>): INDArray {
        val outputMap = dataFrame.toMutableMap()
        outputMap.remove(HousePriceDataAttributes.Id.index)

        // Map each discrete column to an encoding object
        val keySet = this.discreteMapping.keys.toSet()
        for (discreteKey in keySet) {
            if (!outputMap.containsKey(discreteKey)) {
                throw RuntimeException("Error. The future data does not contain the column $discreteKey")
            }

            // Encoding object
            // Map each value of this discrete column to a new column
            val oneHotCodes = this.discreteMapping[discreteKey]!!.oneHotEncoded
            val oneHotSubArray = mutableMapOf<String, MutableList<String>>()
            for (entry in oneHotCodes.entries) {
                oneHotSubArray[entry.key] = mutableListOf()
            }
            for (value in outputMap[discreteKey]!!) {
                for (entry in oneHotSubArray.entries) {
                    if (entry.key == value) {
                        entry.value.add("1")
                    } else {
                        entry.value.add("0")
                    }
                }
            }

            // Merge the one hot sub array into the main data frame
            val discreteEncode = discreteMapping[discreteKey]!!
            for (entry in discreteEncode.oneHotEncoded.entries) {
                outputMap[entry.value] = oneHotSubArray[entry.key]!!
            }

            // Remove the original discrete
            outputMap.remove(discreteKey)
        }

        // Convert the value of the features columns to a 2D double list
        val features = {
            val ret = mutableListOf<MutableList<Double>>()
            for (key in outputMap.keys) {
                if (key != HousePriceDataAttributes.SalePrice.index) {
                    val column = mutableListOf<Double>()
                    for (value in outputMap[key]!!) {
                        try {
                            column.add(value.toDouble())
                        } catch (e: NumberFormatException) {
                            column.add(0.0)
                        }
                    }
                    ret.add(column)
                }
            }
            ret
        }()

        // Convert the value of the target column to a 2D array
        val featuresNdarray = {
            val ret = arrayOfNulls<DoubleArray>(features.size)
            for (featureColumnIndex in features.indices) {
                val featureColumn = features[featureColumnIndex]
                val colArray = DoubleArray(featureColumn.size)
                for (cellIndex in featureColumn.indices) {
                    colArray[cellIndex] = featureColumn[cellIndex]
                }
                ret[featureColumnIndex] = colArray
            }
            ret
        }()

        return Nd4j.createFromArray(featuresNdarray)
    }
}

/**
 * In place discrete features encoding
 * @return the information for future encoding
 */
fun encodeDiscreteAttributes(dataFrame: MutableMap<Int, MutableList<String>>): MutableMap<Int, DiscreteAttributesCode> {
    val returnValue = HashMap<Int, DiscreteAttributesCode>()

    // Current biggest index.
    // If the dataset has 81 attributes, max will be 81.
    // One hot encode the attributes by incrementing
    var maximalIndex = dataFrame.keys.max()

    // Map each index to the corresponding attribute enum
    val indexToAttributeMap = {
        val ret = HashMap<Int, HousePriceDataAttributes>()
        for (i in HousePriceDataAttributes.values()) {
            ret[i.index] = i
        }
        ret
    }()

    val dataFrameKeys = dataFrame.keys.toSet()

    // Iterate every column of the data frame
    for (attributeIndex in dataFrameKeys) {

        // Check if the column is suitable for discrete encoding
        if (indexToAttributeMap.containsKey(attributeIndex) && !indexToAttributeMap[attributeIndex]!!.target) {

            // Check if the column exists in the hard coded enum class for house price dataset's attributes
            if (indexToAttributeMap[attributeIndex]!!.dataType == DataType.DISCRETE) {

                // Collect the unique values of the column
                val uniqueDiscreteValues = {
                       val ret = mutableSetOf<String>()
                    for (value in dataFrame[attributeIndex]!!) {
                        ret.add(value)
                    }
                    ret
                }()

                // Create the one hot encode map for each unique value
                // Map the unique value with a new column
                val oneHotEncodeMap = HashMap<String, Int>()

                // Map the unique value with the intern index of the one hot encoded data frame
                val segmentIndexMap = HashMap<String, Int>()

                for ((segmentIndex, uniqueValue) in uniqueDiscreteValues.withIndex()) {
                    maximalIndex = maximalIndex!! + 1

                    assert(!dataFrame.containsKey(maximalIndex)) { "The newly created index for the data frame should not be present in the data frame $maximalIndex" }
                    assert(!oneHotEncodeMap.containsKey(uniqueValue)) { "The newly created index for the data frame should not be present in the one hot encode map $maximalIndex" }

                    oneHotEncodeMap[uniqueValue] = maximalIndex
                    segmentIndexMap[uniqueValue] = segmentIndex
                }

                // Map the memorized one hot encoded map in the return value of the method
                assert(!returnValue.containsKey(attributeIndex)) { "The attributes encoding should not be present in the return value $attributeIndex" }
                returnValue[attributeIndex] = DiscreteAttributesCode(attributeIndex, oneHotEncodeMap)

                // Start one hot encoding
                val oneHotEncodes = {
                    val ret = mutableListOf<MutableList<String>>()
                    for (i in uniqueDiscreteValues.indices) {
                        ret.add(mutableListOf())
                    }
                    ret
                }()

                for (value in dataFrame[attributeIndex]!!) {
                    for (oneHotIndex in oneHotEncodes.indices) {
                        if (oneHotIndex == segmentIndexMap[value]!!) {
                            oneHotEncodes[oneHotIndex].add("1")
                        } else {
                            oneHotEncodes[oneHotIndex].add("0")
                        }
                    }
                }

                for (segmentKey in segmentIndexMap.keys) {
                    dataFrame[oneHotEncodeMap[segmentKey]!!] = oneHotEncodes[segmentIndexMap[segmentKey]!!]
                }
                dataFrame.remove(attributeIndex)
            }
        }
    }
    return returnValue
}

/**
 * In place continuous features scaling
 * @return the information for future scaling
 */
fun normalizeContinuousAttributes(dataFrame: MutableMap<Int, MutableList<String>>): MutableMap<Int, ContinuousAttributesCode> {
    val returnValue = HashMap<Int, ContinuousAttributesCode>()

    // Map each index to the corresponding attribute enum
    val indexToAttributeMap = {
        val ret = HashMap<Int, HousePriceDataAttributes>()
        for (i in HousePriceDataAttributes.values()) {
            ret[i.index] = i
        }
        ret
    }()

    val dataFrameKeys = dataFrame.keys.toSet()

    // Iterate every column of the data frame
    for (attributeIndex in dataFrameKeys) {

        // Check if the column is suitable for discrete encoding
        if (indexToAttributeMap.containsKey(attributeIndex) && !indexToAttributeMap[attributeIndex]!!.target) {

            // Check if the column exists in the hard coded enum class for house price dataset's attributes
            if (indexToAttributeMap[attributeIndex]!!.dataType == DataType.CONTINUOUS) {

                // Convert the data of the current column into numeric data for scaling
                val numericList = mutableListOf<Double>()
                for (str in dataFrame[attributeIndex]!!) {
                    try {
                        val value = str.toDouble()
                        numericList.add(value)
                    } catch (e: NumberFormatException) {
                        assert(str == "NA")
                        numericList.add(0.0)
                    }
                }

                // Find the min and max value of the column
                val min = numericList.min()!!
                val max = numericList.max()!!

                if (max > min) {
                    for (index in numericList.indices) {
                        numericList[index] = (numericList[index] - min) / (max - min)
                    }
                } else if (max == min) {
                    for (index in numericList.indices) {
                        numericList[index] = 1.0
                    }
                }

                // Reconvert the numeric data back to string
                val stringList = mutableListOf<String>()
                for (num in numericList) {
                    stringList.add("$num")
                }
                dataFrame[attributeIndex] = stringList

                // Replace the original column with the scaled column
                returnValue[attributeIndex] = ContinuousAttributesCode(attributeIndex, min, max)
            }
        }
    }
    return returnValue
}