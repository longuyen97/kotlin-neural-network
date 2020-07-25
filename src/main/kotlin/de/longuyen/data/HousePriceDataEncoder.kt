package de.longuyen.data

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.lang.Exception
import java.util.*
import kotlin.collections.HashMap


class HousePriceDataEncoder(dataFrame: Map<Int, MutableList<String>>) : DataEncoder(dataFrame) {
    private var discreteMapping: MutableMap<Int, DiscreteAttributesCode> = HashMap()
    private var continuousMapping: MutableMap<Int, ContinuousAttributesCode> = HashMap()

    init {
        val encoded = this.dataFrame.toMutableMap()
        this.discreteMapping = encodeDiscreteAttributes(encoded)
        this.continuousMapping = normalizeContinuousAttributes(encoded)
    }

    override fun encode(): Pair<INDArray, INDArray> {
        TODO("Not yet implemented")
    }

    override fun encode(dataFrame: Map<Int, MutableList<String>>): INDArray {
        return Nd4j.zeros(3, 3)
    }
}

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
                    assert(oneHotEncodeMap.containsKey(value))
                    oneHotEncodes[segmentIndexMap[value]!!].add("${oneHotEncodeMap[value]!!}")
                }

                for(segmentKey in segmentIndexMap.keys){
                    dataFrame[oneHotEncodeMap[segmentKey]!!] = oneHotEncodes[segmentIndexMap[segmentKey]!!]
                }
                dataFrame.remove(attributeIndex)
            }
        }
    }
    return returnValue
}



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
                val numericList = mutableListOf<Double>()
                for(str in dataFrame[attributeIndex]!!) {
                    try {
                        numericList.add(str.toDouble())
                    }catch (e: Exception){
                        println("$attributeIndex: ${indexToAttributeMap[attributeIndex]!!}")
                        throw e
                    }
                }
                val min = numericList.min()!!
                val max = numericList.max()!!
                for(index in numericList.indices){
                    numericList[index] = (numericList[index] - min) / (max - min)
                }
                val stringList = mutableListOf<String>()
                for(num in numericList){
                    stringList.add("$num")
                }
                dataFrame[attributeIndex] = stringList
            }
        }
    }
    return returnValue
}