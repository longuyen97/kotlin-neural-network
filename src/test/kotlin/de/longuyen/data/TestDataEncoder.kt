package de.longuyen.data

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TestDataEncoder{
    @Test
    fun `Test methode encodeDiscreteAttributess with only one lines`(){
        val dataGenerator = HousePriceDataGenerator()
        val valData = dataGenerator.getValidatingData().toMutableMap()
        valData.remove(0)

        encodeDiscreteAttributes(valData)
        normalizeContinuousAttributes(valData)
        println(valData)
    }

    @Test
    fun `Test methode encodeDiscreteAttributess with training data`(){
        val dataGenerator = HousePriceDataGenerator()
        val valData = dataGenerator.getTrainingData().toMutableMap()
        valData.remove(0)
        val discreteEncoding = encodeDiscreteAttributes(valData)
        val continuousNormalization = normalizeContinuousAttributes(valData)

        val dataEncoder = HousePriceDataEncoder(dataGenerator.getTrainingData().toMutableMap())
        assertEquals(discreteEncoding, dataEncoder.discreteMapping)
        assertEquals(continuousNormalization, dataEncoder.continuousMapping)


        val indexToAttributeMap = {
            val ret = HashMap<Int, HousePriceDataAttributes>()
            for (i in HousePriceDataAttributes.values()) {
                ret[i.index] = i
            }
            ret
        }()

        for(discreteIndex in discreteEncoding.keys){
            assertEquals(indexToAttributeMap[discreteIndex]!!.dataType, DataType.DISCRETE)
            assertFalse(valData.containsKey(discreteIndex))
            for(oneHotEncodedKey in discreteEncoding[discreteIndex]!!.oneHotEncoded.values){
                assertTrue(valData.containsKey(oneHotEncodedKey))
            }
        }

        for(continuousIndex in continuousNormalization.keys){
            assertEquals(indexToAttributeMap[continuousIndex]!!.dataType, DataType.CONTINUOUS)
            assertTrue(valData.containsKey(continuousIndex))
        }
    }
}