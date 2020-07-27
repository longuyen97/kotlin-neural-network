package de.longuyen.data.houseprice

import org.junit.Test
import java.util.*
import kotlin.collections.HashMap
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TestDataEncoder {
    @Test
    fun `Test methode encodeDiscreteAttributess with only one lines`() {
        val dataGenerator = CsvDataReader()
        val valData = dataGenerator.getValidatingData().toMutableMap()
        valData.remove(0)

        encodeDiscreteAttributes(valData)
        normalizeContinuousAttributes(valData)
    }

    @Test
    fun `Test methode encodeDiscreteAttributess with training data`() {
        val dataGenerator = CsvDataReader()
        val valData = dataGenerator.getTrainingData().toMutableMap()
        valData.remove(0)
        val discreteEncoding = encodeDiscreteAttributes(valData)
        val continuousNormalization = normalizeContinuousAttributes(valData)

        val dataEncoder = DataEncoder(dataGenerator.getTrainingData().toMutableMap())
        assertEquals(discreteEncoding, dataEncoder.discreteMapping)
        assertEquals(continuousNormalization, dataEncoder.continuousMapping)
        assertTrue(dataEncoder.encoded.containsKey(HousePriceDataAttributes.SalePrice.index))


        val indexToAttributeMap = {
            val ret = HashMap<Int, HousePriceDataAttributes>()
            for (i in HousePriceDataAttributes.values()) {
                ret[i.index] = i
            }
            ret
        }()

        for (discreteIndex in discreteEncoding.keys) {
            assertEquals(indexToAttributeMap[discreteIndex]!!.dataType, DataType.DISCRETE)
            assertFalse(valData.containsKey(discreteIndex))
            for (oneHotEncodedKey in discreteEncoding[discreteIndex]!!.oneHotEncoded.values) {
                assertTrue(valData.containsKey(oneHotEncodedKey))
            }
        }

        for (continuousIndex in continuousNormalization.keys) {
            assertEquals(indexToAttributeMap[continuousIndex]!!.dataType, DataType.CONTINUOUS)
            assertTrue(valData.containsKey(continuousIndex))
        }

        val firstColumnLength = valData.values.first().size;
        for (value in valData.values) {
            assertEquals(firstColumnLength, value.size)
        }
    }

    @Test
    fun `test methode encode`() {
        val dataGenerator = CsvDataReader()
        val trainingData = dataGenerator.getTrainingData().toMutableMap()
        assertTrue(trainingData.containsKey(HousePriceDataAttributes.SalePrice.index))
        val dataEncoder = DataEncoder(trainingData)
        assertTrue(dataEncoder.encoded.containsKey(HousePriceDataAttributes.SalePrice.index))
        val data = dataEncoder.encode()
        assertTrue(Arrays.equals(data.first.shape(), longArrayOf(318, 1460)))
        assertTrue(Arrays.equals(data.second.shape(), longArrayOf(1, 1460)))
        assertEquals(data.first.shape()[1], trainingData[trainingData.keys.first()]!!.size.toLong())
        assertEquals(data.second.shape()[1], trainingData[trainingData.keys.first()]!!.size.toLong())
        assertEquals(data.first.shape()[0], dataEncoder.encoded.size.toLong() - 1)
    }

    @Test
    fun `test methode encode with testing data`(){
        val dataGenerator = CsvDataReader()
        val trainingData = dataGenerator.getTrainingData().toMutableMap()
        val testingData = dataGenerator.getTestingData().toMutableMap()
        val valData = dataGenerator.getValidatingData().toMutableMap()

        assertTrue(trainingData.containsKey(HousePriceDataAttributes.SalePrice.index))
        val dataEncoder = DataEncoder(trainingData)
        assertTrue(dataEncoder.encoded.containsKey(HousePriceDataAttributes.SalePrice.index))
        val testingNdarray = dataEncoder.encodeFutureData(testingData)
        assertTrue(Arrays.equals(testingNdarray.shape(), longArrayOf(318, 1459)))
        val validatingNdarray = dataEncoder.encodeFutureData(valData)
        assertTrue(Arrays.equals(validatingNdarray.shape(), longArrayOf(318, 1)))
    }
}