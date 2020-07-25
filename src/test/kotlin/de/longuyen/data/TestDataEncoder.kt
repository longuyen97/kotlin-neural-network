package de.longuyen.data

import org.junit.Test

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
        val continniousNormalization = normalizeContinuousAttributes(valData)
        println(discreteEncoding)
        println(continniousNormalization)
    }
}