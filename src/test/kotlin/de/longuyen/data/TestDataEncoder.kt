package de.longuyen.data

import org.junit.Test

class TestDataEncoder{
    @Test
    fun `Test methode encodeDiscreteAttributess`(){
        val dataGenerator = HousePriceDataGenerator()
        val valData = dataGenerator.getValidatingData().toMutableMap()
        println(valData)
        valData.remove(0)
        encodeDiscreteAttributes(valData)
        println(valData)
        normalizeContinuousAttributes(valData)
        println(valData)
    }
}