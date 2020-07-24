package de.longuyen.data

import org.junit.Test
import kotlin.test.assertTrue

class TestDataGenerator {
    @Test
    fun testGenerator() {
        val dataGenerator = HousePriceDataGenerator()
        val inputStream = javaClass.getResourceAsStream("/train.csv")
        assert(inputStream != null)
        val data = dataGenerator.getData(inputStream)
        assertTrue { data.length() > 0 }
    }
}