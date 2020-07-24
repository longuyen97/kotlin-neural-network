package de.longuyen.data

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.junit.Test
import java.io.InputStreamReader
import kotlin.test.assertEquals
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

    @Test
    fun testReadCsv(){
        val dataGenerator = HousePriceDataGenerator()
        val inputStream = javaClass.getResourceAsStream("/train.csv")
        InputStreamReader(inputStream).use {
            val csvParser = CSVParser(it, CSVFormat.DEFAULT)
            val csvData = dataGenerator.readRawCsvToMap(csvParser, true)
            assertTrue { csvData.isNotEmpty() }

            val firstColumnSize = csvData[0]?.size
            for(i in csvData.entries){
                assertEquals(i.value.size, firstColumnSize)
            }
        }
    }
}