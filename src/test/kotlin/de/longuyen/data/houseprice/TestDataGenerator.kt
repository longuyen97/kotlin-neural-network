package de.longuyen.data.houseprice

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.junit.Test
import java.io.InputStreamReader
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TestDataGenerator {
    @Test
    fun `Test method read raw csv to map with training data`(){
        val dataGenerator = CsvDataReader()
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

    @Test
    fun `Test method read raw csv to map with testing data`(){
        val dataGenerator = CsvDataReader()
        val inputStream = javaClass.getResourceAsStream("/test.csv")
        InputStreamReader(inputStream).use {
            val csvParser = CSVParser(it, CSVFormat.DEFAULT)
            val csvData = dataGenerator.readRawCsvToMap(csvParser, false)
            assertTrue { csvData.isNotEmpty() }

            val firstColumnSize = csvData[0]?.size
            for(i in csvData.entries){
                assertEquals(i.value.size, firstColumnSize)
            }
        }
    }
}