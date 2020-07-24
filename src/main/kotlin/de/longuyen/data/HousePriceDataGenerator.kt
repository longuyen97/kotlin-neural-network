package de.longuyen.data

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.InputStreamReader

class HousePriceDataGenerator : DataGenerator {
    fun readRawCsvToMap(csvReader: CSVParser, training: Boolean): Map<Int, MutableList<String>> {
        val returnValue = HashMap<Int, MutableList<String>>()
        for (csvRecord in csvReader) {
            for (attribute in HousePriceDataAttributes.values()) {
                if ((training && attribute.target) || !attribute.target) {
                    if (!returnValue.containsKey(attribute.index)) {
                        returnValue[attribute.index] = mutableListOf()
                    }
                    returnValue[attribute.index]!!.add(csvRecord.get(attribute.index))
                }
            }
        }
        assert(training && returnValue.size == HousePriceDataAttributes.values().size) { "Expecting raw data to have ${HousePriceDataAttributes.values().size} columns. Instead got ${returnValue.size} columns" }
        if(!training) {
            assert(returnValue.size == HousePriceDataAttributes.values().size - 1) { "Expecting raw data to have ${HousePriceDataAttributes.values().size - 1} columns. Instead got ${returnValue.size} columns" }
        }
        return returnValue
    }

    override fun getTrainingData(): INDArray {
        javaClass.getResourceAsStream("/train.csv").use { inputStream ->
            InputStreamReader(inputStream).use {inputStreamReader ->
                val csvParser = CSVParser(inputStreamReader, CSVFormat.DEFAULT)
                val csvData = readRawCsvToMap(csvParser, true)
                return Nd4j.zeros(3, 3)
            }
        }
    }

    override fun getTestingData(): INDArray {
        javaClass.getResourceAsStream("/test.csv").use { inputStream ->
            InputStreamReader(inputStream).use {inputStreamReader ->
                val csvParser = CSVParser(inputStreamReader, CSVFormat.DEFAULT)
                val csvData = readRawCsvToMap(csvParser, false)
                return Nd4j.zeros(3, 3)
            }
        }
    }

}