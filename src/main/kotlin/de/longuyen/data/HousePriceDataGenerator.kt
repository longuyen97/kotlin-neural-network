package de.longuyen.data

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.InputStreamReader

/**
 * Data generator for the dataset Boston House price.
 */
class HousePriceDataGenerator : DataGenerator {
    /**
     * Undertake the task of parsing the raw csv file into a Map.
     * For the map we assume the keys are the indices of each attribute in the CSV table
     * and the values are the list value at each row.
     *
     * Each values of the map therefore must have the same length.
     * @param csvReader: The data object of the CSV, ready to be parsed
     * @param training: Indicates if the data object contains a target attribute or not
     */
    fun readRawCsvToMap(csvReader: CSVParser, training: Boolean): Map<Int, MutableList<String>> {
        val returnValue = HashMap<Int, MutableList<String>>()

        // Iterate each record in the CSV
        for (csvRecord in csvReader) {
            // Iterate each value of the record
            for (attribute in HousePriceDataAttributes.values()) {
                if ((training && attribute.target) || !attribute.target) {
                    if (!returnValue.containsKey(attribute.index)) {
                        returnValue[attribute.index] = mutableListOf()
                    }
                    returnValue[attribute.index]!!.add(csvRecord.get(attribute.index))
                }
            }
        }
        if(!training) {
            assert(returnValue.size == HousePriceDataAttributes.values().size - 1) { "Expecting raw data to have ${HousePriceDataAttributes.values().size - 1} columns. Instead got ${returnValue.size} columns" }
        }else{
            assert(training && returnValue.size == HousePriceDataAttributes.values().size) { "Expecting raw data to have ${HousePriceDataAttributes.values().size} columns. Instead got ${returnValue.size} columns" }
        }
        return returnValue
    }

    override fun getTrainingData(): Pair<INDArray, INDArray> {
        javaClass.getResourceAsStream("/train.csv").use { inputStream ->
            InputStreamReader(inputStream).use {inputStreamReader ->
                val csvParser = CSVParser(inputStreamReader, CSVFormat.DEFAULT)
                val csvData = readRawCsvToMap(csvParser, true)
                return Pair(Nd4j.zeros(3, 3), Nd4j.ones(3, 3))
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