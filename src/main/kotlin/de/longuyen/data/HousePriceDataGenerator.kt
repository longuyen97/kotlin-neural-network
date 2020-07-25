package de.longuyen.data

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
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
        var skipped = false
        for (csvRecord in csvReader) {
            if(!skipped){
                skipped = true
            }else {
                for (attribute in HousePriceDataAttributes.values()) {
                    if ((training && attribute.target) || !attribute.target) {
                        if (!returnValue.containsKey(attribute.index)) {
                            returnValue[attribute.index] = mutableListOf()
                        }
                        returnValue[attribute.index]!!.add(csvRecord.get(attribute.index))
                    }
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

    /**
     * Obtain the training data of the house price dataset
     */
    override fun getTrainingData(): Map<Int, MutableList<String>> {
        javaClass.getResourceAsStream("/train.csv").use { inputStream ->
            InputStreamReader(inputStream).use {inputStreamReader ->
                val csvParser = CSVParser(inputStreamReader, CSVFormat.DEFAULT)
                return readRawCsvToMap(csvParser, true)
            }
        }
    }

    /**
     * Obtain the testing data of the house price dataset
     */
    override fun getTestingData(): Map<Int, MutableList<String>> {
        javaClass.getResourceAsStream("/test.csv").use { inputStream ->
            InputStreamReader(inputStream).use {inputStreamReader ->
                val csvParser = CSVParser(inputStreamReader, CSVFormat.DEFAULT)
                return readRawCsvToMap(csvParser, false)
            }
        }
    }

    /**
     * Obtain the testing data of the house price dataset
     */
    override fun getValidatingData(): Map<Int, MutableList<String>> {
        javaClass.getResourceAsStream("/validation.csv").use { inputStream ->
            InputStreamReader(inputStream).use {inputStreamReader ->
                val csvParser = CSVParser(inputStreamReader, CSVFormat.DEFAULT)
                return readRawCsvToMap(csvParser, false)
            }
        }
    }
}