package de.longuyen.data

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.InputStream
import java.io.InputStreamReader


interface DataGenerator {
    fun getTrainingData(): INDArray

    fun getTestingData(): INDArray
}

class HousePriceDataGenerator : DataGenerator{
    fun getData(inputStream: InputStream) : INDArray {
        InputStreamReader(inputStream).use {
            val csvParser = CSVParser(it, CSVFormat.DEFAULT)

            return Nd4j.zeros(3, 3)
        }
    }

    override fun getTrainingData(): INDArray {
        return this.getData(javaClass.getResourceAsStream("/train.csv"))
    }

    override fun getTestingData(): INDArray {
        return this.getData(javaClass.getResourceAsStream("/test.csv"))
    }

}