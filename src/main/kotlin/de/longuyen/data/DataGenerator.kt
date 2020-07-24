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