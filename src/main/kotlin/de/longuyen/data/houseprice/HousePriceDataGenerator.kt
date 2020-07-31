package de.longuyen.data.houseprice

import de.longuyen.data.SupervisedDataGenerator
import org.nd4j.linalg.api.ndarray.INDArray

class HousePriceDataGenerator : SupervisedDataGenerator {
    private val csvDataReader: CsvDataReader = CsvDataReader()
    private val dataEncoder: DataEncoder = DataEncoder(csvDataReader.getTrainingData())


    override fun getTrainingData(): Pair<INDArray, INDArray> {
        return dataEncoder.encode()
    }

    override fun getTestingData(): INDArray {
        return dataEncoder.encodeFutureData(csvDataReader.getTestingData())
    }

    override fun getTestingDataWithLabels(): Pair<INDArray, INDArray> {
        TODO("Not implemented")
    }

}