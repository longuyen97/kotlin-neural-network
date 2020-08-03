package de.longuyen.data.mnist

import de.longuyen.data.SupervisedDataGenerator
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.io.InputStreamReader

/**
 * MNIST data generator
 * Original downloaded from https://www.kaggle.com/oddrationale/mnist-in-csv
 */
class MnistDataGenerator : SupervisedDataGenerator {
    /**
     * Utility method for generating data from a CSV file
     */
    private fun getData(path: String): Pair<INDArray, INDArray> {
        val featureList = mutableListOf<DoubleArray>()
        val targetList = mutableListOf<DoubleArray>()

        // Read the CSV file and parse the features and the target into 2D list
        InputStreamReader(File(path).inputStream()).use {
            CSVParser(it, CSVFormat.DEFAULT).use { csvRecords ->
                var skip = false

                // Read each image
                for (record in csvRecords) {
                    // Skip the first line since it is only the columns' names
                    if (skip) {

                        // One hot encode target
                        val target = DoubleArray(10) { 0.0 }
                        target[record[0].toInt()] = 1.0

                        // Read the image and flatten it into a 1D image
                        val feature = DoubleArray(record.size() - 1) { 0.0 }
                        for (i in 1 until record.size()) {
                            feature[i - 1] = record[i].toDouble()
                        }
                        featureList.add(feature)
                        targetList.add(target)
                    } else {
                        skip = true
                    }
                }
            }
        }

        // Convert 2D list into 2D Array
        val convert = {list: MutableList<DoubleArray> ->
            val ret = arrayOfNulls<DoubleArray>(list.size)
            for(y in list.indices){
                val colArray = DoubleArray(list[y].size)
                for(x in list[y].indices){
                    colArray[x] = list[y][x]
                }
                ret[y] = colArray
            }
            ret
        }

        // Normalize data
        var X = Nd4j.createFromArray(convert(featureList)).castTo(DataType.DOUBLE)
        var Y = Nd4j.createFromArray(convert(targetList)).castTo(DataType.DOUBLE)

        // Transpose the tensor
        X = X.transpose()
        Y = Y.transpose()
        X = X.div(255.0)

        return Pair(X, Y)

    }

    override fun getTrainingData(): Pair<INDArray, INDArray> {
        return getData("data/mnist/train.csv")
    }

    override fun getTestingData(): INDArray {
        return getData("data/mnist/validation.csv").first
    }

    override fun getTestingDataWithLabels(): Pair<INDArray, INDArray> {
        return getData("data/mnist/test.csv")
    }
}