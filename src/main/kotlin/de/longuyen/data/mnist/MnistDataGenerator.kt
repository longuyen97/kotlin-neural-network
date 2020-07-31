package de.longuyen.data.mnist

import de.longuyen.data.SupervisedDataGenerator
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.InputStreamReader

class MnistDataGenerator : SupervisedDataGenerator {
    private fun getData(path: String): Pair<INDArray, INDArray> {
        val featureList = mutableListOf<DoubleArray>()
        val targetList = mutableListOf<DoubleArray>()
        InputStreamReader(this.javaClass.getResourceAsStream(path)).use {
            CSVParser(it, CSVFormat.DEFAULT).use { csvRecords ->
                var skip = false
                for (record in csvRecords) {
                    if (skip) {
                        val target = DoubleArray(10) { 0.0 }
                        target[record[0].toInt()] = 1.0
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
        return Pair(Nd4j.createFromArray(convert(featureList)), Nd4j.createFromArray(convert(targetList)))

    }

    override fun getTrainingData(): Pair<INDArray, INDArray> {
        val ret = getData("/mnist/train.csv")
        var X = ret.first.castTo(DataType.DOUBLE)
        var Y = ret.second.castTo(DataType.DOUBLE)
        X = X.reshape(*longArrayOf(X.shape()[1], X.shape()[0]))
        Y = Y.reshape(*longArrayOf(Y.shape()[1], Y.shape()[0]))
        X = (X.div(255.0))
        return Pair(X, Y)
    }

    override fun getTestingData(): INDArray {
        TODO("Not implemented")
    }

    override fun getTestingDataWithLabels(): Pair<INDArray, INDArray> {
        val ret = getData("/mnist/test.csv")
        var X = ret.first.castTo(DataType.DOUBLE)
        var Y = ret.second.castTo(DataType.DOUBLE)
        X = X.reshape(*longArrayOf(X.shape()[1], X.shape()[0]))
        Y = Y.reshape(*longArrayOf(Y.shape()[1], Y.shape()[0]))
        X = (X.div(255.0))
        return Pair(X, Y)
    }
}