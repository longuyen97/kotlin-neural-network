package de.longuyen.data.mnist

import de.longuyen.data.SupervisedDataGenerator
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.awt.image.BufferedImage
import java.io.File
import java.io.InputStreamReader
import javax.imageio.ImageIO

val debug = false


fun debugFeatureAndTarget(feature: DoubleArray, target: String, targets: DoubleArray){
    val features = Nd4j.createFromArray(arrayOf(feature)).reshape(intArrayOf(28, 28)).toDoubleMatrix()
    val bufferedImage = BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
    for(y in 0 until 28){
        for(x in 0 until 28){
            val color = features[y][x].toInt()
            val grayScale = (color shl 16) + (color shl 8) + color
            bufferedImage.setRGB(x, y, grayScale)
        }
    }
    ImageIO.write(bufferedImage, "png", File("target/$target"))
}

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
        var X = Nd4j.createFromArray(convert(featureList)).castTo(DataType.DOUBLE)
        var Y = Nd4j.createFromArray(convert(targetList)).castTo(DataType.DOUBLE)
        X = X.transpose()
        Y = Y.transpose()
        X = X.div(255.0)

        if(debug) {
            val index = 1
            var x =  X.get(NDArrayIndex.interval(0, 784), NDArrayIndex.interval(index, index + 1)).mul(255.0)
            var y = Y.get(NDArrayIndex.interval(0, 10), NDArrayIndex.interval(index, index + 1))
            x = x.transpose()
            y = y.transpose()
            debugFeatureAndTarget(x.toDoubleVector(), Nd4j.argMax(y, 1).element().toString(), y.toDoubleVector())
        }

        return Pair(X, Y)

    }

    override fun getTrainingData(): Pair<INDArray, INDArray> {
        return getData("/mnist/train.csv")
    }

    override fun getTestingData(): INDArray {
        return getData("/mnist/validation.csv").first
    }

    override fun getTestingDataWithLabels(): Pair<INDArray, INDArray> {
        return getData("/mnist/test.csv")
    }
}