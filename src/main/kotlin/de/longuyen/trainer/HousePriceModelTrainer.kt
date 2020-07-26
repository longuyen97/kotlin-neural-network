package de.longuyen.trainer

import de.longuyen.data.DataEncoder
import de.longuyen.data.DataGenerator
import de.longuyen.data.HousePriceDataEncoder
import de.longuyen.data.HousePriceDataGenerator
import de.longuyen.neuronalnetwork.LeakyRelu
import de.longuyen.neuronalnetwork.MAE
import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.NoActivation
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.Serializable

class HousePriceModelTrainer(private val layers: IntArray, private val learningRate: Double, private val epochs: Long) : Serializable {
    private val dataGenerator: DataGenerator = HousePriceDataGenerator()
    private val dataEncoder: DataEncoder
    val neuronalNetwork: NeuronalNetwork


    init {
        dataEncoder = HousePriceDataEncoder(dataGenerator.getTrainingData())
        neuronalNetwork = NeuronalNetwork(layers, LeakyRelu(0.01), NoActivation(), MAE())
    }

    fun train() {
        val dataGenerator = HousePriceDataGenerator()
        val trainingData = dataGenerator.getTrainingData()
        val dataEncoder = HousePriceDataEncoder(trainingData)
        val encodedTrainingData = dataEncoder.encode()

        val X = encodedTrainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(0, 1000))
        val Y = encodedTrainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 1000))
        val x = encodedTrainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(1000, 1460))
        val y = encodedTrainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1000, 1460))

        val losses = neuronalNetwork.train(X, Y,x,y, learningRate, epochs)
        val xData = DoubleArray(losses.first.size)
        for (i in 0 until losses.first.size) {
            xData[i] = i.toDouble()
        }
        val yTrain = losses.first.toDoubleArray()
        val yTest = losses.second.toDoubleArray()

        val chart: XYChart = XYChartBuilder().width(600).height(500).title("Two-layer-network's performance").xAxisTitle("X").yAxisTitle("Y").build()
        chart.addSeries("Training", xData, yTrain)
        chart.addSeries("Validating", xData, yTest)

        BitmapEncoder.saveBitmap(chart, "target/Sample_Chart", BitmapEncoder.BitmapFormat.PNG)

        FileOutputStream("target/model.ser").use { fos ->
            ObjectOutputStream(fos).use { os ->
                os.writeObject(neuronalNetwork)
            }
        }
    }
}