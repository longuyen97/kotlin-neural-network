package de.longuyen.trainer

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.houseprice.HousePriceDataGenerator
import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.NoActivation
import de.longuyen.neuronalnetwork.initializers.ChainInitializer
import de.longuyen.neuronalnetwork.losses.MAE
import de.longuyen.neuronalnetwork.optimizers.MomentumGradientDescent
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.Serializable

/**
 * Driver code to run a model on the house price data set
 */
class HousePriceModelTrainer(layers: IntArray, learningRate: Double, private val epochs: Long) :
    Serializable {
    private val neuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(layers, ChainInitializer(), LeakyRelu(), NoActivation(), MAE(), MomentumGradientDescent(learningRate))

    fun train() {
        val housePriceData: SupervisedDataGenerator = HousePriceDataGenerator()
        val trainingData = housePriceData.getTrainingData()

        val X = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(0, 1000))
        val Y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 1000))
        val x = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(1000, 1460))
        val y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1000, 1460))

        val losses = neuronalNetwork.train(X, Y, x, y, epochs)
        val xData = DoubleArray(losses.first.size)
        for (i in 0 until losses.first.size) {
            xData[i] = i.toDouble()
        }
        val yTrain = losses.first.toDoubleArray()
        val yTest = losses.second.toDoubleArray()

        val chart: XYChart = XYChartBuilder()
            .width(1200)
            .height(1200)
            .title("Two-layer-network's performance")
            .xAxisTitle("Training epochs")
            .yAxisTitle("Model's mean absolute error loss")
            .build()
        chart.addSeries("Training", xData, yTrain)
        chart.addSeries("Validating", xData, yTest)
        BitmapEncoder.saveBitmap(chart, "target/performance", BitmapEncoder.BitmapFormat.PNG)
    }
}

fun main() {
    val trainer = HousePriceModelTrainer(intArrayOf(318, 32, 1), 0.001, 1000)
    trainer.train()
}