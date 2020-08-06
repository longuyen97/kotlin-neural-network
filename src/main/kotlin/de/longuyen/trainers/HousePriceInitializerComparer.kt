package de.longuyen.trainers

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.houseprice.HousePriceDataGenerator
import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.NoActivation
import de.longuyen.neuronalnetwork.initializers.HeInitializer
import de.longuyen.neuronalnetwork.initializers.PositiveOnlyInitializer
import de.longuyen.neuronalnetwork.losses.MAE
import de.longuyen.neuronalnetwork.optimizers.MomentumGradientDescent
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.Serializable

class HousePriceInitializerComparer(layers: IntArray = intArrayOf(318, 64, 32, 1), learningRate: Double = 0.001, private val epochs: Long = 200) :
    Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }
    private val firstNeuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(layers, HeInitializer(), LeakyRelu(), NoActivation(), MAE(), MomentumGradientDescent(learningRate),
            de.longuyen.neuronalnetwork.metrics.MAE()
        )
    private val secondNeuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(layers, PositiveOnlyInitializer(), LeakyRelu(), NoActivation(), MAE(), MomentumGradientDescent(learningRate),
            de.longuyen.neuronalnetwork.metrics.MAE())

    fun train() {
        val housePriceData: SupervisedDataGenerator = HousePriceDataGenerator()
        val trainingData = housePriceData.getTrainingData()

        val x = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(1000, 1460))
        val y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1000, 1460))
        val X = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(0, 1000))
        val Y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 1000))

        val firstHistory = firstNeuronalNetwork.train(X, Y, x, y, epochs)
        val secondHistory = secondNeuronalNetwork.train(X, Y, x, y, epochs)

        val xData = DoubleArray(firstHistory["val-loss"]!!.size)

        for (i in 0 until firstHistory["val-loss"]!!.size) {
            xData[i] = i.toDouble()
        }
        val chart: XYChart = XYChartBuilder()
            .width(1200)
            .height(1200)
            .title("Comparing performance of models. One with positive-only initial parameters (yellow), one with negative and positive initial parameters.")
            .xAxisTitle("Training epochs")
            .yAxisTitle("Models' mean absolute error loss")
            .build()
        chart.addSeries("He Initialization", xData, firstHistory["val-loss"])
        chart.addSeries("Positive-only He Initialization", xData, secondHistory["val-loss"])
        BitmapEncoder.saveBitmap(chart, "target/performance", BitmapEncoder.BitmapFormat.PNG)
    }
}

fun main() {
    val trainer = HousePriceInitializerComparer()
    trainer.train()
}