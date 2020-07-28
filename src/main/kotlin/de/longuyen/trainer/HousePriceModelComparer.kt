package de.longuyen.trainer

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.houseprice.HousePriceDataGenerator
import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.NoActivation
import de.longuyen.neuronalnetwork.initializers.ChainInitializer
import de.longuyen.neuronalnetwork.losses.MAE
import de.longuyen.neuronalnetwork.optimizers.GradientDescent
import de.longuyen.neuronalnetwork.optimizers.MomentumGradientDescent
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.Serializable

class HousePriceModelComparer(layers: IntArray, learningRate: Double, private val epochs: Long) :
    Serializable {
    private val firstNeuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(layers, ChainInitializer(), LeakyRelu(), NoActivation(), MAE(), GradientDescent(learningRate))
    private val secondNeuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(layers, ChainInitializer(), LeakyRelu(), NoActivation(), MAE(), MomentumGradientDescent(learningRate))

    fun train() {
        val housePriceData: SupervisedDataGenerator = HousePriceDataGenerator()
        val trainingData = housePriceData.getTrainingData()

        val x = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(1000, 1460))
        val y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1000, 1460))
        val X = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(0, 1000))
        val Y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 1000))

        val firstLosses = firstNeuronalNetwork.train(X, Y, x, y, epochs)
        val secondLosses = secondNeuronalNetwork.train(X, Y, x, y, epochs)

        val firstYTrain = firstLosses.first.toDoubleArray()
        val firstYTest = firstLosses.second.toDoubleArray()
        val secondYTrain = secondLosses.first.toDoubleArray()
        val secondYTest = secondLosses.second.toDoubleArray()

        val xData = DoubleArray(firstLosses.first.size)

        for (i in 0 until firstLosses.first.size) {
            xData[i] = i.toDouble()
        }
        val chart: XYChart = XYChartBuilder()
            .width(1200)
            .height(1200)
            .title("Two-layer-network's performance")
            .xAxisTitle("Epochs")
            .yAxisTitle("Model's loss")
            .build()
        chart.addSeries("First model Training", xData, firstYTrain)
        chart.addSeries("First model Validating", xData, firstYTest)
        chart.addSeries("Second model Training", xData, secondYTrain)
        chart.addSeries("Second model Validating", xData, secondYTest)
        BitmapEncoder.saveBitmap(chart, "target/performance", BitmapEncoder.BitmapFormat.PNG)
    }
}

fun main() {
    val trainer = HousePriceModelComparer(intArrayOf(318, 32, 1), 0.001, 2000)
    trainer.train()
}