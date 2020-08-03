package de.longuyen.trainer

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.houseprice.HousePriceDataGenerator
import de.longuyen.neuronalnetwork.DeepNeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.NoActivation
import de.longuyen.neuronalnetwork.initializers.HeInitializer
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
class HousePriceModelTrainer(layers: IntArray= intArrayOf(318, 64, 32, 1), learningRate: Double=0.001, private val epochs: Long=2000) :
    Serializable {
    private val deepNeuronalNetwork: DeepNeuronalNetwork =
        DeepNeuronalNetwork(
            layers,
            HeInitializer(),
            LeakyRelu(),
            NoActivation(),
            MAE(),
            MomentumGradientDescent(learningRate),
            de.longuyen.neuronalnetwork.metrics.MAE()
        )

    fun train() {
        val housePriceData: SupervisedDataGenerator = HousePriceDataGenerator()
        val trainingData = housePriceData.getTrainingData()

        val X = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(0, 1000))
        val Y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 1000))
        val x = trainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(1000, 1460))
        val y = trainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1000, 1460))

        val history = deepNeuronalNetwork.train(X, Y, x, y, epochs)
        val xData = DoubleArray(history["val-loss"]!!.size)
        for (i in 0 until history["val-loss"]!!.size) {
            xData[i] = i.toDouble()
        }

        val chart: XYChart = XYChartBuilder()
            .width(1200)
            .height(1200)
            .title("Two-layer-network's performance")
            .xAxisTitle("Training epochs")
            .yAxisTitle("Model's loss and metric")
            .build()
        chart.addSeries("Training Loss", xData, history["train-loss"])
        chart.addSeries("Validating Loss", xData, history["val-loss"])
        chart.addSeries("Training Metric", xData, history["train-metric"])
        chart.addSeries("Validating Metric", xData, history["val-metric"])
        BitmapEncoder.saveBitmap(chart, "target/performance", BitmapEncoder.BitmapFormat.PNG)
    }
}

fun main() {
    val trainer = HousePriceModelTrainer()
    trainer.train()
}