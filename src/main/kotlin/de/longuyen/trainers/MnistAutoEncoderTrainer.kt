package de.longuyen.trainers

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.mnist.MnistDataGenerator
import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.Softmax
import de.longuyen.neuronalnetwork.initializers.HeInitializer
import de.longuyen.neuronalnetwork.initializers.XavierInitializer
import de.longuyen.neuronalnetwork.losses.MAE
import de.longuyen.neuronalnetwork.optimizers.Adam
import de.longuyen.neuronalnetwork.optimizers.MomentumGradientDescent
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.Serializable

class MnistAutoEncoderTrainer : Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    private val neuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(
            intArrayOf(784, 512, 256, 512, 784),
            HeInitializer(),
            LeakyRelu(),
            Softmax(),
            MAE(),
            MomentumGradientDescent(learningRate = 0.001),
            de.longuyen.neuronalnetwork.metrics.MAE()
        )

    fun train() {
        val housePriceData: SupervisedDataGenerator = MnistDataGenerator()
        val trainingData = housePriceData.getTrainingData()
        val testingData = housePriceData.getTestingDataWithLabels()
        val X = trainingData.first
        val x = testingData.first

        val history = neuronalNetwork.train(X, X, x, x, epochs = 50, batchSize = 64)
        val xData = DoubleArray(history["val-loss"]!!.size)
        for (i in 0 until history["val-loss"]!!.size) {
            xData[i] = i.toDouble()
        }

        val chart: XYChart = XYChartBuilder()
            .width(1200)
            .height(1200)
            .title("Mnist's performance")
            .xAxisTitle("Training epochs")
            .yAxisTitle("Model's loss and metric")
            .build()
        chart.addSeries("Training loss", xData, history["train-loss"])
        chart.addSeries("Validating loss", xData, history["val-loss"])
        chart.addSeries("Training metric", xData, history["train-metric"])
        chart.addSeries("Validating metric", xData, history["val-metric"])
        BitmapEncoder.saveBitmap(chart, "target/performance", BitmapEncoder.BitmapFormat.PNG)
        FileOutputStream("target/neuronalnetwork.ser").use { fos ->
            ObjectOutputStream(fos).use { oos ->
                oos.writeObject(neuronalNetwork)
            }
        }
    }
}

fun main() {
    val trainer = MnistAutoEncoderTrainer()
    trainer.train()
}