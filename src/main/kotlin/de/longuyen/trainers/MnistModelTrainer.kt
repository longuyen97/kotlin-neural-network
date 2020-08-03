package de.longuyen.trainers

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.mnist.MnistDataGenerator
import de.longuyen.neuronalnetwork.DeepNeuronalNetwork
import de.longuyen.neuronalnetwork.activations.Relu
import de.longuyen.neuronalnetwork.activations.Softmax
import de.longuyen.neuronalnetwork.initializers.HeInitializer
import de.longuyen.neuronalnetwork.losses.CrossEntropy
import de.longuyen.neuronalnetwork.metrics.Accuracy
import de.longuyen.neuronalnetwork.optimizers.GradientDescent
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.Serializable


class MnistModelTrainer :
    Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    private val deepNeuronalNetwork: DeepNeuronalNetwork =
        DeepNeuronalNetwork(
            intArrayOf(784, 32, 16, 10),
            HeInitializer(),
            Relu(),
            Softmax(),
            CrossEntropy(),
            GradientDescent(learningRate = 0.025),
            Accuracy()
        )

    fun train() {
        val housePriceData: SupervisedDataGenerator = MnistDataGenerator()
        val trainingData = housePriceData.getTrainingData()
        val testingData = housePriceData.getTestingDataWithLabels()
        val X = trainingData.first
        val Y = trainingData.second
        val x = testingData.first
        val y = testingData.second

        val history = deepNeuronalNetwork.train(X, Y, x, y, epochs = 150, batchSize = 32)
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
                oos.writeObject(deepNeuronalNetwork)
            }
        }
    }
}

fun main() {
    val trainer = MnistModelTrainer()
    trainer.train()
}
