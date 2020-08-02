package de.longuyen.trainer

import de.longuyen.data.SupervisedDataGenerator
import de.longuyen.data.mnist.MnistDataGenerator
import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.Relu
import de.longuyen.neuronalnetwork.activations.Softmax
import de.longuyen.neuronalnetwork.initializers.ChainInitializer
import de.longuyen.neuronalnetwork.losses.CrossEntropy
import de.longuyen.neuronalnetwork.metrics.Accuracy
import de.longuyen.neuronalnetwork.optimizers.Adagrad
import de.longuyen.neuronalnetwork.optimizers.Adam
import de.longuyen.neuronalnetwork.optimizers.GradientDescent
import de.longuyen.neuronalnetwork.optimizers.MomentumGradientDescent
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.Serializable


class MnistModelTrainer :
    Serializable {
    private val neuronalNetwork: NeuronalNetwork =
        NeuronalNetwork(
            intArrayOf(784, 64, 10),
            ChainInitializer(),
            Relu(),
            Softmax(),
            CrossEntropy(),
            Adagrad(learningRate = 0.0001),
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

        val losses = neuronalNetwork.train(X, Y, x, y, epochs = 200)
        val xData = DoubleArray(losses.first.size)
        for (i in 0 until losses.first.size) {
            xData[i] = i.toDouble()
        }
        val yTrain = losses.first.toDoubleArray()
        val yTest = losses.second.toDoubleArray()

        val chart: XYChart = XYChartBuilder()
            .width(1200)
            .height(1200)
            .title("Mnist's performance")
            .xAxisTitle("Training epochs")
            .yAxisTitle("Model's cross entropy loss")
            .build()
        chart.addSeries("Training", xData, yTrain)
        chart.addSeries("Validating", xData, yTest)
        BitmapEncoder.saveBitmap(chart, "target/performance", BitmapEncoder.BitmapFormat.PNG)
        FileOutputStream("target/neuronalnetwork.ser").use { fos ->
            ObjectOutputStream(fos).use { oos ->
                oos.writeObject(neuronalNetwork)
            }
        }
    }
}

fun main() {
    val trainer = MnistModelTrainer()
    trainer.train()
}
