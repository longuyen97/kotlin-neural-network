package de.longuyen.trainer

import de.longuyen.data.DataEncoder
import de.longuyen.data.DataGenerator
import de.longuyen.data.HousePriceDataEncoder
import de.longuyen.data.HousePriceDataGenerator
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.Serializable

class HousePriceModelTrainer(private val layers: IntArray, private val learningRate: Double, private val epochs: Long) : Serializable {
    private val dataGenerator: DataGenerator = HousePriceDataGenerator()
    private val dataEncoder: DataEncoder
    private val parameters = mutableMapOf<String, INDArray>()
    private val biases = mutableMapOf<String, INDArray>()
    private val hiddens = layers.size - 1

    init {
        dataEncoder = HousePriceDataEncoder(dataGenerator.getTrainingData())
        for (i in 1 until layers.size) {
            // i - 1 rows, i columns
            parameters["W$i"] = Nd4j.rand(*intArrayOf(layers[i], layers[i - 1])).castTo(DataType.DOUBLE)
            biases["b$i"] = Nd4j.zeros(*intArrayOf(layers[i], 1)).castTo(DataType.DOUBLE)
        }
    }

    fun train(): Pair<MutableList<Double>, MutableList<Double>> {
        val dataGenerator = HousePriceDataGenerator()
        val trainingData = dataGenerator.getTrainingData()
        val dataEncoder = HousePriceDataEncoder(trainingData)
        val encodedTrainingData = dataEncoder.encode()

        val X = encodedTrainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(0, 1000))
        val Y = encodedTrainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 1000))
        val x = encodedTrainingData.first.get(NDArrayIndex.interval(0, 318), NDArrayIndex.interval(1000, 1460))
        val y = encodedTrainingData.second.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1000, 1460))

        val validationLosses = mutableListOf<Double>()
        val trainingLosses = mutableListOf<Double>()
        for (epoch in 0 until epochs) {
            val cache = forward(X)
            backward(X, Y, cache)
            val yPredition = inference(x)
            val trainingLoss = mae(Y, cache.second["A$hiddens"]!!)
            val validationLoss = mae(y, yPredition)
            println("Epoch $epoch - Training loss ${(trainingLoss.element() as Double).toInt()} - Validation Loss ${(validationLoss.element() as Double).toInt()}")

            validationLosses.add(validationLoss.element() as Double)
            trainingLosses.add(trainingLoss.element() as Double)
        }

        return Pair(trainingLosses, validationLosses)
    }

    fun inference(x: INDArray): INDArray {
        return forward(x).second["A$hiddens"]!!
    }

    private fun forward(x: INDArray): Pair<MutableMap<String, INDArray>, MutableMap<String, INDArray>> {
        val logits = mutableMapOf<String, INDArray>()
        val activations = mutableMapOf<String, INDArray>()

        logits["Z1"] = (parameters["W1"]!!.mmul(x)).add(biases["b1"]!!)
        activations["A1"] = relu(logits["Z1"]!!)

        for (i in 2 until layers.size - 1) {
            logits["Z$i"] = (parameters["W$i"]!!.mmul(activations["A${i - 1}"])).add(biases["b$i"]!!)
            activations["A$i"] = relu(logits["Z$i"]!!)
        }
        logits["Z${layers.size - 1}"] =
            (parameters["W${layers.size - 1}"]!!.mmul(activations["A${layers.size - 2}"])).add(biases["b${layers.size - 1}"]!!)
        activations["A${layers.size - 1}"] = logits["Z${layers.size - 1}"]!!.dup()

        return Pair(logits, activations)
    }

    private fun backward(
        x: INDArray,
        y: INDArray,
        cache: Pair<MutableMap<String, INDArray>, MutableMap<String, INDArray>>
    ) {
        val logits = cache.first
        val activations = cache.second
        val activationGrads = mutableMapOf<String, INDArray>()
        val logitGrads = mutableMapOf<String, INDArray>()
        val parameterGrads = mutableMapOf<String, INDArray>()
        val biasGrads = mutableMapOf<String, INDArray>()

        logitGrads["dZ$hiddens"] = dMae(y, activations["A$hiddens"]!!)

        for (i in 1 until hiddens) {
            val activationGrad =
                (parameters["W${hiddens - i + 1}"]!!.transpose()).mmul(logitGrads["dZ${hiddens - i + 1}"]!!)
            activationGrads["dA${hiddens - i}"] = activationGrad


            val logitGrad = activationGrads["dA${hiddens - i}"]!!.mul(dRelu(logits["Z${hiddens - i}"]!!))
            logitGrads["dZ${hiddens - i}"] = logitGrad
        }

        parameterGrads["dW1"] = logitGrads["dZ1"]!!.mmul(x.transpose())
        biasGrads["db1"] = logitGrads["dZ1"]!!.sum(true, 1)
        for (i in 2 until layers.size) {
            parameterGrads["dW$i"] = logitGrads["dZ$i"]!!.mmul(activations["A${i - 1}"]!!.transpose())
            biasGrads["db$i"] = logitGrads["dZ$i"]!!.sum(true, 1)
        }

        for (i in 2 until layers.size) {
            parameters["W$i"] = parameters["W$i"]!!.sub(parameterGrads["dW$i"]!!.mul(learningRate))
            biases["b$i"] = biases["b$i"]!!.sub(biasGrads["db$i"]!!.mul(learningRate))
        }
    }
}