package de.longuyen.trainer

import de.longuyen.data.DataEncoder
import de.longuyen.data.DataGenerator
import de.longuyen.data.HousePriceDataEncoder
import de.longuyen.data.HousePriceDataGenerator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class HousePriceModelTrainer(private val layers: IntArray, private val learningRate: Double, private val epochs: Long) {
    private val dataGenerator: DataGenerator = HousePriceDataGenerator()
    private val dataEncoder: DataEncoder
    private val parameters = mutableMapOf<String, INDArray>()
    private val biases = mutableMapOf<String, INDArray>()
    private val hiddens = layers.size - 1

    init {
        dataEncoder = HousePriceDataEncoder(dataGenerator.getTrainingData())
        for (i in 1 until layers.size) {
            // i - 1 rows, i columns
            parameters["W$i"] = Nd4j.rand(*intArrayOf(layers[i], layers[i - 1]))
            biases["b$i"] = Nd4j.zeros(*intArrayOf(layers[i], 1))
        }
    }

    fun train() {
        val dataGenerator = HousePriceDataGenerator()
        val trainingData = dataGenerator.getTrainingData()
        val testingData = dataGenerator.getTestingData()
        val dataEncoder = HousePriceDataEncoder(trainingData)

    }

    fun forward(x: INDArray) : Pair<MutableMap<String, INDArray>, MutableMap<String, INDArray>>{
        val logits = mutableMapOf<String, INDArray>()
        val activations = mutableMapOf<String, INDArray>()

        logits["Z1"] = (parameters["W1"]!!.mmul(x)).add(biases["b1"]!!)
        activations["A1"] = relu(logits["Z1"]!!)

        for (i in 2 until layers.size - 1) {
            logits["Z$i"] = (parameters["W$i"]!!.mmul(activations["A${i - 1}"])).add(biases["b$i"]!!)
            activations["A$i"] = relu(logits["Z$i"]!!)
        }
        logits["Z${layers.size - 1}"] = (parameters["W${layers.size - 1}"]!!.mmul(activations["A${layers.size - 2}"])).add(biases["b${layers.size - 1}"]!!)
        activations["A${layers.size - 1}"] = logits["Z${layers.size - 1}"]!!.dup()

        return Pair(logits, activations)
    }

    fun backward(x: INDArray, y: INDArray, cache: Pair<MutableMap<String, INDArray>, MutableMap<String, INDArray>>) {
        val logits = cache.first
        val activations = cache.second
        val activationGrads = mutableMapOf<String, INDArray>()
        val logitGrads = mutableMapOf<String, INDArray>()
        val parameterGrads = mutableMapOf<String, INDArray>()
        val biasGrads = mutableMapOf<String, INDArray>()

        logitGrads["dZ$hiddens"] = dMae(y, activations["A$hiddens"]!!)
        val loss = mae(y, activations["A$hiddens"]!!)

        for (i in 1 until hiddens) {
            val activationGrad =
                (parameters["W${hiddens - i + 1}"]!!.transpose()).mmul(logitGrads["dZ${hiddens - i + 1}"]!!)
            activationGrads["dA${hiddens - i}"] = activationGrad


            val logitGrad = activationGrads["dA${hiddens - i}"]!!.mul(dRelu(logits["Z${hiddens - i}"]!!))
            logitGrads["dZ${hiddens - i}"] = logitGrad
        }

        parameterGrads["dW1"] = logitGrads["dZ1"]!!.mmul(x.transpose())
        biasGrads["db1"] = logitGrads["dZ1"]!!.sum(true, 1)
        for(i in 2 until layers.size){
            parameterGrads["dW$i"] = logitGrads["dZ$i"]!!.mmul(activations["A${i - 1}"]!!.transpose())
            biasGrads["db$i"] = logitGrads["dZ$i"]!!.sum(true, 1)
        }

        val learningRate = 0.001
        for(i in 2 until layers.size) {
            parameters["W$i"] = parameters["W$i"]!!.sub(parameterGrads["dW$i"]!!.mul(learningRate))
            biases["b$i"] = biases["b$i"]!!.sub(biasGrads["db$i"]!!.mul(learningRate))
        }
    }

    fun getTrainingData() : Pair<INDArray, INDArray>{
        return dataEncoder.encode()
    }

    fun getTestingData() : INDArray {
        return dataEncoder.encodeFutureData(dataGenerator.getTestingData())
    }

    fun getValidatingData() : INDArray {
        return dataEncoder.encodeFutureData(dataGenerator.getValidatingData())
    }
}