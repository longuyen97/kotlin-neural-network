package de.longuyen

import de.longuyen.trainer.mae
import de.longuyen.trainer.relu
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun main() {
    val layers = longArrayOf(10, 5, 2, 1)

    val parameters = mutableMapOf<String, INDArray>()
    val biases = mutableMapOf<String, INDArray>()
    for (i in 1 until layers.size){
        // i - 1 rows, i columns
        parameters["W$i"] = Nd4j.rand(*longArrayOf(layers[i], layers[i - 1]))
        biases["b$i"] = Nd4j.zeros(*longArrayOf(layers[i], 1))
    }
    val logits = mutableMapOf<String, INDArray>()
    val activations = mutableMapOf<String, INDArray>()

    val features = Nd4j.rand(*longArrayOf(10, 2))
    val target = Nd4j.rand(*longArrayOf(1, 2))
    logits["Z1"] = (parameters["W1"]!!.mmul(features))
    activations["A1"] = relu(logits["Z1"]!!)

    for(i in 2 until layers.size){
        logits["Z$i"] = (parameters["W$i"]!!.mmul(activations["A${i - 1}"])).add(biases["b$i"]!!)
        activations["A$i"] = relu(logits["Z$i"]!!)
    }

    val activationGrads = mutableMapOf<String, INDArray>()
    val logitGrads = mutableMapOf<String, INDArray>()

    println(mae(activations["A${layers.size - 1}"]!!, target))
}