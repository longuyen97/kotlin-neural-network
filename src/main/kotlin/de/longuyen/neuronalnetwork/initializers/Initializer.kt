package de.longuyen.neuronalnetwork.initializers

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Initializer {
    abstract fun initialize(layers: IntArray) :  MutableMap<String, INDArray>

    override fun toString(): String {
        return "Initializer {${this.javaClass.simpleName}}"
    }
}