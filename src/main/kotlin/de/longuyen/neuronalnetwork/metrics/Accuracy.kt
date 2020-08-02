package de.longuyen.neuronalnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

class Accuracy : Metric(), Serializable {
    override fun compute(yTrue: INDArray, yPrediction: INDArray): Double {
        val yTrueArgMax = Nd4j.argMax(yTrue, 0).toDoubleVector()
        val yPredictionArgMax = Nd4j.argMax(yPrediction, 0).toDoubleVector()
        var ret = 0.0
        for (y in yTrueArgMax.indices) {
            if (yTrueArgMax[y] == yPredictionArgMax[y]) {
                ret += 1.0
            }
        }
        return ret / yTrueArgMax.size
    }

}