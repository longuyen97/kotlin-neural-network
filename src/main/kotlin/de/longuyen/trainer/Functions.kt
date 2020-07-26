package de.longuyen.trainer

import org.nd4j.evaluation.meta.Prediction
import org.nd4j.linalg.activations.impl.ActivationReLU
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

fun relu(x: INDArray): INDArray {
    return ActivationReLU().getActivation(x, false)
}

fun mae(yTrue: INDArray, yPrediction: INDArray) : INDArray {
    val diff = yPrediction.sub(yTrue)
    val absolute = Transforms.abs(diff)
    return absolute.sum(true, 0).mean()
}






