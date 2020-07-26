package de.longuyen.neuronalnetwork

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

interface Activation{
    fun forward(x: INDArray) : INDArray

    fun backward(x: INDArray): INDArray
}

class Relu : Activation {
    override fun forward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for(yi in input.indices){
            for(xi in input[yi].indices){
                if(input[yi][xi] < 0){
                    input[yi][xi] = 0.0
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }

    override fun backward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for(yi in input.indices){
            for(xi in input[yi].indices){
                if(input[yi][xi] > 0){
                    input[yi][xi] = 1.0
                }else{
                    input[yi][xi] = 0.0
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }
}

class LeakyRelu(private val alpha: Double) : Activation {
    override fun forward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for(yi in input.indices){
            for(xi in input[yi].indices){
                if(input[yi][xi] < 0){
                    input[yi][xi] *= alpha
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }

    override fun backward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for(yi in input.indices){
            for(xi in input[yi].indices){
                if(input[yi][xi] > 0){
                    input[yi][xi] = 1.0
                }else{
                    input[yi][xi] = alpha
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }
}

class NoActivation : Activation {
    override fun forward(x: INDArray): INDArray {
        return x.dup()
    }

    override fun backward(x: INDArray): INDArray {
        return Nd4j.ones(*x.shape())
    }
}