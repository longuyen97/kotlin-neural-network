package de.longuyen

import org.nd4j.linalg.factory.Nd4j

fun main(){
    val input = Nd4j.rand(*longArrayOf(5, 1))
    val parameters1 = Nd4j.rand(*longArrayOf(2, 5))
    println(parameters1)
    println(input)
    val output1 = parameters1.mmul(input)
}