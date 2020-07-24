package de.longuyen

import org.nd4j.linalg.factory.Nd4j


fun main() {

    val x = Nd4j.zeros(3, 4)
    println(x)

    val ranks = x.rank()
    println(ranks)

    val shape = x.shape()
    shape.forEach { print("$it ") }
    println()

    val length = x.length()
    println(length)

    val dt = x.dataType()
    println(dt)
}

