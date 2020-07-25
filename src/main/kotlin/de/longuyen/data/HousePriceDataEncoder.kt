package de.longuyen.data

class HousePriceDataEncoder(dataFrame: Map<Int, MutableList<String>>) : DataEncoder(dataFrame) {
    override fun encodeDiscreteAttributes() : Map<Int, DiscreteAttributesCode>{
        TODO("Not yet implemented")
    }

    override fun normalizeContinuousAttributes() : Map<Int, ContinuousAttributesCode>{
        TODO("Not yet implemented")
    }

    override fun produce(): Map<Int, MutableList<String>> {
        TODO("Not yet implemented")
    }

}