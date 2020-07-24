package de.longuyen.data

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TestHousePriceDataAttributes {
    @Test
    fun `Test House Price Data attributes`(){
        val indexMap = HashMap<Int, HousePriceDataAttributes>()
        for(i in HousePriceDataAttributes.values()){
            indexMap[i.index] = i
        }
        for(i in HousePriceDataAttributes.values().indices){
            assertTrue(indexMap.containsKey(i) , "$i")
        }

        val targetMap = HashMap<Boolean, MutableList<HousePriceDataAttributes>>()
        for(i in HousePriceDataAttributes.values()){
            if(!targetMap.containsKey(i.target)){
                targetMap[i.target] = mutableListOf()
            }
            targetMap[i.target]!!.add(i)
        }
        assertTrue(targetMap[true]?.size == 1, targetMap[true]?.size.toString())
        assertEquals(targetMap[false]?.size, HousePriceDataAttributes.values().size - 1, targetMap[false]?.size.toString())
    }
}