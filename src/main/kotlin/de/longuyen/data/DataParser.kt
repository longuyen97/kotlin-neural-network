package de.longuyen.data

import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.api.ndarray.INDArray

enum class DataType {
    REGRESSION,
    CLASSIFICATION
}

enum class DataColumn(index: Int, dataType: DataType) {
    Id(0, DataType.REGRESSION),
    MSSubClass(1, DataType.CLASSIFICATION),
    MSZoning(2, DataType.CLASSIFICATION),
    LotFrontage(3, DataType.REGRESSION),
    LotArea(4, DataType.REGRESSION),
    Street(5, DataType.CLASSIFICATION),
    Alley(6, DataType.CLASSIFICATION),
    LotShape(7, DataType.CLASSIFICATION),
    LandContour(8, DataType.CLASSIFICATION),
    Utilities(9, DataType.CLASSIFICATION),
    const val LotConfig = 10
    const val LandSlope = 11
    const val Neighborhood = 12
    const val Condition1 = 13
    const val Condition2 = 14
    const val BldgType = 15
    const val HouseStyle = 16
    const val OverallQual = 17
    const val OverallCond = 18

    /**
     * Original construction date
     */
    const val YearBuilt = 19
    const val YearRemodAdd = 20
    const val RoofStyle = 21
    const val RoofMatl = 22
    const val Exterior1st = 23
    const val Exterior2nd = 24
    const val MasVnrType = 25
    const val MasVnrArea = 26
    const val ExterQual = 27
    const val ExterCond = 28
    const val Foundation = 29
    const val BsmtQual = 30
    const val BsmtCond = 31
    const val BsmtExposure = 32
    const val BsmtFinType1 = 33
    const val BsmtFinSF1 = 34
    const val BsmtFinType2 = 35
    const val BsmtFinSF2 = 36
    const val BsmtUnfSF = 37
    const val TotalBsmtSF = 38
    const val Heating = 39
    const val HeatingQC = 40
    const val CentralAir = 41
    const val Electrical = 42

    /**
     * First Floor square feet
     */
    const val t1stFlrSF = 43

    /**
     * Second floor square feet
     */
    const val t2ndFlrSF = 44
    const val LowQualFinSF = 45
    const val GrLivArea = 46
    const val BsmtFullBath = 47
    const val BsmtHalfBath = 48
    const val FullBath = 49
    const val HalfBath = 50
    const val BedroomAbvGr = 51
    const val KitchenAbvGr = 52
    const val KitchenQual = 53
    const val TotRmsAbvGrd = 54
    const val Functional = 55
    const val Fireplaces = 56
    const val FireplaceQu = 57
    const val GarageType = 58

    /**
     * Year garage was built
     */
    const val GarageYrBlt = 59
    const val GarageFinish = 60
    const val GarageCars = 61
    const val GarageArea = 62
    const val GarageQual = 63
    const val GarageCond = 64
    const val PavedDrive = 65
    const val WoodDeckSF = 66
    const val OpenPorchSF = 67
    const val EnclosedPorch = 68
    const val t3SsnPorch = 69
    const val ScreenPorch = 70
    const val PoolArea = 71
    const val PoolQC = 72
    const val Fence = 73
    const val MiscFeature = 74

    /**
     * Month Sold
     */
    const val MiscValMoSold = 75

    /**
     * Year Sold
     */
    const val YrSold = 76
    const val SaleType = 77
    const val SaleCondition = 78
}

const val Id = 0
const val MSSubClass = 1
const val MSZoning = 2
const val LotFrontage = 3

/**
 *  Lot size in square feet
 */
const val LotArea = 4
const val Street = 5
const val Alley = 6
const val LotShape = 7
const val LandContour = 8
const val Utilities = 9
const val LotConfig = 10
const val LandSlope = 11
const val Neighborhood = 12
const val Condition1 = 13
const val Condition2 = 14
const val BldgType = 15
const val HouseStyle = 16
const val OverallQual = 17
const val OverallCond = 18

/**
 * Original construction date
 */
const val YearBuilt = 19
const val YearRemodAdd = 20
const val RoofStyle = 21
const val RoofMatl = 22
const val Exterior1st = 23
const val Exterior2nd = 24
const val MasVnrType = 25
const val MasVnrArea = 26
const val ExterQual = 27
const val ExterCond = 28
const val Foundation = 29
const val BsmtQual = 30
const val BsmtCond = 31
const val BsmtExposure = 32
const val BsmtFinType1 = 33
const val BsmtFinSF1 = 34
const val BsmtFinType2 = 35
const val BsmtFinSF2 = 36
const val BsmtUnfSF = 37
const val TotalBsmtSF = 38
const val Heating = 39
const val HeatingQC = 40
const val CentralAir = 41
const val Electrical = 42

/**
 * First Floor square feet
 */
const val t1stFlrSF = 43

/**
 * Second floor square feet
 */
const val t2ndFlrSF = 44
const val LowQualFinSF = 45
const val GrLivArea = 46
const val BsmtFullBath = 47
const val BsmtHalfBath = 48
const val FullBath = 49
const val HalfBath = 50
const val BedroomAbvGr = 51
const val KitchenAbvGr = 52
const val KitchenQual = 53
const val TotRmsAbvGrd = 54
const val Functional = 55
const val Fireplaces = 56
const val FireplaceQu = 57
const val GarageType = 58

/**
 * Year garage was built
 */
const val GarageYrBlt = 59
const val GarageFinish = 60
const val GarageCars = 61
const val GarageArea = 62
const val GarageQual = 63
const val GarageCond = 64
const val PavedDrive = 65
const val WoodDeckSF = 66
const val OpenPorchSF = 67
const val EnclosedPorch = 68
const val t3SsnPorch = 69
const val ScreenPorch = 70
const val PoolArea = 71
const val PoolQC = 72
const val Fence = 73
const val MiscFeature = 74

/**
 * Month Sold
 */
const val MiscValMoSold = 75

/**
 * Year Sold
 */
const val YrSold = 76
const val SaleType = 77
const val SaleCondition = 78

/**
 * the property's sale price in dollars. This is the target variable that you're trying to predict.
 *
 */
const val SalePrice = 79


interface DataParser {
    fun parse(csvReader: CSVParser): INDArray
}

class HousePriceDataParser : DataParser{
    override fun parse(csvReader: CSVParser): INDArray {
        TODO("Not yet implemented")
    }
}
