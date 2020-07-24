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
    LotConfig(10, DataType.CLASSIFICATION),
    LandSlope(11, DataType.CLASSIFICATION),
    Neighborhood(12, DataType.CLASSIFICATION),
    Condition1(13, DataType.CLASSIFICATION),
    Condition2(14, DataType.CLASSIFICATION),
    BldgType(15, DataType.CLASSIFICATION),
    HouseStyle(16, DataType.CLASSIFICATION),
    OverallQual(17, DataType.REGRESSION),
    OverallCond(18, DataType.REGRESSION),
    YearBuilt(19, DataType.REGRESSION),
    YearRemodAdd (20, DataType.REGRESSION),
    RoofStyle(21, DataType.CLASSIFICATION),
    RoofMatl(22, DataType.CLASSIFICATION),
    Exterior1st(23, DataType.CLASSIFICATION),
    Exterior2nd(24, DataType.CLASSIFICATION),
    MasVnrType(25, DataType.CLASSIFICATION),
    MasVnrArea(26, DataType.REGRESSION),
    ExterQual(27, DataType.CLASSIFICATION),
    ExterCond (28, DataType.CLASSIFICATION),
    Foundation(29, DataType.CLASSIFICATION),
    BsmtQual(30, DataType.CLASSIFICATION),
    BsmtCond(31, DataType.CLASSIFICATION),
    BsmtExposure(32, DataType.CLASSIFICATION),
    BsmtFinType1(33, DataType.CLASSIFICATION),
    BsmtFinSF1(34, DataType.REGRESSION),
    BsmtFinType2(35, DataType.CLASSIFICATION),
    BsmtFinSF2(36, DataType.REGRESSION),
    BsmtUnfSF(37, DataType.REGRESSION),
    TotalBsmtSF(38, DataType.REGRESSION),
    Heating(39, DataType.CLASSIFICATION),
    HeatingQC(40, DataType.CLASSIFICATION),
    CentralAir(41, DataType.CLASSIFICATION),
    Electrical(42, DataType.CLASSIFICATION),
    T1stFlrSF(43, DataType.REGRESSION),
    T2ndFlrSF(44, DataType.REGRESSION),
    LowQualFinSF(45, DataType.REGRESSION),
    GrLivArea(46, DataType.REGRESSION),
    BsmtFullBath(47, DataType.REGRESSION),
    BsmtHalfBath(48, DataType.REGRESSION),
    FullBath(49, DataType.REGRESSION),
    HalfBath(50, DataType.REGRESSION),
    BedroomAbvGr(51, DataType.REGRESSION),
    KitchenAbvGr(52, DataType.REGRESSION),
    KitchenQual(53, DataType.CLASSIFICATION),
    TotRmsAbvGrd(54, DataType.REGRESSION),
    Functional(55, DataType.CLASSIFICATION),
    Fireplaces(56, DataType.REGRESSION),
    FireplaceQu(57, DataType.CLASSIFICATION),
    GarageType(58, DataType.CLASSIFICATION),
    GarageYrBlt(59, DataType.REGRESSION),
    GarageFinish(60, DataType.CLASSIFICATION),
    GarageCars(61, DataType.REGRESSION),
    GarageArea(62, DataType.REGRESSION),
    GarageQual(63, DataType.CLASSIFICATION),
    GarageCond(64, DataType.CLASSIFICATION),
    PavedDrive(65, DataType.CLASSIFICATION),
    WoodDeckSF(66, DataType.REGRESSION),
    OpenPorchSF(67, DataType.REGRESSION),
    EnclosedPorch(68, DataType.REGRESSION),
    T3SsnPorch(69, DataType.REGRESSION),
    ScreenPorch(70, DataType.REGRESSION),
    PoolArea(71, DataType.REGRESSION),
    PoolQC(72, DataType.CLASSIFICATION),
    Fence(73, DataType.CLASSIFICATION),
    MiscFeature(74, DataType.CLASSIFICATION),
    MiscValMoSold(75, DataType.REGRESSION),
    MoSold(76, DataType.REGRESSION),
    YrSold(77, DataType.REGRESSION),
    SaleType(77, DataType.CLASSIFICATION),
    SaleCondition(78, DataType.CLASSIFICATION),
    SalePrice(79, DataType.REGRESSION)
}


interface DataParser {
    fun parse(csvReader: CSVParser): INDArray
}

class HousePriceDataParser : DataParser{
    override fun parse(csvReader: CSVParser): INDArray {
        TODO("Not yet implemented")
    }
}
