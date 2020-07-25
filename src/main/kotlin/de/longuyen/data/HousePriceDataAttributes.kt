package de.longuyen.data

/**
 * Classify each attribute of the data type into two types:
 * DISCRETE corresponds: Nominal, Ordinal
 * CONTINUOUS corresponds: Interval, Ratio
 */
enum class DataType {
    CONTINUOUS,
    DISCRETE
}

/**
 * Concrete attributes' names of the dataset.
 * @param index: Indicates at which position in the CSV the attribute will appear
 * @param dataType: Level of measurement of the data type. Continuous data will be normalized
 * @param target: Indicates if the attribute is the target of the data set
 */
enum class HousePriceDataAttributes(val index: Int, val dataType: DataType, val target: Boolean = false) {
    Id(0, DataType.CONTINUOUS),
    MSSubClass(1, DataType.DISCRETE),
    MSZoning(2, DataType.DISCRETE),
    LotFrontage(3, DataType.CONTINUOUS),
    LotArea(4, DataType.CONTINUOUS),
    Street(5, DataType.DISCRETE),
    Alley(6, DataType.DISCRETE),
    LotShape(7, DataType.DISCRETE),
    LandContour(8, DataType.DISCRETE),
    Utilities(9, DataType.DISCRETE),
    LotConfig(10, DataType.DISCRETE),
    LandSlope(11, DataType.DISCRETE),
    Neighborhood(12, DataType.DISCRETE),
    Condition1(13, DataType.DISCRETE),
    Condition2(14, DataType.DISCRETE),
    BldgType(15, DataType.DISCRETE),
    HouseStyle(16, DataType.DISCRETE),
    OverallQual(17, DataType.CONTINUOUS),
    OverallCond(18, DataType.CONTINUOUS),
    YearBuilt(19, DataType.CONTINUOUS),
    YearRemodAdd (20, DataType.CONTINUOUS),
    RoofStyle(21, DataType.DISCRETE),
    RoofMatl(22, DataType.DISCRETE),
    Exterior1st(23, DataType.DISCRETE),
    Exterior2nd(24, DataType.DISCRETE),
    MasVnrType(25, DataType.DISCRETE),
    MasVnrArea(26, DataType.CONTINUOUS),
    ExterQual(27, DataType.DISCRETE),
    ExterCond (28, DataType.DISCRETE),
    Foundation(29, DataType.DISCRETE),
    BsmtQual(30, DataType.DISCRETE),
    BsmtCond(31, DataType.DISCRETE),
    BsmtExposure(32, DataType.DISCRETE),
    BsmtFinType1(33, DataType.DISCRETE),
    BsmtFinSF1(34, DataType.CONTINUOUS),
    BsmtFinType2(35, DataType.DISCRETE),
    BsmtFinSF2(36, DataType.CONTINUOUS),
    BsmtUnfSF(37, DataType.CONTINUOUS),
    TotalBsmtSF(38, DataType.CONTINUOUS),
    Heating(39, DataType.DISCRETE),
    HeatingQC(40, DataType.DISCRETE),
    CentralAir(41, DataType.DISCRETE),
    Electrical(42, DataType.DISCRETE),
    T1stFlrSF(43, DataType.CONTINUOUS),
    T2ndFlrSF(44, DataType.CONTINUOUS),
    LowQualFinSF(45, DataType.CONTINUOUS),
    GrLivArea(46, DataType.CONTINUOUS),
    BsmtFullBath(47, DataType.CONTINUOUS),
    BsmtHalfBath(48, DataType.CONTINUOUS),
    FullBath(49, DataType.CONTINUOUS),
    HalfBath(50, DataType.CONTINUOUS),
    BedroomAbvGr(51, DataType.CONTINUOUS),
    KitchenAbvGr(52, DataType.CONTINUOUS),
    KitchenQual(53, DataType.DISCRETE),
    TotRmsAbvGrd(54, DataType.CONTINUOUS),
    Functional(55, DataType.DISCRETE),
    Fireplaces(56, DataType.CONTINUOUS),
    FireplaceQu(57, DataType.DISCRETE),
    GarageType(58, DataType.DISCRETE),
    GarageYrBlt(59, DataType.CONTINUOUS),
    GarageFinish(60, DataType.DISCRETE),
    GarageCars(61, DataType.CONTINUOUS),
    GarageArea(62, DataType.CONTINUOUS),
    GarageQual(63, DataType.DISCRETE),
    GarageCond(64, DataType.DISCRETE),
    PavedDrive(65, DataType.DISCRETE),
    WoodDeckSF(66, DataType.CONTINUOUS),
    OpenPorchSF(67, DataType.CONTINUOUS),
    EnclosedPorch(68, DataType.CONTINUOUS),
    T3SsnPorch(69, DataType.CONTINUOUS),
    ScreenPorch(70, DataType.CONTINUOUS),
    PoolArea(71, DataType.CONTINUOUS),
    PoolQC(72, DataType.DISCRETE),
    Fence(73, DataType.DISCRETE),
    MiscFeature(74, DataType.DISCRETE),
    MiscValMoSold(75, DataType.CONTINUOUS),
    MoSold(76, DataType.CONTINUOUS),
    YrSold(77, DataType.CONTINUOUS),
    SaleType(78, DataType.DISCRETE),
    SaleCondition(79, DataType.DISCRETE),
    SalePrice(80, DataType.CONTINUOUS, true)
}