---
title: Introduction to Kaggle - House Prices, Advanced Regression Techniques.
author: Andre Larsson
date: 2021-02-11
key: kart21
comment: true
tags: R Kaggle regression xgboost random-forest dplyr recursive-feature-selection
output:
 md_document:
  pandoc_args: ["--wrap=none"]
  variant: gfm
  toc: true
  preserve_yaml: TRUE
---

  - [Introduction](#introduction)
      - [Getting started with Kaggle](#getting-started-with-kaggle)
      - [The Ames Housing data](#the-ames-housing-data)
      - [dplyr and DescTools](#dplyr-and-desctools)
  - [Data import and overview](#data-import-and-overview)
      - [Importing the data](#importing-the-data)
      - [Overview of the data](#overview-of-the-data)
  - [Data wrangling](#data-wrangling)
      - [Check numerical variables for errors](#check-numerical-variables-for-errors)
      - [Check categorical variables for errors](#check-categorical-variables-for-errors)
      - [Replacing NA’s with Absent](#replacing-nas-with-absent)
      - [Check dataset consistency](#check-dataset-consistency)
      - [Imputation with missForest](#imputation-with-missforest)
  - [Model evaluation](#model-evaluation)
      - [Defining the metric](#defining-the-metric)
      - [Training/validation](#trainingvalidation)
  - [Random forests](#random-forests)
      - [Creating the model](#creating-the-model)
      - [Recursive feature selection](#recursive-feature-selection)
      - [Submitting predictions to Kaggle](#submitting-predictions-to-kaggle)
  - [XGBoost](#xgboost)
      - [Data preparation](#data-preparation)
      - [Defining the model](#defining-the-model)
      - [Create final model and submit result](#create-final-model-and-submit-result)
  - [Conclusion](#conclusion)

# Introduction

This tutorial will go through the steps to make an initial submission to Kaggle competition *House Prices - Advanced Regression Techniques* available at <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>, using R for the analysis.

As it is an old competition, currently in the *Getting Started Prediction* category at Kaggle, there are no prizes to win, but it can be a great exercise for getting comfortable working with tabular data. I find that although (or because of?) it is an older competition, Kaggle competitions can be pretty competitive, and while the main focus of this tutorial is learning, I hope that this tutorial also could provide a good starting point for anyone new to this dataset intending to reach for higher scores. In summary this tutorial will cover

  - Basic data manipulation with dplyr.
  - Feature selection using recursive feature elimination.
  - Handling missing values using Random Forests with the package missForest.
  - Obtaining predictions using Random Forests and XGBoost.

I will be as explicit as possible regarding parameter values, models and corresponding score on Kaggle, providing a baseline for the reader to improve upon.

Note that in order to retrieve the data for this competition you need to register for a free Kaggle account and join this competition at the Kaggle website.

## Getting started with Kaggle

If you are new to Kaggle, you can create a free account at [kaggle.com](https://www.kaggle.com/), and register for the competition you want to participate in, for this tutorial you want to register for: House Prices - Advanced Regression Techniques.

The next step is to create a notebook on the Kaggle website with your solution (at the time of writing they support either R or Python notebooks), run it at their servers, and submit your solution to get the score on the test set for the leaderboard. You could also use the command line interface (CLI) to submit your solution, running your code at your own computer, or at a cloud computing service of choice.

Apart from the competition covered here, I found that the Titanic dataset is a good starting point, with many great tutorials available online. I recommend watching the official Kaggle introduction for the Titanic dataset on youtube, as a gateway to making your first submission: [How to Get Started with Kaggle’s Titanic Competition | Kaggle](https://youtu.be/8yZMXCaFshs).

Personally, I prefer using the Kaggle CLI rather that the website, the CLI provides a quick and convenient way of interacting with Kaggle, making submissions, signing up for competitions, etc (you can send up to a total of 10 submissions/day using the website and CLI). Check out the official Kaggle documentation for their CLI here: [How to Use Kaggle](https://www.kaggle.com/docs/api)

## The Ames Housing data

The dataset for this challenge consists of sale prices of residential property in Ames, Iowa, US, between 2006 and 2010. In total there are 2919 observations (rows) and 81 variables (columns) both numerical, categorical and discrete. The independent variable which we want to predict is the sale price in US dollars.

The result will be evaluated by Kaggle with the root mean squared error between the logarithmised price of the prediction you submit and the true sale price. The submission file should have two columns: property ID and SalePrice, for more details check out <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.

## dplyr and DescTools

We will begin with loading both the test/train data and join them to make sure we do all initial data wrangling on all the data. We will make use of the dplyr package and Desctools, so we start by importing these, you might need to install them with e.g. Install.packages(“package\_name”).

``` r
require("dplyr")
require("DescTools")
require("tidyr") # included in tidyverse and required for pivot_longer() used later
```

# Data import and overview

## Importing the data

Now we load the test/train data, which I have downloaded from the Kaggle site and placed in the same working directory as this script. Then, we merge the test and training data to one larger dataset to make sure we do all preprocessing/data wrangling on the complete dataset.

We also save the id’s of all observations in the test data to keep track of what is train and what is test.

``` r
train_orig = read.csv("train.csv", stringsAsFactors = T) # read training data
test_orig = read.csv("test.csv", stringsAsFactors = T) # read test data
test_id = test_orig$Id # get ID of rows corresponding to the test data
all_data = full_join(train_orig, test_orig) # join test/train data (this works since test/train have exactly the same type of columns)
```

Now, define a convenience function to get back the test and train sets from the full merged data set, and check the resulting number of rows and columns of our data.

``` r
get_tt = function(df){
  train = df %>% filter(!(Id %in% test_id)) %>% mutate(Id=NULL)
  test = df %>% filter(Id %in% test_id) %>% mutate(Id=NULL)
  return(list("train" = train, "test" = test))
}
res = get_tt(all_data)
cat("#Columns in data:", ncol(all_data), "\n")
```

    ## #Columns in data: 81

``` r
cat("#Rows in training data:", nrow(all_data), "\n")
```

    ## #Rows in training data: 2919

``` r
cat("#Rows in training data:", nrow(res$train), "\n")
```

    ## #Rows in training data: 1460

``` r
cat("#Rows in test data:",nrow(res$test), "\n")
```

    ## #Rows in test data: 1459

## Overview of the data

We start by plotting the distribution of the dependent variable SalePrice, where we note that the distribution seems skewed towards the left.

``` r
hist(main="SalePrice", all_data$SalePrice, xlab="SalePrice ($)")
```

![](/images/housing-prices-kaggle/unnamed-chunk-1-1.png)<!-- -->

Interestingly, if we instead plot the log of the SalePrice, the distribution looks more normal, providing further motivation for using the logarithmised sale price rather than the value in dollars.

``` r
hist(main="SalePrice (log)", log(all_data$SalePrice), xlab="SalePrice (log)")
```

![](/images/housing-prices-kaggle/unnamed-chunk-2-1.png)<!-- -->

Now, with DescTools, we can get an overview of the whole dataset. The function Desc gives a summary of each variable, depending on the data type. As it gives us a very long list of output, we will instead use Abstract which only lists all variables with their name, type, missing values, along with the different levels for factors (Abstract is otherwise also printed by Desc)

``` r
#Desc(all_data, plotit=FALSE) # maybe too much information
Abstract(all_data, list.len=100)
```

    ## ----------------------------------------------------------------------------------------------------------------------------------------- 
    ## all_data
    ## 
    ## data frame:  2919 obs. of  81 variables
    ##      0 complete cases (0.0%)
    ## 
    ##   Nr  ColName        Class    NAs           Levels                                                         
    ##   1   Id             integer     .                                                                         
    ##   2   MSSubClass     integer     .                                                                         
    ##   3   MSZoning       factor      4 (0.1%)   (5): 1-C (all), 2-FV, 3-RH, 4-RL, 5-RM                         
    ##   4   LotFrontage    integer   486 (16.6%)                                                                 
    ##   5   LotArea        integer     .                                                                         
    ##   6   Street         factor      .          (2): 1-Grvl, 2-Pave                                            
    ##   7   Alley          factor   2721 (93.2%)  (2): 1-Grvl, 2-Pave                                            
    ##   8   LotShape       factor      .          (4): 1-IR1, 2-IR2, 3-IR3, 4-Reg                                
    ##   9   LandContour    factor      .          (4): 1-Bnk, 2-HLS, 3-Low, 4-Lvl                                
    ##   10  Utilities      factor      2 (0.1%)   (2): 1-AllPub, 2-NoSeWa                                        
    ##   11  LotConfig      factor      .          (5): 1-Corner, 2-CulDSac, 3-FR2, 4-FR3, 5-Inside               
    ##   12  LandSlope      factor      .          (3): 1-Gtl, 2-Mod, 3-Sev                                       
    ##   13  Neighborhood   factor      .          (25): 1-Blmngtn, 2-Blueste, 3-BrDale, 4-BrkSide, 5-ClearCr, ...
    ##   14  Condition1     factor      .          (9): 1-Artery, 2-Feedr, 3-Norm, 4-PosA, 5-PosN, ...            
    ##   15  Condition2     factor      .          (8): 1-Artery, 2-Feedr, 3-Norm, 4-PosA, 5-PosN, ...            
    ##   16  BldgType       factor      .          (5): 1-1Fam, 2-2fmCon, 3-Duplex, 4-Twnhs, 5-TwnhsE             
    ##   17  HouseStyle     factor      .          (8): 1-1.5Fin, 2-1.5Unf, 3-1Story, 4-2.5Fin, 5-2.5Unf, ...     
    ##   18  OverallQual    integer     .                                                                         
    ##   19  OverallCond    integer     .                                                                         
    ##   20  YearBuilt      integer     .                                                                         
    ##   21  YearRemodAdd   integer     .                                                                         
    ##   22  RoofStyle      factor      .          (6): 1-Flat, 2-Gable, 3-Gambrel, 4-Hip, 5-Mansard, ...         
    ##   23  RoofMatl       factor      .          (8): 1-ClyTile, 2-CompShg, 3-Membran, 4-Metal, 5-Roll, ...     
    ##   24  Exterior1st    factor      1 (0.0%)   (15): 1-AsbShng, 2-AsphShn, 3-BrkComm, 4-BrkFace, 5-CBlock, ...
    ##   25  Exterior2nd    factor      1 (0.0%)   (16): 1-AsbShng, 2-AsphShn, 3-Brk Cmn, 4-BrkFace, 5-CBlock, ...
    ##   26  MasVnrType     factor     24 (0.8%)   (4): 1-BrkCmn, 2-BrkFace, 3-None, 4-Stone                      
    ##   27  MasVnrArea     integer    23 (0.8%)                                                                  
    ##   28  ExterQual      factor      .          (4): 1-Ex, 2-Fa, 3-Gd, 4-TA                                    
    ##   29  ExterCond      factor      .          (5): 1-Ex, 2-Fa, 3-Gd, 4-Po, 5-TA                              
    ##   30  Foundation     factor      .          (6): 1-BrkTil, 2-CBlock, 3-PConc, 4-Slab, 5-Stone, ...         
    ##   31  BsmtQual       factor     81 (2.8%)   (4): 1-Ex, 2-Fa, 3-Gd, 4-TA                                    
    ##   32  BsmtCond       factor     82 (2.8%)   (4): 1-Fa, 2-Gd, 3-Po, 4-TA                                    
    ##   33  BsmtExposure   factor     82 (2.8%)   (4): 1-Av, 2-Gd, 3-Mn, 4-No                                    
    ##   34  BsmtFinType1   factor     79 (2.7%)   (6): 1-ALQ, 2-BLQ, 3-GLQ, 4-LwQ, 5-Rec, ...                    
    ##   35  BsmtFinSF1     integer     1 (0.0%)                                                                  
    ##   36  BsmtFinType2   factor     80 (2.7%)   (6): 1-ALQ, 2-BLQ, 3-GLQ, 4-LwQ, 5-Rec, ...                    
    ##   37  BsmtFinSF2     integer     1 (0.0%)                                                                  
    ##   38  BsmtUnfSF      integer     1 (0.0%)                                                                  
    ##   39  TotalBsmtSF    integer     1 (0.0%)                                                                  
    ##   40  Heating        factor      .          (6): 1-Floor, 2-GasA, 3-GasW, 4-Grav, 5-OthW, ...              
    ##   41  HeatingQC      factor      .          (5): 1-Ex, 2-Fa, 3-Gd, 4-Po, 5-TA                              
    ##   42  CentralAir     factor      .          (2): 1-N, 2-Y                                                  
    ##   43  Electrical     factor      1 (0.0%)   (5): 1-FuseA, 2-FuseF, 3-FuseP, 4-Mix, 5-SBrkr                 
    ##   44  X1stFlrSF      integer     .                                                                         
    ##   45  X2ndFlrSF      integer     .                                                                         
    ##   46  LowQualFinSF   integer     .                                                                         
    ##   47  GrLivArea      integer     .                                                                         
    ##   48  BsmtFullBath   integer     2 (0.1%)                                                                  
    ##   49  BsmtHalfBath   integer     2 (0.1%)                                                                  
    ##   50  FullBath       integer     .                                                                         
    ##   51  HalfBath       integer     .                                                                         
    ##   52  BedroomAbvGr   integer     .                                                                         
    ##   53  KitchenAbvGr   integer     .                                                                         
    ##   54  KitchenQual    factor      1 (0.0%)   (4): 1-Ex, 2-Fa, 3-Gd, 4-TA                                    
    ##   55  TotRmsAbvGrd   integer     .                                                                         
    ##   56  Functional     factor      2 (0.1%)   (7): 1-Maj1, 2-Maj2, 3-Min1, 4-Min2, 5-Mod, ...                
    ##   57  Fireplaces     integer     .                                                                         
    ##   58  FireplaceQu    factor   1420 (48.6%)  (5): 1-Ex, 2-Fa, 3-Gd, 4-Po, 5-TA                              
    ##   59  GarageType     factor    157 (5.4%)   (6): 1-2Types, 2-Attchd, 3-Basment, 4-BuiltIn, 5-CarPort, ...  
    ##   60  GarageYrBlt    integer   159 (5.4%)                                                                  
    ##   61  GarageFinish   factor    159 (5.4%)   (3): 1-Fin, 2-RFn, 3-Unf                                       
    ##   62  GarageCars     integer     1 (0.0%)                                                                  
    ##   63  GarageArea     integer     1 (0.0%)                                                                  
    ##   64  GarageQual     factor    159 (5.4%)   (5): 1-Ex, 2-Fa, 3-Gd, 4-Po, 5-TA                              
    ##   65  GarageCond     factor    159 (5.4%)   (5): 1-Ex, 2-Fa, 3-Gd, 4-Po, 5-TA                              
    ##   66  PavedDrive     factor      .          (3): 1-N, 2-P, 3-Y                                             
    ##   67  WoodDeckSF     integer     .                                                                         
    ##   68  OpenPorchSF    integer     .                                                                         
    ##   69  EnclosedPorch  integer     .                                                                         
    ##   70  X3SsnPorch     integer     .                                                                         
    ##   71  ScreenPorch    integer     .                                                                         
    ##   72  PoolArea       integer     .                                                                         
    ##   73  PoolQC         factor   2909 (99.7%)  (3): 1-Ex, 2-Fa, 3-Gd                                          
    ##   74  Fence          factor   2348 (80.4%)  (4): 1-GdPrv, 2-GdWo, 3-MnPrv, 4-MnWw                          
    ##   75  MiscFeature    factor   2814 (96.4%)  (4): 1-Gar2, 2-Othr, 3-Shed, 4-TenC                            
    ##   76  MiscVal        integer     .                                                                         
    ##   77  MoSold         integer     .                                                                         
    ##   78  YrSold         integer     .                                                                         
    ##   79  SaleType       factor      1 (0.0%)   (9): 1-COD, 2-Con, 3-ConLD, 4-ConLI, 5-ConLw, ...              
    ##   80  SaleCondition  factor      .          (6): 1-Abnorml, 2-AdjLand, 3-Alloca, 4-Family, 5-Normal, ...   
    ##   81  SalePrice      integer  1459 (50.0%)

There is a mixture of numerical and categorical variables, with some categorical variables (factors) having overlapping levels (e.g. GarageQual, GarageCond, FireplaceQu, …). Information of the meaning of all different features and levels can be found on Kaggle.

Importantly, we note several features have missing values, which we want to correct later, but before this we first check for errors related to the data type or input form when loading the data into R.

# Data wrangling

## Check numerical variables for errors

When loading our data, all non-numerical should have been loaded as *Factors* in R, while numerical columns are *Numeric* or *integer*.

To check if these datatypes makes sense for the respective features, we first look at all numerical variables to see if we can spot any obvious errors, utilizing the capabilities of the dplyr package, calculating mean, min, max and number of unique values for each column.

``` r
num_summary = all_data %>%
  summarise(across(where(~is.numeric(.)), # do this for numeric columns
                   list(mean = ~mean(.x, na.rm = TRUE), 
                        min = ~min(.x, na.rm = TRUE),
                        max = ~max(.x, na.rm = TRUE),
                        nUnique = ~length(unique(.x, na.rm = TRUE))))) %>%
  pivot_longer(cols = everything(),
               names_sep = "_",
               names_to  = c("variable", ".value"))
print(num_summary, n=100)
```

    ## # A tibble: 38 x 5
    ##    variable             mean   min    max nUnique
    ##    <chr>               <dbl> <int>  <int>   <int>
    ##  1 Id              1460          1   2919    2919
    ##  2 MSSubClass        57.1       20    190      16
    ##  3 LotFrontage       69.3       21    313     129
    ##  4 LotArea        10168.      1300 215245    1951
    ##  5 OverallQual        6.09       1     10      10
    ##  6 OverallCond        5.56       1      9       9
    ##  7 YearBuilt       1971.      1872   2010     118
    ##  8 YearRemodAdd    1984.      1950   2010      61
    ##  9 MasVnrArea       102.         0   1600     445
    ## 10 BsmtFinSF1       441.         0   5644     992
    ## 11 BsmtFinSF2        49.6        0   1526     273
    ## 12 BsmtUnfSF        561.         0   2336    1136
    ## 13 TotalBsmtSF     1052.         0   6110    1059
    ## 14 X1stFlrSF       1160.       334   5095    1083
    ## 15 X2ndFlrSF        336.         0   2065     635
    ## 16 LowQualFinSF       4.69       0   1064      36
    ## 17 GrLivArea       1501.       334   5642    1292
    ## 18 BsmtFullBath       0.430      0      3       5
    ## 19 BsmtHalfBath       0.0614     0      2       4
    ## 20 FullBath           1.57       0      4       5
    ## 21 HalfBath           0.380      0      2       3
    ## 22 BedroomAbvGr       2.86       0      8       8
    ## 23 KitchenAbvGr       1.04       0      3       4
    ## 24 TotRmsAbvGrd       6.45       2     15      14
    ## 25 Fireplaces         0.597      0      4       5
    ## 26 GarageYrBlt     1978.      1895   2207     104
    ## 27 GarageCars         1.77       0      5       7
    ## 28 GarageArea       473.         0   1488     604
    ## 29 WoodDeckSF        93.7        0   1424     379
    ## 30 OpenPorchSF       47.5        0    742     252
    ## 31 EnclosedPorch     23.1        0   1012     183
    ## 32 X3SsnPorch         2.60       0    508      31
    ## 33 ScreenPorch       16.1        0    576     121
    ## 34 PoolArea           2.25       0    800      14
    ## 35 MiscVal           50.8        0  17000      38
    ## 36 MoSold             6.21       1     12      12
    ## 37 YrSold          2008.      2006   2010       5
    ## 38 SalePrice     180921.     34900 755000     664

Comparing the list of numerical variables with the data description on Kaggle, we note that *MSSubClass* is actually a code for the type of residence (see data description on the Kaggle website), and should be a categorical variable (factor), not a numerical value, therefore we simply convert it to a factor as such, with the levels corresponding to the different residential codes for the type of property:

``` r
all_data$MSSubClass = as.factor(all_data$MSSubClass)
```

Further, at least one house has a garage from the future (GarageYrBlt=2207), lets set this value to the same year (YearBuilt) the house was built. In fact, lets do this for all garages built after 2010 (there is only one row like this though), using the mutate and ifelse functions from dplyr.

``` r
all_data = all_data %>%
  mutate(GarageYrBlt = ifelse(GarageYrBlt>2010, YearBuilt, GarageYrBlt))
```

Otherwise, except for the missing values which we save for later, I cannot spot any obvious error in these datasets from this very simple analysis, and we will move on.

## Check categorical variables for errors

After looking at the numerical values, we move on to the categorical features in the dataset. We note that categorical values can be any of two following types:

  - **Ordinal**: Categorical variable with intrinsic order, can be placed on a scale. Example: height (short, average, tall).
  - **Nominal**: Categorical variable with no intrinsic order. Example: list of colors (blue, green, red).

For simplicity, we assume all categorical variables to be of the nominal type, despite this obviously not being true. Several of the categorical variables indicate some form of condition/quality ranging from worst to best, and it might be beneficial to leverage this information by converting these to e.g. (ordered) integers, putting these on a scale of what we think is from low to high value/SalePrice. Deeming this to be out of scope for this tutorial, such an exercise is left to the reader, as it seemed possible to get decent results assuming only nominal categories.

Otherwise, a quick comparison of the categorical variables and the [data description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=data_description.txt) reveals nothing wrong with any of the categorical variables as far as I can see. It could be that some values are mistyped but this is hard for me to judge, and I will not be modifying any of these values. However, several of the categorical values are still missing, and this is what we will focus on next. \# Missing values Shifting our focus to the missing values in the data, we begin defining a function for calculating the percentage of missing values per column, to keep track of our progress:

``` r
perc_missing = function(x) colSums(is.na(x)/nrow(x))*100
pm = perc_missing(all_data)

# Missing values (%) per column, descending
print(pm[order(-pm)])
```

    ##        PoolQC   MiscFeature         Alley         Fence     SalePrice   FireplaceQu   LotFrontage   GarageYrBlt  GarageFinish 
    ##   99.65741692   96.40287770   93.21685509   80.43850634   49.98287085   48.64679685   16.64953751    5.44707091    5.44707091 
    ##    GarageQual    GarageCond    GarageType      BsmtCond  BsmtExposure      BsmtQual  BsmtFinType2  BsmtFinType1    MasVnrType 
    ##    5.44707091    5.44707091    5.37855430    2.80918123    2.80918123    2.77492292    2.74066461    2.70640630    0.82219938 
    ##    MasVnrArea      MSZoning     Utilities  BsmtFullBath  BsmtHalfBath    Functional   Exterior1st   Exterior2nd    BsmtFinSF1 
    ##    0.78794108    0.13703323    0.06851662    0.06851662    0.06851662    0.06851662    0.03425831    0.03425831    0.03425831 
    ##    BsmtFinSF2     BsmtUnfSF   TotalBsmtSF    Electrical   KitchenQual    GarageCars    GarageArea      SaleType            Id 
    ##    0.03425831    0.03425831    0.03425831    0.03425831    0.03425831    0.03425831    0.03425831    0.03425831    0.00000000 
    ##    MSSubClass       LotArea        Street      LotShape   LandContour     LotConfig     LandSlope  Neighborhood    Condition1 
    ##    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000 
    ##    Condition2      BldgType    HouseStyle   OverallQual   OverallCond     YearBuilt  YearRemodAdd     RoofStyle      RoofMatl 
    ##    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000 
    ##     ExterQual     ExterCond    Foundation       Heating     HeatingQC    CentralAir     X1stFlrSF     X2ndFlrSF  LowQualFinSF 
    ##    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000 
    ##     GrLivArea      FullBath      HalfBath  BedroomAbvGr  KitchenAbvGr  TotRmsAbvGrd    Fireplaces    PavedDrive    WoodDeckSF 
    ##    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000 
    ##   OpenPorchSF EnclosedPorch    X3SsnPorch   ScreenPorch      PoolArea       MiscVal        MoSold        YrSold SaleCondition 
    ##    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000    0.00000000

Except for SalePrice, which we want to predict, we note that only PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage have more than 10% missing values, (see official Kaggle description for the meaning of these variables), though there are many columns with some missing values:

``` r
cat("Number of columns in data set:", ncol(all_data), "\n")
```

    ## Number of columns in data set: 81

``` r
cat("Number of columns with NA's:", sum(0 != pm), "\n")
```

    ## Number of columns with NA's: 35

Several of the features with missing values will be remedied in the next section, noting that a missing value is actually a special value indicating a feature being absent. After this modification, the remaining missing values will then be imputed with a Random Forest algorithm.

## Replacing NA’s with Absent

We note 34 features have missing values, ignoring dependent variable SalePrice.

According to the [data description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=data_description.txt) available at the Kaggle website, NA is a special value denoting absent/none for the following categories: FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature, Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2. In theory it is possible that some of these NA’s actually are true missing values, data which was not collected, but we will follow the description and replace all NA values with the category ‘absent’, meaning that an NA indicate lack of e.g. a basement, garage, for the columns mentioned above.

``` r
# define name of columns for which we want to replace NA's with the new category 'Absent'
na_means_absence = c("FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
"MiscFeature", "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")

# helper function for replacing NA's with Absent
fact_replace_na = function(x){
  x =  as.character(x)
  x =  if_else(is.na(x), "Absent", x)
  x = as.factor(x)
}

# apply 'fact_replace_na' to columns listed in 'na_means_absent'
all_data = all_data %>%
  mutate_at(na_means_absence, fact_replace_na)
```

Now, we have 21 columns with missing values instead of 31:

``` r
pm = perc_missing(all_data)
cat("Number of columns with NA's (after replacing NA's with 'absent'):", sum(0 != pm), "\n")
```

    ## Number of columns with NA's (after replacing NA's with 'absent'): 21

## Check dataset consistency

Let’s do a quick check on the subset of columns defined above to see if the dataset is consistent, by noting that if one of the garage variables indicates an absence, the rest of the garage-related variables should also have the value absent.

``` r
garage_c = c("GarageType", "GarageFinish", "GarageQual", "GarageCond")
# extract indices of inconsistent rows (if any)
inconsistent_rows = which(apply(all_data[,garage_c] == "Absent", 1, any) & apply(all_data[,garage_c] != "Absent", 1, any))
all_data[inconsistent_rows,] %>% select(contains("garage"))
```

    ##      GarageType GarageYrBlt GarageFinish GarageCars GarageArea GarageQual GarageCond
    ## 2127     Detchd          NA       Absent          1        360     Absent     Absent
    ## 2577     Detchd          NA       Absent         NA         NA     Absent     Absent

Two observations have a detached (Detchd) GarageType, while other columns indicates absence. It seems possible that a ‘detached’ garage can be seen as also being absent, or semi-absent, depending on how you view it, and here we will keep these data points as they are for the model.

Let’s also look at the basement columns instead and see if they seem consistent.

``` r
bsmt_c=c("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")

# extract indices of inconsistent rows (if any)
inconsistent_rows = which(apply(all_data[,bsmt_c] == "Absent", 1, any) & apply(all_data[,bsmt_c] != "Absent", 1, any))

all_data[inconsistent_rows,] %>% select(contains("bsmt"))
```

    ##      BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinSF1 BsmtFinType2 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath
    ## 333        Gd       TA           No          GLQ       1124       Absent        479      1603        3206            1            0
    ## 949        Gd       TA       Absent          Unf          0          Unf          0       936         936            0            0
    ## 1488       Gd       TA       Absent          Unf          0          Unf          0      1595        1595            0            0
    ## 2041       Gd   Absent           Mn          GLQ       1044          Rec        382         0        1426            1            0
    ## 2186       TA   Absent           No          BLQ       1033          Unf          0        94        1127            0            1
    ## 2218   Absent       Fa           No          Unf          0          Unf          0       173         173            0            0
    ## 2219   Absent       TA           No          Unf          0          Unf          0       356         356            0            0
    ## 2349       Gd       TA       Absent          Unf          0          Unf          0       725         725            0            0
    ## 2525       TA   Absent           Av          ALQ        755          Unf          0       240         995            0            0

In total 9 rows show the basement as being absent while some of the other columns indicates that the basement exists, by giving scores for e.g., height (BsmtQual) or condition (BsmtCond). Sometimes a house can have more than one basement with one of them unfinished (=absent?), and I found it difficult to see just from the data what is correct here (which column we should be trust). Since there are only 9 rows it probably won’t to much harm to keep them, and as these values could contain some important signal relating to quality/building progress, therefore I will leave these unchanged.

Considering the size of this dataset (\>2900 observations), the sample features we looked at seemed fairly consistent, where Absent-type garages were almost always followed by absent/unfinished/detached status in related columns. For our purposes we will not dig any deeper into this, but we note that some gain might be had by making sure that the entire dataset is consistent, by e.g. changing all related values to absent if at least one value indicates the absence of a feature, or by some other strategy.

Despite this, we will treat the status of our dataset at this point to be “good enough” and we will now move on to impute the rest of the missing values using random forests.

## Imputation with missForest

First, we see that although many (=21) features still have missing values, only LotFrontage and GarageYrBlt have more than 1% missing, after replacing NA’s with Absent where applicable according to description (ignoring the target variable SalePrice).

``` r
pm = perc_missing(all_data)
print(pm[pm>1]) # more than 1% missing
```

    ## LotFrontage GarageYrBlt   SalePrice 
    ##   16.649538    5.447071   49.982871

``` r
print(pm[pm>0]) # features with missing values
```

    ##     MSZoning  LotFrontage    Utilities  Exterior1st  Exterior2nd   MasVnrType   MasVnrArea   BsmtFinSF1   BsmtFinSF2    BsmtUnfSF 
    ##   0.13703323  16.64953751   0.06851662   0.03425831   0.03425831   0.82219938   0.78794108   0.03425831   0.03425831   0.03425831 
    ##  TotalBsmtSF   Electrical BsmtFullBath BsmtHalfBath  KitchenQual   Functional  GarageYrBlt   GarageCars   GarageArea     SaleType 
    ##   0.03425831   0.03425831   0.06851662   0.06851662   0.03425831   0.06851662   5.44707091   0.03425831   0.03425831   0.03425831 
    ##    SalePrice 
    ##  49.98287085

We will now impute the remaining missing values with the missForest library.

missForest uses random forests to predict values for the NA’s in the dataset. First we load the package, install it if needed (it is available at CRAN).

``` r
#install.packages("missForest")
require(missForest)
```

Important parameters that we will change here the Random Forest algorithm are ntree, the number of trees in each forest, maxiter, the maximum number of iterations, but see also the documentation (the one you get by typing ?missForest in R) for more information.

We will reduce the number of trees to 10 (from 100) and keep maxiter at 10, and having all other parameters at their default values, as this seems to work well for the current problem.

``` r
# Impute missing values, excluding the target column (SalePrice)
```

``` r
imp_result = missForest(select(all_data, -SalePrice), maxiter=10, ntree=10) # !!!
```

    ##   missForest iteration 1 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 2 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 3 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 4 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 5 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 6 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 7 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 8 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!
    ##   missForest iteration 9 in progress...

    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?
    
    ## Warning in randomForest.default(x = obsX, y = obsY, ntree = ntree, mtry = mtry, : The response has five or fewer unique values. Are you
    ## sure you want to do regression?

    ## done!

``` r
# ' missForest additionally gives us the out-of-bag (OOB) errors calculated by comparing imputed values to
# ' the values in the training data, calculated as the normalized root mean squared error (NRMSE)
# ' for continuous variables and proportion of falsely classified (PFC) for the
# ' categorical variables, providing an indication of the overall imputation error.
imp_result$OOBerror
```

    ##      NRMSE        PFC 
    ## 0.03179195 0.03478810

The error rate introduced by imputation of the missing data is estimated at about 4%, which does not seem to bad. We save this result and verify that no values are missing.

``` r
data_imp = imp_result$ximp

cat("Columns with missing values:", sum(0 != perc_missing(data_imp)), "\n")
```

    ## Columns with missing values: 0

``` r
# add back dependent variable SalePrice to the data
data_imp$SalePrice = all_data$SalePrice
```

After some rudimentary data cleaning and imputation of missing values we now we have a complete dataset (no missing values), and are ready to move on to the modelling.

# Model evaluation

We will model the data with both Random Forests and XGBoost. Random Forests works by combining (bagging) the predictions of several classifiers, each trained on different part of the dataset.

XGBoost is similar in that it also a collection of several classifiers, but unlike Random Forest, classifiers/boosters are added sequentially, with each booster training on the errors/residuals of the previous booster, rather than the errors of the full data set.

Importantly, each tree in a Random Forest are independent, adding more trees, growing larger forests is unlikely to result in overfitting. For XGBoost, adding more boosters could eventually result in overfitting. To check for overfitting, we will split our training dataset to training and validation as outlined below.

## Defining the metric

The score on the Kaggle leaderboard is evaluated by using a root mean square error (RMSE) on the logarithmic difference between the true sale price and our prediction.

Here, we define a function for evaluating the accuracy given a model/fit to the training data, and the actual data/sale price. By default we logarithmise the SalePrice to correspond with the error used to evaluate performance on Kaggle.

If no values for the targets are given, we assume that the targets can be obtained from the data object as data$SalePrice. Further, if the keyword all\_errors=TRUE is given, the individual (squared) errors will be return along with total RMSE of the errors. Note this function is not at all general, and is written for convenience for this tutorial,

``` r
calc_RMSE = function(fit, data, target=NULL, log=TRUE, all_errors=FALSE){
  pred = predict(fit, data, se.fit=FALSE)
  if(is.null(target)) target = data$SalePrice
  if(log){
    pred = log(pred)
    target = log(target)}
  errors = (pred-target)^2
  error = sqrt(mean(errors))
  if(all_errors) return(list("error" = error, "errors" = errors))
  else return(error)
}
```

## Training/validation

Then, define a function for splitting the training data further to training/validation, using 80% for the training and 20% for validation.

``` r
split_data = function(df, f=0.8){
  train.i = sample(nrow(df), f*nrow(df), replace = FALSE)
  return(list("train" = df[train.i,], "val" = df[-train.i,]))
}
tt = get_tt(data_imp) # first split all data to training/test
s = split_data(tt$train) # the split training to training/validation
```

# Random forests

## Creating the model

Now we define and run a random forest model, using the randomForest package. We set the number of trees in each forest ntree=200, the minimum size of end nodes nodesize=10. Further we set importance=TRUE to also save the importance scores for each of the variables (see section on Feature selection below)

``` r
require(randomForest)
rf.1 = randomForest(SalePrice ~ ., data=s$train, importance=TRUE, nodesize=10, ntree=200)
```

Next we evaluate our model by calculating the RMSE (Log) on both train and validation.

``` r
calc_RMSE(rf.1, s$train, log=TRUE)
```

    ## [1] 0.07462615

``` r
calc_RMSE(rf.1, s$val, log=TRUE)
```

    ## [1] 0.1348091

When I run it, I get a higher error for the validation set, however, the first two significant digits of the error for both the training/validation does not change when increasing the number of trees ntree by a factor 10 (simulations not shown here), suggesting that having more trees than 200 will not result in any significant improvement, but also that adding more trees does not reduce the performance by overfitting.

``` r
rf.11 = randomForest(SalePrice ~ ., data=s$train, importance=TRUE, nodesize=10, ntree=2000)
calc_RMSE(rf.11, s$train, log=TRUE)
```

    ## [1] 0.07403767

``` r
calc_RMSE(rf.11, s$val, log=TRUE)
```

    ## [1] 0.1330932

With this, we now have a baseline result from the Random Forest model above (using ntree=200). To further analyse the Random Forest model, lets plot the (squared) errors as a function of (log) SalePrice to see if there is any region in SalePrice where the model performs best/worst.

``` r
plot(log(s$train$SalePrice), calc_RMSE(rf.1, s$train, log=TRUE, all_errors=TRUE)$errors,
     xlab="SalePrice (log)", ylab="Squared log errors")
```

![](/images/housing-prices-kaggle/unnamed-chunk-20-1.png)<!-- -->

From the plot we see a (very slightly) bowl-shaped error distribution biased to the left, with higher errors for low sale prices, but also some large outlier errors throughout the range of all sale prices. One might speculate that model does not prioritize the lower values of SalePrice, since these correspond to small absolute errors with the root mean square error metric used by randomForest, while they in the log-space used by Kaggle correspond to larger relative errors.

However, my experience was that log-transforming the SalePrice did not reduce the error for the Random Forest model (code not shown here, feel free to re-run model with log(SalePrice) instead and see if any improvement is possible).

## Recursive feature selection

Now, we will try to find variables which can be excluded from the analysis using recursive feature selection. This will be done in an attempt to simplify the dataset and avoid overfitting.

To this end we use the same Random Forest model as above, but extracting the importance scores for each variable to see which variables are important for the accuracy of our prediction of SalePrice on the validation set. The method randomForest in R calculates two importance scores, the first one relating to the mean decrease in the mean squared error (MSE), and the second one relating to the decrease in average node purity/Gini index. As I am not sure on how to interpret the importance score for Gini index, I will instead focus on the decrease in MSE.

A positive value for the MSE importance score *%IncMSE* reported by randomForest means that, when the corresponding variable is replaced with random permutation, the average MSE increases by the indicated amount. We plot the importance score (from the Random Forest model trained above) with varImpPlot(), and retrieve them with importance(),

``` r
varImpPlot(main="Variable Importance", rf.1)
```

![](/images/housing-prices-kaggle/varImp-1.png)<!-- -->

``` r
imp = importance(rf.1)
print(imp)
```

    ##                   %IncMSE IncNodePurity
    ## MSSubClass    13.94828540  8.537607e+10
    ## MSZoning       5.20953566  8.693113e+09
    ## LotFrontage    3.32240198  6.686194e+10
    ## LotArea        8.63612496  9.001305e+10
    ## Street         0.00000000  1.525113e+07
    ## Alley          2.28940631  1.890624e+09
    ## LotShape       1.85828577  8.991048e+09
    ## LandContour    0.78165860  1.201193e+10
    ## Utilities      0.00000000  0.000000e+00
    ## LotConfig     -0.25436435  6.148034e+09
    ## LandSlope      0.02868953  5.608428e+09
    ## Neighborhood  16.43919739  5.951878e+11
    ## Condition1     0.93448371  4.291198e+09
    ## Condition2     0.00000000  9.826347e+07
    ## BldgType       2.86931642  2.487355e+09
    ## HouseStyle     5.63338578  1.095229e+10
    ## OverallQual   14.37079299  1.901925e+12
    ## OverallCond    5.42901017  1.821344e+10
    ## YearBuilt      6.72594092  1.586183e+11
    ## YearRemodAdd   5.42076143  3.794508e+10
    ## RoofStyle      1.95772714  4.601820e+09
    ## RoofMatl      -1.13178897  2.237628e+09
    ## Exterior1st    5.06659316  4.467979e+10
    ## Exterior2nd    5.57089140  4.131884e+10
    ## MasVnrType     3.22689957  5.410232e+09
    ## MasVnrArea     3.50786897  4.095333e+10
    ## ExterQual      6.86463758  4.315535e+11
    ## ExterCond      0.39664800  4.074968e+09
    ## Foundation     3.32878587  6.093636e+09
    ## BsmtQual       6.28535101  2.791654e+11
    ## BsmtCond       2.92281567  5.064502e+09
    ## BsmtExposure   2.48078928  2.182818e+10
    ## BsmtFinType1   9.27219325  2.689165e+10
    ## BsmtFinSF1     8.53113551  1.769531e+11
    ## BsmtFinType2  -0.92990215  3.312520e+09
    ## BsmtFinSF2     1.80435906  2.274085e+09
    ## BsmtUnfSF      4.44901758  3.062591e+10
    ## TotalBsmtSF   11.67709552  2.522547e+11
    ## Heating       -0.93441428  1.181318e+09
    ## HeatingQC      2.53746326  6.977427e+09
    ## CentralAir     7.47321129  1.562434e+10
    ## Electrical    -2.19439050  1.218730e+09
    ## X1stFlrSF     11.31890268  2.367449e+11
    ## X2ndFlrSF     10.33234768  7.458474e+10
    ## LowQualFinSF   1.68086049  2.439087e+09
    ## GrLivArea     20.98920220  7.207352e+11
    ## BsmtFullBath   3.68217098  8.718224e+09
    ## BsmtHalfBath   1.36993191  9.584828e+08
    ## FullBath       4.90949521  8.935012e+10
    ## HalfBath       4.11247888  4.461927e+09
    ## BedroomAbvGr   2.32784822  1.415470e+10
    ## KitchenAbvGr   2.33223016  1.063762e+09
    ## KitchenQual    6.62440472  2.138217e+11
    ## TotRmsAbvGrd   2.92439039  5.634508e+10
    ## Functional     2.09159503  3.394030e+09
    ## Fireplaces     5.58035006  2.373917e+10
    ## FireplaceQu    7.72604918  6.031233e+10
    ## GarageType     5.58240876  3.494757e+10
    ## GarageYrBlt    2.42426712  3.837650e+10
    ## GarageFinish   5.41466994  5.168643e+10
    ## GarageCars    10.89518933  6.186495e+11
    ## GarageArea     8.69859534  1.967675e+11
    ## GarageQual     1.18784590  7.185648e+09
    ## GarageCond     2.19416640  5.250321e+09
    ## PavedDrive     2.16316199  2.792274e+09
    ## WoodDeckSF     4.28441065  2.143519e+10
    ## OpenPorchSF    6.01244998  2.192954e+10
    ## EnclosedPorch -0.15250179  2.112278e+09
    ## X3SsnPorch     0.94110007  1.502575e+09
    ## ScreenPorch    1.70749194  5.406866e+09
    ## PoolArea       0.00000000  4.942437e+09
    ## PoolQC         1.00250941  6.166950e+09
    ## Fence          1.77321395  6.946657e+09
    ## MiscFeature    0.72043205  2.281530e+08
    ## MiscVal       -1.57902954  4.192645e+08
    ## MoSold        -1.03270568  1.237473e+10
    ## YrSold        -0.55112362  4.852335e+09
    ## SaleType       1.49536958  6.487723e+09
    ## SaleCondition  1.22189311  1.297315e+10

It seems that there are many variables making small contributions to the total prediction, and we keep as many as possible as we there are many features and we haven’t fully explored the interdependencies.

However, some variables actually correspond to a reduction of the error if they are ignored/changed, as they have a negative value for the reported *%IncMSE*. We will here remove all variable with negative importance, using recursive feature elimination. I was not able to find such a function in R, so instead we will do the feature elimination ourselves.

To this end, we will create a loop where we for each iteration train a Random Forest model and find the variable with the lowest importance. If this importance is less than zero, we remove the feature from the dataset, and start a new iteration, until all variables have a positive importance.

``` r
m_data = data_imp # make a copy of the imputed data
min_imp = -1 # to run loop at least one time

while(min_imp<0){
  tt = get_tt(m_data) # split all data to training/test (again)
  s = split_data(tt$train) # split training to training/validation (again)
  rf.temp = randomForest(SalePrice ~ ., data=s$train, importance=TRUE, nodesize=10, ntree=200) #!!!
  imp = importance(rf.temp)
  imp_worst = which.min(imp[,1])
  min_imp = imp[imp_worst,1]
  print("Variable with least importance:")
  print(imp_worst)
  cat("Min. importance:", min_imp, "\n" )
  if(min_imp<0){
  m_data = m_data %>% select(-imp_worst)
  m_data$SalePrice = all_data$SalePrice # new dataset with SalePrice added back
  m_data$Id = all_data$Id # new dataset with Id added back
  cat("Variables in original dataset:", ncol(all_data),
      ". Variables in new dataset:", ncol(m_data), "\n")
  }
  else{
    cat("No variable removed.\n")
  }
}
```

    ## [1] "Variable with least importance:"
    ## LotShape 
    ##        7 
    ## Min. importance: -1.815413 
    ## Variables in original dataset: 81 . Variables in new dataset: 80 
    ## [1] "Variable with least importance:"
    ## Heating 
    ##      38 
    ## Min. importance: -2.142186 
    ## Variables in original dataset: 81 . Variables in new dataset: 79 
    ## [1] "Variable with least importance:"
    ## YrSold 
    ##     75 
    ## Min. importance: -2.110181 
    ## Variables in original dataset: 81 . Variables in new dataset: 78 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       21 
    ## Min. importance: -1.620655 
    ## Variables in original dataset: 81 . Variables in new dataset: 77 
    ## [1] "Variable with least importance:"
    ## BsmtHalfBath 
    ##           45 
    ## Min. importance: -1.439187 
    ## Variables in original dataset: 81 . Variables in new dataset: 76 
    ## [1] "Variable with least importance:"
    ## MiscFeature 
    ##          70 
    ## Min. importance: -1.939519 
    ## Variables in original dataset: 81 . Variables in new dataset: 75 
    ## [1] "Variable with least importance:"
    ## X3SsnPorch 
    ##         65 
    ## Min. importance: -1.69245 
    ## Variables in original dataset: 81 . Variables in new dataset: 74 
    ## [1] "Variable with least importance:"
    ## Heating 
    ##      36 
    ## Min. importance: -1.68866 
    ## Variables in original dataset: 81 . Variables in new dataset: 73 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       20 
    ## Min. importance: -1.792645 
    ## Variables in original dataset: 81 . Variables in new dataset: 72 
    ## [1] "Variable with least importance:"
    ## LandContour 
    ##           7 
    ## Min. importance: -1.314049 
    ## Variables in original dataset: 81 . Variables in new dataset: 71 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##         12 
    ## Min. importance: -1.562408 
    ## Variables in original dataset: 81 . Variables in new dataset: 70 
    ## [1] "Variable with least importance:"
    ## MiscVal 
    ##      65 
    ## Min. importance: -1.475161 
    ## Variables in original dataset: 81 . Variables in new dataset: 69 
    ## [1] "Variable with least importance:"
    ## PoolArea 
    ##       62 
    ## Min. importance: -1.843396 
    ## Variables in original dataset: 81 . Variables in new dataset: 68 
    ## [1] "Variable with least importance:"
    ## PoolQC 
    ##     62 
    ## Min. importance: -1.414819 
    ## Variables in original dataset: 81 . Variables in new dataset: 67 
    ## [1] "Variable with least importance:"
    ## MiscVal 
    ##      62 
    ## Min. importance: -2.40238 
    ## Variables in original dataset: 81 . Variables in new dataset: 66 
    ## [1] "Variable with least importance:"
    ## MiscVal 
    ##      61 
    ## Min. importance: -1.824029 
    ## Variables in original dataset: 81 . Variables in new dataset: 65 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           38 
    ## Min. importance: -1.926025 
    ## Variables in original dataset: 81 . Variables in new dataset: 64 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##         11 
    ## Min. importance: -1.979578 
    ## Variables in original dataset: 81 . Variables in new dataset: 63 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##         10 
    ## Min. importance: -1.029346 
    ## Variables in original dataset: 81 . Variables in new dataset: 62 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##          9 
    ## Min. importance: -2.522418 
    ## Variables in original dataset: 81 . Variables in new dataset: 61 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           34 
    ## Min. importance: -1.614837 
    ## Variables in original dataset: 81 . Variables in new dataset: 60 
    ## [1] "Variable with least importance:"
    ## Electrical 
    ##         32 
    ## Min. importance: -2.365893 
    ## Variables in original dataset: 81 . Variables in new dataset: 59 
    ## [1] "Variable with least importance:"
    ## Electrical 
    ##         31 
    ## Min. importance: -1.313228 
    ## Variables in original dataset: 81 . Variables in new dataset: 58 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       14 
    ## Min. importance: -1.592968 
    ## Variables in original dataset: 81 . Variables in new dataset: 57 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       13 
    ## Min. importance: -2.188269 
    ## Variables in original dataset: 81 . Variables in new dataset: 56 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       12 
    ## Min. importance: -2.332414 
    ## Variables in original dataset: 81 . Variables in new dataset: 55 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       11 
    ## Min. importance: -1.688323 
    ## Variables in original dataset: 81 . Variables in new dataset: 54 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##       10 
    ## Min. importance: -1.488006 
    ## Variables in original dataset: 81 . Variables in new dataset: 53 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##          8 
    ## Min. importance: -1.862336 
    ## Variables in original dataset: 81 . Variables in new dataset: 52 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##          7 
    ## Min. importance: -0.9858868 
    ## Variables in original dataset: 81 . Variables in new dataset: 51 
    ## [1] "Variable with least importance:"
    ## Condition2 
    ##          6 
    ## Min. importance: -2.033151 
    ## Variables in original dataset: 81 . Variables in new dataset: 50 
    ## [1] "Variable with least importance:"
    ## Electrical 
    ##         22 
    ## Min. importance: -1.584624 
    ## Variables in original dataset: 81 . Variables in new dataset: 49 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##        6 
    ## Min. importance: -1.400095 
    ## Variables in original dataset: 81 . Variables in new dataset: 48 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           21 
    ## Min. importance: -0.1391719 
    ## Variables in original dataset: 81 . Variables in new dataset: 47 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           20 
    ## Min. importance: -1.755043 
    ## Variables in original dataset: 81 . Variables in new dataset: 46 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           19 
    ## Min. importance: -1.679535 
    ## Variables in original dataset: 81 . Variables in new dataset: 45 
    ## [1] "Variable with least importance:"
    ## YrSold 
    ##     41 
    ## Min. importance: -1.062073 
    ## Variables in original dataset: 81 . Variables in new dataset: 44 
    ## [1] "Variable with least importance:"
    ## YrSold 
    ##     40 
    ## Min. importance: -2.095699 
    ## Variables in original dataset: 81 . Variables in new dataset: 43 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##        5 
    ## Min. importance: -1.634689 
    ## Variables in original dataset: 81 . Variables in new dataset: 42 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           17 
    ## Min. importance: -1.2926 
    ## Variables in original dataset: 81 . Variables in new dataset: 41 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##        4 
    ## Min. importance: -1.785696 
    ## Variables in original dataset: 81 . Variables in new dataset: 40 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           15 
    ## Min. importance: -1.25113 
    ## Variables in original dataset: 81 . Variables in new dataset: 39 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           14 
    ## Min. importance: -0.9638666 
    ## Variables in original dataset: 81 . Variables in new dataset: 38 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           13 
    ## Min. importance: -0.6903612 
    ## Variables in original dataset: 81 . Variables in new dataset: 37 
    ## [1] "Variable with least importance:"
    ## RoofMatl 
    ##        3 
    ## Min. importance: -1.432617 
    ## Variables in original dataset: 81 . Variables in new dataset: 36 
    ## [1] "Variable with least importance:"
    ## LowQualFinSF 
    ##           11 
    ## Min. importance: -1.085729 
    ## Variables in original dataset: 81 . Variables in new dataset: 35 
    ## [1] "Variable with least importance:"
    ## YrSold 
    ##     31 
    ## Min. importance: -1.890011 
    ## Variables in original dataset: 81 . Variables in new dataset: 34 
    ## [1] "Variable with least importance:"
    ## SaleCondition 
    ##            32 
    ## Min. importance: 0.3700816 
    ## No variable removed.

Now let’s calculate the RMSE (log) of our model with the new set of features to compare it to the model with the original set.

``` r
cat("Variables in original dataset:", ncol(all_data),
    ". Variables in new dataset:", ncol(m_data), "\n")
```

    ## Variables in original dataset: 81 . Variables in new dataset: 34

``` r
calc_RMSE(rf.temp, s$train, log=TRUE)
```

    ## [1] 0.09553543

``` r
calc_RMSE(rf.temp, s$val, log=TRUE)
```

    ## [1] 0.1683352

Interestingly, our model with removed variables performed worse on the validation compared to the original data set, despite only removing features with negative importance scores. Maybe this suggests large interdependence effects across the variables, with certain combinations of variables being important, relations cannot be seen when looking at the variables one by one.

Since we got a worse result after feature selection, we are obliged to keep the original variable set for the final model.

## Submitting predictions to Kaggle

For the final Kaggle submission, we will use the full training data set (training+validation) for fitting the model and increase the number of trees to ntree=2000.

``` r
tt = get_tt(data_imp) # split all data to training/test (again)

rf.2 = randomForest(SalePrice ~ ., data=tt$train, nodesize=10, ntree=2000)
print(rf.2)
```

    ## 
    ## Call:
    ##  randomForest(formula = SalePrice ~ ., data = tt$train, nodesize = 10,      ntree = 2000) 
    ##                Type of random forest: regression
    ##                      Number of trees: 2000
    ## No. of variables tried at each split: 26
    ## 
    ##           Mean of squared residuals: 767505817
    ##                     % Var explained: 87.83

``` r
calc_RMSE(rf.2, tt$train, log=TRUE)
```

    ## [1] 0.07182295

Looks similar to the result we got before on the training data (where we keept some observations for validation). We now use this model to get the predictions for SalePrince on the test set,

``` r
pred.test = predict(rf.2, tt$test)
```

Before we submit our results, we plot the distribution of the predicted (logarithmised) SalePrice in our test set (red) on top of the distribution of the (logarithmised) SalePrice in the original training data (blue), to make sure everything looks okay.

``` r
hist(log(tt$train$SalePrice), main="Training (blue), test (red) predictions", xlab = "SalePrice (log)", freq=FALSE, col=scales::alpha('blue',0.5), ylim=c(0,1.5))
hist(log(pred.test),  freq=FALSE, add=T, col=scales::alpha('red',.5))
```

![](/images/housing-prices-kaggle/unnamed-chunk-23-1.png)<!-- -->

We note that the distributions look fairly similar, although there seem to be less outliers, high/low sale prices for the test predictions (red) compared to the training data (blue), but not sure if this is important.

Let’s save our test predictions and the corresponding ID’s (which we stored in the variable *test\_id* at the beginning) as *submission.csv*, this file can then be submitted to Kaggle, I got the very modest score 0.150 using this model.

``` r
subm = cbind(test_id, pred.test)
colnames(subm) = c("Id", "SalePrice")
write.csv(subm, "submission.csv", row.names=FALSE)
```

# XGBoost

We will also train a model based on XGBoost. We begin by importing the *xgboost* package (install if need be, it is available at CRAN) which provide an R interface to the XGBoost algorithm.

``` r
require(xgboost)
```

XGBoost, short for extreme gradient boosting, uses boosting which sequentially adds models that trains on/corrects the errors made by previous models. As we will see it can be a powerful method able to bring about great results at relatively low computational cost, although with a risk of overfitting.

## Data preparation

We start by retrieving new training/validation/test data sets. Here, we need to consider that xgboost, unlike randomForest, only accepts numerical variables, as such we need to somehow recode our categorical values into integers. This can be done e.g., using one-hot encoding, where the presence of each category is indicated by a 1 and its absence is indicated by a 0.

Fortunately for us, this is very easy to do with the package fastDummies.

``` r
tt = get_tt(data_imp) # get training/test from imputed data

# one-hot encoding of categorical variables
require("fastDummies")
tt$train = dummy_cols(tt$train, remove_selected_columns =TRUE, remove_first_dummy=TRUE)
tt$test = dummy_cols(tt$test, remove_selected_columns   =TRUE, remove_first_dummy=TRUE)
```

When creating the dummy columns we set remove\_selected\_columns=TRUE to make sure we also discard the old columns with factors. I also set the remove\_first\_dummy=TRUE, meaning that the first category is removed for each variable, as this somewhat reduces the amount of columns in our final matrix. Let’s continue and get the train/validation/test sets.

``` r
s = split_data(tt$train) # split data to train/test
train = s$train
val = s$val
test = tt$test
```

The package xgboost wants data in the DMatrix format. A DMatrix can be created from a matrix **X** of features (independent variables) and vector **y** of the corresponding targets (dependent variable).

To this end we will extract X (features) and y (target) for the training/validation/test sets and create the DMatrices we need for xgboost.

For xgboost, I found better results logarithmising the SalePrice before training the model, so this is what we’ll be doing here.

``` r
train_y = log(train$SalePrice)
train_X = data.matrix(subset(train, select = - SalePrice))

val_y = log(val$SalePrice)
val_X = data.matrix(subset(val, select = - SalePrice))

test_y = log(test$SalePrice)
test_X = data.matrix(subset(test, select = - SalePrice))

dtrain = xgb.DMatrix(data = train_X, label = train_y)
dtest = xgb.DMatrix(data = test_X, label = test_y)
```

## Defining the model

Now we set the parameters for running xgboost. The ones we will set here is max\_depth, maximum depth of each tree (default=6), lambda and alpha which are the regularization strengths for L1 and L2 regularization (default=0 for both), min\_child\_weight which for linear regression corresponds to the minimum number of instances in each node (default=1), eta meaning the learning rate (default=0.3). For the objective function, I found the squarederror to work well (objective=reg:squarederror).

We begin by setting the parameters at the default values and fit the model, to get a baseline result.

``` r
xgb.paras=list(max_depth=6,lambda=0.0,alpha=0.0,min_child_weight=0, booster ="gbtree",
               eta=0.3, objective="reg:squarederror")
xgb.fit0 = xgboost(data = dtrain, nrounds=300,params=xgb.paras, nthread = 6, verbose=0)
```

Then calculate the error on both training and validation

``` r
calc_RMSE(xgb.fit0, train_X, target=train_y, log=FALSE)
```

    ## [1] 0.001050024

``` r
calc_RMSE(xgb.fit0, val_X, target=val_y, log=FALSE)
```

    ## [1] 0.1606479

Note that although we were able to get a very low error on the training data, the error on the validation was much higher. After some behind-the-scenes experimenting/scanning of parameter values, I found the parameters below to work better, providing a lower error on the validation set, feel free to see if you can find any improvements.

``` r
xgb.paras=list(max_depth=3,lambda=0.01,alpha=0.01,min_child_weight=0, booster ="gbtree",
               eta=0.01, objective="reg:squarederror")
xgb.fit = xgboost(data = dtrain, nrounds=2000,params=xgb.paras, nthread = 6, verbose=0)
```

Again, when calculating the errors for training and validation, the above parameters yields a somewhat lower validation error. As the training error also is higher, it might be that we avoided some overfitting, indicating a better model with less variance.

``` r
calc_RMSE(xgb.fit, train_X, target=train_y, log=FALSE)
```

    ## [1] 0.0598248

``` r
calc_RMSE(xgb.fit, val_X, target=val_y, log=FALSE)
```

    ## [1] 0.1388961

Now, we analyse the model by plotting the log-errors as a function of the log-SalePrice, as was done for the Random Forests. In the validation set, the errors seem evenly distributed, with no clear bias towards being better at predicting high/low sale prices, but with some outliers where the prediction was off by larger amounts.

``` r
plot(train_y, calc_RMSE(xgb.fit, train_X, target=train_y, log=TRUE, all_errors=TRUE)$errors,
     xlab="SalePrice (log)", ylab="root mean square log errors")
```

![](/images/housing-prices-kaggle/unnamed-chunk-31-1.png)<!-- -->

``` r
plot(val_y, calc_RMSE(xgb.fit, val_X, target=val_y, log=TRUE, all_errors=TRUE)$errors,
     xlab="SalePrice (log)", ylab="root mean square log errors")
```

![](/images/housing-prices-kaggle/unnamed-chunk-31-2.png)<!-- -->

## Create final model and submit result

For the final model used in the Kaggle submission, we train on all available training data (training + validation) with the same parameters as the best model found above.

``` r
trainall_y = log(tt$train$SalePrice)
trainall_X = data.matrix(subset(tt$train, select = - SalePrice))
dtrainall = xgb.DMatrix(data = trainall_X, label = trainall_y)

xgb.fitall = xgboost(data = dtrainall, nrounds=2000,params=xgb.paras, nthread = 6, verbose=0)
```

To get predictions on test set, we exponentiate to get back the price in dollars.

``` r
pred2.test = exp(predict(xgb.fitall, dtest))
```

To check the sanity of our result on the test set, we plot the distribution of value of salePrice in the training data (blue) and the distribution of predicted SalePrice for the test set using xgboost (green)

``` r
hist(log(tt$train$SalePrice), main="Training (blue), test (green) predictions", xlab = "SalePrice (log)", freq=FALSE, col=scales::alpha('blue',0.5), ylim=c(0,1.5))
hist(log(pred2.test), freq=FALSE, add=T, col=scales::alpha('green',.5))
```

![](/images/housing-prices-kaggle/unnamed-chunk-34-1.png)<!-- -->

This is just to quickly check the results before submitting, in order to see that we haven’t accidentally exponentiated the target an extra time or something else. As they seem similar enough, we then save the result and submit to Kaggle.

``` r
subm = cbind(test_id, pred2.test)
colnames(subm) = c("Id", "SalePrice")
write.csv(subm, "submission.csv", row.names=FALSE)
```

The best score I was able to get using this approach and these parameter values was 0.131 when submitting to the Kaggle leaderboard. Interestingly, a substantial improvement to the result from the Random Forest model, showing how in this case XGBoost is able to improve the prediction even further compared to RandomForests, when building models sequentially training on the errors of the previous models.

# Conclusion

At the end of this tutorial, we have made a submission to Kaggle that deals with a varied set of features related to housing prices, in which we had to deal with missing values and where we predicted the sale price using Random Forests and XGBoost.

To handle the missing values, we used the package missForest available in R which imputed the missing values with a Random Forest algorithm. Though not touched upon in this tutorial, my own casual observation is that missForest does improve accuracy of our final score compared to that of a simpler imputation (e.g. replacing all missing values with their means/mode), but at the cost of time-consuming computations, and my feeling is that in a production environment where new data is continuously added, one should use the simpler, less computationally-intensive way of imputation using the means/modes.

Moving on, we found that the best score on the test set was obtained with a XGBoost model, which placed us, at the time of writing, at the (upper) 30-40 percentiles of scores.

We attempted to reduce the number of features in our data set with recursive feature selection (RFE), where we only kept features that improved the out-of-bag (OOB) error of a recursively trained Random Forest model. However, as the end result of the RFE was a model with less accuracy on the validation set, we kept all variables for the final model. Though the RFE approach was unsuccessful and reduced the accuracy of the final model, the results presented here could provide some insight into the dataset, e.g. suggesting important variables.
