package claim

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor, RandomForestRegressor}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object GbtTrainer {

  case class Spec(
      name: String,
      numeric: Seq[String],
      cats: Seq[String],
      maxDepth: Int,
      maxIter: Int,
      stepSize: Double,
      minInstances: Int,
      subsample: Double,
      featureSubset: String,
      seed: Long
  )

  val ArmMain: Spec = Spec(
    name = "main",
    numeric = FeatureEngine.NumericMain,
    cats = FeatureEngine.LowCardCats,
    maxDepth = 5,
    maxIter = 80,
    stepSize = 0.05,
    minInstances = 80,
    subsample = 0.8,
    featureSubset = "0.85",
    seed = 2026L
  )

  val ArmAlt: Spec = Spec(
    name = "alt",
    numeric = FeatureEngine.NumericAlt,
    cats = FeatureEngine.LowCardCats,
    maxDepth = 6,
    maxIter = 90,
    stepSize = 0.04,
    minInstances = 70,
    subsample = 0.8,
    featureSubset = "0.3",
    seed = 2030L
  )

  def fitPredict(train: DataFrame, applyDf: DataFrame, spec: Spec, labelCol: String = "label"): DataFrame = {
    val trainF = fillNumeric(train, spec.numeric)
    val applyF = fillNumeric(applyDf, spec.numeric)
    val catOut = spec.cats.map(_ + "_idx_" + spec.name)
    val indexers = spec.cats.zip(catOut).map { case (c, o) =>
      new StringIndexer()
        .setInputCol(c)
        .setOutputCol(o)
        .setHandleInvalid("keep")
    }
    val assembleCols = spec.numeric ++ catOut
    val assembler = new VectorAssembler()
      .setInputCols(assembleCols.toArray)
      .setOutputCol("raw_features_" + spec.name)
      .setHandleInvalid("keep")

    val vecIndexer = new VectorIndexer()
      .setInputCol("raw_features_" + spec.name)
      .setOutputCol("features_" + spec.name)
      .setMaxCategories(64)
      .setHandleInvalid("keep")

    val gbt = new GBTRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("features_" + spec.name)
      .setPredictionCol("pred_" + spec.name)
      .setMaxDepth(spec.maxDepth)
      .setMaxIter(spec.maxIter)
      .setStepSize(spec.stepSize)
      .setMinInstancesPerNode(spec.minInstances)
      .setSubsamplingRate(spec.subsample)
      .setFeatureSubsetStrategy(spec.featureSubset)
      .setMaxBins(128)
      .setSeed(spec.seed)
      .setCacheNodeIds(true)

    val stages = indexers :+ assembler :+ vecIndexer :+ gbt
    val pipe = new Pipeline().setStages(stages.toArray)
    val model = pipe.fit(trainF)
    model.transform(applyF)
  }

  def fillNumeric(df: DataFrame, cols: Seq[String]): DataFrame = {
    cols.foldLeft(df) { (acc, c) =>
      if (acc.columns.contains(c)) acc.withColumn(c, coalesce(col(c).cast("double"), lit(0.0)))
      else acc.withColumn(c, lit(0.0))
    }
  }

  def fitRf(train: DataFrame, applyDf: DataFrame, spec: Spec, labelCol: String = "label"): DataFrame = {
    val trainF = fillNumeric(train, spec.numeric)
    val applyF = fillNumeric(applyDf, spec.numeric)
    val catOut = spec.cats.map(_ + "_idx_rf")
    val indexers = spec.cats.zip(catOut).map { case (c, o) =>
      new StringIndexer().setInputCol(c).setOutputCol(o).setHandleInvalid("keep")
    }
    val assembler = new VectorAssembler()
      .setInputCols((spec.numeric ++ catOut).toArray)
      .setOutputCol("features")
      .setHandleInvalid("keep")
    val rf = new RandomForestRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("features")
      .setPredictionCol("pred_rf")
      .setNumTrees(200)
      .setMaxDepth(6)
      .setMinInstancesPerNode(40)
      .setSubsamplingRate(0.8)
      .setFeatureSubsetStrategy("sqrt")
      .setMaxBins(64)
      .setSeed(spec.seed + 99)
    val pipe = new Pipeline().setStages((indexers :+ assembler :+ rf).toArray)
    pipe.fit(trainF).transform(applyF)
  }
}
