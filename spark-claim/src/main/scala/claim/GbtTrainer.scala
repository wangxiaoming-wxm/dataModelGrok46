package claim

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, RandomForestRegressor}
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

  def armMain(maxIter: Int): Spec = Spec(
    name = "main",
    numeric = FeatureEngine.NumericMain,
    cats = FeatureEngine.LowCardCatsMain,
    maxDepth = 5,
    maxIter = maxIter,
    stepSize = 0.05,
    minInstances = 80,
    subsample = 0.8,
    featureSubset = "0.85",
    seed = 2026L
  )

  def armAlt(maxIter: Int): Spec = Spec(
    name = "alt",
    numeric = FeatureEngine.NumericAlt,
    cats = FeatureEngine.LowCardCatsAlt,
    maxDepth = 6,
    maxIter = math.max(10, (maxIter * 1.1).toInt),
    stepSize = 0.04,
    minInstances = 70,
    subsample = 0.8,
    featureSubset = "0.3",
    seed = 2030L
  )

  def armBig(maxIter: Int): Spec = Spec(
    name = "big",
    numeric = FeatureEngine.NumericBig,
    cats = Seq("src", "reg", "age"),
    maxDepth = 4,
    maxIter = math.max(8, maxIter / 2),
    stepSize = 0.05,
    minInstances = 50,
    subsample = 0.75,
    featureSubset = "0.8",
    seed = 2040L
  )

  def fillNumeric(df: DataFrame, cols: Seq[String]): DataFrame = {
    FeatureEngine.sanitize(df, cols)
  }

  def fitPredict(train: DataFrame, applyDf: DataFrame, spec: Spec, labelCol: String = "label"): DataFrame = {
    fitPredictPair(train, applyDf, spec, labelCol)._2
  }

  def fitPredictPair(
      train: DataFrame,
      applyDf: DataFrame,
      spec: Spec,
      labelCol: String = "label"
  ): (DataFrame, DataFrame) = {
    val trainF = fillNumeric(train, spec.numeric)
    val applyF = fillNumeric(applyDf, spec.numeric)
    val tag = spec.name + "_" + spec.seed
    val catOut = spec.cats.map(_ + "_idx_" + tag)
    val indexers = spec.cats.zip(catOut).map { case (c, o) =>
      new StringIndexer()
        .setInputCol(c)
        .setOutputCol(o)
        .setHandleInvalid("keep")
    }
    val assembler = new VectorAssembler()
      .setInputCols((spec.numeric ++ catOut).toArray)
      .setOutputCol("raw_features_" + tag)
      .setHandleInvalid("keep")

    // No VectorIndexer: maxCategories that are too large recode noisy continuous cols as cats.
    val gbt = new GBTRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("raw_features_" + tag)
      .setPredictionCol("pred_" + spec.name)
      .setLossType("squared")
      .setMaxDepth(spec.maxDepth)
      .setMaxIter(spec.maxIter)
      .setStepSize(spec.stepSize)
      .setMinInstancesPerNode(spec.minInstances)
      .setSubsamplingRate(spec.subsample)
      .setFeatureSubsetStrategy(spec.featureSubset)
      .setMaxBins(64)
      .setSeed(spec.seed)
      .setCacheNodeIds(true)
      .setMaxMemoryInMB(512)

    val pipe = new Pipeline().setStages((indexers :+ assembler :+ gbt).toArray)
    val model = pipe.fit(trainF)
    val predCol = "pred_" + spec.name
    def slim(df: DataFrame): DataFrame =
      model.transform(df)
        .withColumn(predCol, FeatureEngine.finiteFill(col(predCol), 0.1))
        .select(col("id"), col(predCol))
    (slim(trainF), slim(applyF))
  }

  def averageIdPred(dfs: Seq[DataFrame], predCol: String): DataFrame = {
    val tagged = dfs.zipWithIndex.map { case (d, i) =>
      d.select(col("id"), col(predCol).as(s"bag_$i"))
    }
    var acc = tagged.head
    tagged.tail.foreach { d => acc = acc.join(d, Seq("id")) }
    val n = dfs.size.toDouble
    val avg = dfs.indices.map(i => col(s"bag_$i")).reduce(_ + _) / lit(n)
    acc.withColumn(predCol, avg).select(col("id"), col(predCol))
  }

  /** Average several GBT seeds / subsample rates. Returns (trainPred, applyPred) with id + pred_{name}. */
  def fitBaggedPair(train: DataFrame, applyDf: DataFrame, spec: Spec, nBags: Int): (DataFrame, DataFrame) = {
    val bags = bagSpecs(spec, nBags)
    val predCol = "pred_" + spec.name
    val parts = bags.zipWithIndex.map { case (s, i) =>
      val t0 = System.nanoTime()
      val pair = fitPredictPair(train, applyDf, s)
      println(f"[gbt] ${spec.name} bag=$i seed=${s.seed} sub=${s.subsample}%.2f iter=${s.maxIter} ${(System.nanoTime()-t0)/1e9}%.1fs")
      pair
    }
    (averageIdPred(parts.map(_._1), predCol), averageIdPred(parts.map(_._2), predCol))
  }

  def fitBagged(train: DataFrame, applyDf: DataFrame, spec: Spec, nBags: Int): DataFrame =
    fitBaggedPair(train, applyDf, spec, nBags)._2

  def bagSpecs(spec: Spec, nBags: Int): Seq[Spec] = {
    val n = math.max(1, nBags)
    (0 until n).map { i =>
      val sub = math.max(0.55, spec.subsample - 0.10 * i)
      spec.copy(seed = spec.seed + i * 17L, subsample = sub)
    }
  }

  def fitRf(train: DataFrame, applyDf: DataFrame, spec: Spec, numTrees: Int): DataFrame = {
    val trainF = fillNumeric(train, spec.numeric)
    val applyF = fillNumeric(applyDf, spec.numeric)
    val catOut = spec.cats.map(_ + "_idx_rf")
    val indexers = spec.cats.zip(catOut).map { case (c, o) =>
      new StringIndexer().setInputCol(c).setOutputCol(o).setHandleInvalid("keep")
    }
    val assembler = new VectorAssembler()
      .setInputCols((spec.numeric ++ catOut).toArray)
      .setOutputCol("features_rf")
      .setHandleInvalid("keep")
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features_rf")
      .setPredictionCol("pred_rf")
      .setNumTrees(numTrees)
      .setMaxDepth(6)
      .setMinInstancesPerNode(40)
      .setSubsamplingRate(0.8)
      .setFeatureSubsetStrategy("sqrt")
      .setMaxBins(64)
      .setSeed(spec.seed + 99)
    val t0 = System.nanoTime()
    val out = new Pipeline().setStages((indexers :+ assembler :+ rf).toArray).fit(trainF).transform(applyF)
    println(f"[rf] trees=$numTrees ${(System.nanoTime()-t0)/1e9}%.1fs")
    out.withColumn("pred_rf", FeatureEngine.finiteFill(col("pred_rf"), 0.1))
      .select(col("id"), col("pred_rf"))
  }

  /**
   * Residual GBT on CAR_0/1/2 only. pred_big is the residual for large cars, 0 otherwise.
   * Small blend weight — captures source-specific condition shape the global model misses.
   */
  def fitBigCarResidual(
      train: DataFrame,
      applyDf: DataFrame,
      trainMain: DataFrame,
      spec: Spec
  ): DataFrame = {
    val large = FeatureEngine.LargeCars
    val tr = train.join(trainMain.select(col("id"), col("pred_main").as("pm")), Seq("id"))
      .withColumn("resid", col("label") - col("pm"))
      .filter(col("car").isin(large: _*))
    val nBig = tr.count()
    if (nBig < 200) {
      println(s"[big] skip n=$nBig")
      return applyDf.select(col("id")).withColumn("pred_big", lit(0.0))
    }
    val t0 = System.nanoTime()
    val scored = fitPredict(tr, applyDf, spec, labelCol = "resid")
      .join(applyDf.select(col("id"), col("car")), Seq("id"))
    println(f"[big] n=$nBig ${(System.nanoTime()-t0)/1e9}%.1fs")
    scored
      .withColumn(
        "pred_big",
        when(col("car").isin(large: _*), FeatureEngine.finiteFill(col("pred_big"), 0.0)).otherwise(lit(0.0))
      )
      .select(col("id"), col("pred_big"))
  }
}
