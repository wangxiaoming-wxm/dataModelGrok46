package claim

import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.util.Random

/**
 * Spark ML / Scala car-insurance claim model.
 *
 * Design constraints learned from local probes:
 *  - Objective must be RMSE (squared error), not logloss.
 *  - Dual worlds: cond_r/ratio vs rank/rate.
 *  - High-cardinality target encoding is a STANDALONE score; never feed it to GBT.
 *  - No id features. Fold-safe medians / quantiles / TE.
 */
object TrainApp {

  def main(args: Array[String]): Unit = {
    val dataDir = if (args.length > 0) args(0) else "/workspace/data"
    val outDir = if (args.length > 1) args(1) else "/workspace/submissions"
    val nFolds = if (args.length > 2) args(2).toInt else 5
    val seed = 2026

    val spark = SparkSession.builder()
      .appName("spark-claim")
      .master("local[4]")
      .config("spark.driver.memory", "10g")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val trainRaw = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(s"$dataDir/train.csv")
      .withColumn("label", col("label").cast(DoubleType))
    val testRaw = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(s"$dataDir/test.csv")

    val withFold = assignStratifiedFolds(trainRaw, nFolds, seed)
    withFold.cache()
    println(s"[info] train=${withFold.count()} test=${testRaw.count()} folds=$nFolds")

    var oof: DataFrame = null
    for (f <- 0 until nFolds) {
      println(s"[fold] $f / $nFolds")
      val tr = withFold.filter(col("fold") =!= f).drop("fold")
      val va = withFold.filter(col("fold") === f).drop("fold")
      val scored = scoreSplit(tr, va)
      oof = if (oof == null) scored else oof.unionByName(scored)
    }
    oof.cache()
    val nOof = oof.count()
    println(s"[info] oof rows=$nOof")

    val aucMain = auc(oof, "label", "pred_main")
    val aucAlt = auc(oof, "label", "pred_alt")
    val aucRf = auc(oof, "label", "pred_rf")
    val aucTe3 = auc(oof, "label", "te_src_cq_dq")
    val aucCqdq = auc(oof, "label", "te_cq_dq")
    val aucSrcRq = auc(oof, "label", "te_src_ratioq")
    println(f"[OOF] main=$aucMain%.5f alt=$aucAlt%.5f rf=$aucRf%.5f te3=$aucTe3%.5f cqdq=$aucCqdq%.5f src_ratioq=$aucSrcRq%.5f")

    val ranked = oof
      .withColumn("r_main", percentRankCol(oof, "pred_main"))
      .withColumn("r_alt", percentRankCol(oof, "pred_alt"))
      .withColumn("r_rf", percentRankCol(oof, "pred_rf"))
      .withColumn("r_te3", percentRankCol(oof, "te_src_cq_dq"))
      .withColumn("r_cqdq", percentRankCol(oof, "te_cq_dq"))
      .withColumn("r_srcrq", percentRankCol(oof, "te_src_ratioq"))

    val wMain = 0.34
    val wAlt = 0.22
    val wRf = 0.10
    val wTe3 = 0.16
    val wCq = 0.08
    val wRq = 0.10
    val blended = ranked.withColumn(
      "oof_blend",
      lit(wMain) * col("r_main") + lit(wAlt) * col("r_alt") + lit(wRf) * col("r_rf") +
        lit(wTe3) * col("r_te3") + lit(wCq) * col("r_cqdq") + lit(wRq) * col("r_srcrq")
    )
    val aucBlend = auc(blended, "label", "oof_blend")
    val aucW62 = auc(
      ranked.withColumn("w62", lit(0.62) * col("r_main") + lit(0.38) * col("r_alt")),
      "label",
      "w62"
    )
    println(f"[OOF] blend=$aucBlend%.5f w62_gbt=$aucW62%.5f")

    // Full-data fit for test
    println("[info] fitting full-data models")
    val fullScored = scoreSplit(trainRaw, testRaw.withColumn("label", lit(0.0)))
    val testRanked = fullScored
      .withColumn("r_main", percentRankCol(fullScored, "pred_main"))
      .withColumn("r_alt", percentRankCol(fullScored, "pred_alt"))
      .withColumn("r_rf", percentRankCol(fullScored, "pred_rf"))
      .withColumn("r_te3", percentRankCol(fullScored, "te_src_cq_dq"))
      .withColumn("r_cqdq", percentRankCol(fullScored, "te_cq_dq"))
      .withColumn("r_srcrq", percentRankCol(fullScored, "te_src_ratioq"))
      .withColumn(
        "label",
        lit(wMain) * col("r_main") + lit(wAlt) * col("r_alt") + lit(wRf) * col("r_rf") +
          lit(wTe3) * col("r_te3") + lit(wCq) * col("r_cqdq") + lit(wRq) * col("r_srcrq")
      )
      .select(col("id"), col("label"))

    import spark.implicits._
    new java.io.File(outDir).mkdirs()
    val outPath = s"$outDir/submission.csv"
    testRanked.coalesce(1).write.mode(SaveMode.Overwrite).option("header", "true").csv(outPath + "_dir")
    // flatten spark csv dir
    val part = new java.io.File(outPath + "_dir").listFiles().find(_.getName.startsWith("part")).get
    java.nio.file.Files.copy(
      part.toPath,
      java.nio.file.Paths.get(outPath),
      java.nio.file.StandardCopyOption.REPLACE_EXISTING
    )
    println(s"[info] wrote $outPath")

    val report =
      s"""sparkml_oof
         |n_folds=$nFolds
         |auc_main=$aucMain
         |auc_alt=$aucAlt
         |auc_rf=$aucRf
         |auc_te3=$aucTe3
         |auc_cqdq=$aucCqdq
         |auc_src_ratioq=$aucSrcRq
         |auc_w62_gbt=$aucW62
         |auc_blend=$aucBlend
         |""".stripMargin
    java.nio.file.Files.write(java.nio.file.Paths.get(s"$outDir/oof_report.txt"), report.getBytes("UTF-8"))
    println(report)
    spark.stop()
  }

  def scoreSplit(trainRaw: DataFrame, applyRaw: DataFrame): DataFrame = {
    val stats = FeatureEngine.fitStats(trainRaw)
    val trF = FeatureEngine.transform(trainRaw, stats, isTrain = true).cache()
    val vaF = FeatureEngine.transform(applyRaw, stats, isTrain = false)
    val vaTe = TargetEncode.fitTransformMany(trF, vaF, FeatureEngine.TeKeys)
    val teCols = FeatureEngine.TeKeys.map("te_" + _)
    val pMain = GbtTrainer.fitPredict(trF, vaTe, GbtTrainer.ArmMain)
    val pAlt = GbtTrainer.fitPredict(trF, vaTe, GbtTrainer.ArmAlt)
    val pRf = GbtTrainer.fitRf(trF, vaTe, GbtTrainer.ArmMain)
    val keep = (Seq("id") ++ teCols).distinct
    val a = pMain.select((keep ++ Seq("pred_main") ++ (if (pMain.columns.contains("label")) Seq("label") else Seq.empty)).map(col): _*)
    val b = pAlt.select(col("id"), col("pred_alt"))
    val c = pRf.select(col("id"), col("pred_rf"))
    val joined = a.join(b, Seq("id")).join(c, Seq("id"))
    trF.unpersist()
    joined
  }

  def assignStratifiedFolds(df: DataFrame, nFolds: Int, seed: Int): DataFrame = {
    val spark = df.sparkSession
    val rows = df.collect()
    val byLabel = rows.groupBy(_.getAs[Number]("label").doubleValue().toInt)
    val rnd = new Random(seed)
    val idFold = scala.collection.mutable.ArrayBuffer[(String, Int)]()
    byLabel.values.foreach { arr =>
      val shuf = rnd.shuffle(arr.toSeq)
      shuf.zipWithIndex.foreach { case (r, i) =>
        idFold += ((r.getAs[String]("id"), i % nFolds))
      }
    }
    val schema = StructType(Seq(StructField("id_f", StringType, nullable = false), StructField("fold", IntegerType, nullable = false)))
    val foldRows = idFold.map { case (id, f) => org.apache.spark.sql.Row(id, f) }
    val foldDf = spark.createDataFrame(spark.sparkContext.parallelize(foldRows, 1), schema)
    df.join(foldDf, df("id") === foldDf("id_f"), "inner").drop("id_f")
  }

  /** Mann-Whitney AUC via Spark (collect scores; n=15k is fine). */
  def auc(df: DataFrame, label: String, pred: String): Double = {
    val pairs = df.select(col(label).cast(DoubleType), col(pred).cast(DoubleType)).collect()
    val pos = pairs.filter(_.getDouble(0) > 0.5).map(_.getDouble(1))
    val neg = pairs.filter(_.getDouble(0) <= 0.5).map(_.getDouble(1))
    if (pos.isEmpty || neg.isEmpty) return 0.5
    val ps = pos.sorted
    val ns = neg.sorted
    var i = 0
    var j = 0
    var rankSum = 0.0
    val all = (ps.map(_ -> 1) ++ ns.map(_ -> 0)).sortBy(_._1)
    var rank = 1.0
    var k = 0
    while (k < all.length) {
      var t = k
      while (t + 1 < all.length && all(t + 1)._1 == all(k)._1) t += 1
      val avg = (rank + (rank + (t - k))) / 2.0
      var u = k
      while (u <= t) {
        if (all(u)._2 == 1) rankSum += avg
        u += 1
      }
      rank += (t - k + 1)
      k = t + 1
    }
    val n1 = pos.length.toDouble
    val n0 = neg.length.toDouble
    val u = rankSum - n1 * (n1 + 1.0) / 2.0
    u / (n1 * n0)
  }

  def percentRankCol(df: DataFrame, c: String): org.apache.spark.sql.Column = {
    // window percent_rank over all rows (global rank for this dataframe)
    val w = org.apache.spark.sql.expressions.Window.orderBy(col(c))
    percent_rank().over(w)
  }
}
