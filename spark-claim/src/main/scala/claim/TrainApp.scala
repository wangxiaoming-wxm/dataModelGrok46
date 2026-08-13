package claim

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import java.io.{File, PrintWriter}
import java.nio.charset.StandardCharsets
import scala.util.Random

/**
 * Spark ML / Scala car-insurance claim model.
 *
 * Design constraints learned from local probes:
 *  - Objective must be RMSE (squared error), not logloss.
 *  - Dual worlds: cond_r/ratio vs rank/rate.
 *  - High-cardinality target encoding is a STANDALONE score; never feed it to GBT.
 *  - No id features. Fold-safe medians / quantiles / TE.
 *  - Rank fusion weights are chosen from a pre-registered list (no fine grid).
 */
object TrainApp {

  def main(args: Array[String]): Unit = {
    sys.props("SPARK_LOCAL_IP") = "127.0.0.1"
    sys.props("spark.driver.host") = "127.0.0.1"

    val dataDir = if (args.length > 0) args(0) else "/workspace/data"
    val outDir = if (args.length > 1) args(1) else "/workspace/submissions"
    val nFolds = if (args.length > 2) args(2).toInt else 10
    val quick = args.length > 3 && args(3).toLowerCase.contains("quick")
    val seed = 2026
    val nBags = if (quick) 1 else 2
    val gbtIter = if (quick) 12 else 80
    val rfTrees = if (quick) 16 else 48
    val teMs = TargetEncode.Ms

    val spark = SparkSession.builder()
      .appName("spark-claim")
      .master("local[4]")
      .config("spark.driver.memory", "10g")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.sql.adaptive.enabled", "true")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val trainRaw = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(s"$dataDir/train.csv")
      .withColumn("label", col("label").cast(DoubleType))
    val testRaw = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(s"$dataDir/test.csv")
    val testIds = readIds(s"$dataDir/test.csv")

    val withFold = assignStratifiedFolds(trainRaw, nFolds, seed)
    withFold.cache()
    println(s"[info] train=${withFold.count()} test=${testRaw.count()} folds=$nFolds bags=$nBags gbtIter=$gbtIter rf=$rfTrees quick=$quick")

    var oof: DataFrame = null
    for (f <- 0 until nFolds) {
      println(s"[fold] $f / $nFolds")
      val tr = withFold.filter(col("fold") =!= f).drop("fold")
      val va = withFold.filter(col("fold") === f).drop("fold")
      val scored = scoreSplit(tr, va, nBags, gbtIter, rfTrees, teMs, withTrainMain = true)
      oof = if (oof == null) scored else oof.unionByName(scored, allowMissingColumns = true)
    }
    oof = oof.cache()
    val nOof = oof.count()
    println(s"[info] oof rows=$nOof")

    val reportBuf = new StringBuilder

    def logAuc(name: String, c: String): Double = {
      if (!oof.columns.contains(c)) {
        println(s"[OOF] $name missing")
        return 0.5
      }
      val a = auc(oof, "label", c)
      println(f"[OOF] $name=$a%.5f")
      reportBuf.append(f"auc_$name=$a%.5f%n")
      a
    }

    val aucMain = logAuc("main", "pred_main")
    val aucAlt = logAuc("alt", "pred_alt")
    val aucRf = logAuc("rf", "pred_rf")
    val aucLr = logAuc("lr", "pred_lr")
    val aucFm = logAuc("fm", "pred_fm")
    val aucBig = logAuc("big", "pred_big")
    val aucPool = logAuc("te_pool", "te_pool")
    val aucCbMain = logAuc("cb_main", "pred_cb_main")
    val aucCbAlt = logAuc("cb_alt", "pred_cb_alt")

    val bestM = scala.collection.mutable.LinkedHashMap.empty[String, Double]
    FeatureEngine.TeKeys.foreach { k =>
      var best = 20.0
      var bestA = -1.0
      teMs.foreach { m =>
        val c = TargetEncode.colName(k, m)
        if (oof.columns.contains(c)) {
          val a = auc(oof, "label", c)
          println(f"[OOF] te $k m=${m.toInt} $a%.5f")
          reportBuf.append(f"auc_te_${k}_m${m.toInt}=$a%.5f%n")
          if (a > bestA) { bestA = a; best = m }
        }
      }
      bestM(k) = best
      println(s"[te] pick $k m=${best.toInt} auc=$bestA")
      reportBuf.append(s"pick_te_${k}_m=${best.toInt}\n")
    }

    var ranked = oof
    val rankSources = Seq(
      "pred_main", "pred_alt", "pred_rf", "pred_lr", "pred_fm", "pred_big", "te_pool",
      "pred_cb_main", "pred_cb_alt"
    ) ++ FeatureEngine.TeKeys.map(k => TargetEncode.colName(k, bestM(k)))
    rankSources.distinct.filter(ranked.columns.contains).foreach { c =>
      ranked = withPercentRank(ranked, c, "r__" + c)
    }
    ranked = ranked.cache()
    ranked.count()

    val recipes = Seq(
      "w62" -> Map("pred_main" -> 0.62, "pred_alt" -> 0.38),
      "equal_gbt" -> Map("pred_main" -> 0.5, "pred_alt" -> 0.5),
      "classic6" -> Map(
        "pred_main" -> 0.34, "pred_alt" -> 0.22, "pred_rf" -> 0.10,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 0.16,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 0.08,
        TargetEncode.colName("src_ratioq", bestM("src_ratioq")) -> 0.10
      ),
      "equal_core" -> Map(
        "pred_main" -> 1.0, "pred_alt" -> 1.0, "pred_rf" -> 1.0,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 1.0,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 1.0,
        TargetEncode.colName("src_ratioq", bestM("src_ratioq")) -> 1.0
      ),
      "with_glm" -> Map(
        "pred_main" -> 0.28, "pred_alt" -> 0.18, "pred_rf" -> 0.08,
        "pred_lr" -> 0.12, "pred_fm" -> 0.08,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 0.14,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 0.06,
        TargetEncode.colName("src_ratioq", bestM("src_ratioq")) -> 0.06
      ),
      "glm_te" -> Map(
        "pred_main" -> 0.30, "pred_alt" -> 0.18,
        "pred_lr" -> 0.14, "pred_fm" -> 0.08,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 0.14,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 0.08,
        TargetEncode.colName("src_ratioq", bestM("src_ratioq")) -> 0.08
      ),
      "te_pool_w" -> Map(
        "pred_main" -> 0.32, "pred_alt" -> 0.20, "pred_rf" -> 0.08,
        "te_pool" -> 0.22, "pred_lr" -> 0.10, "pred_big" -> 0.08
      ),
      "shape" -> Map(
        "pred_main" -> 0.30, "pred_alt" -> 0.18, "pred_lr" -> 0.10,
        TargetEncode.colName("src_cq", bestM("src_cq")) -> 0.14,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 0.16,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 0.12
      ),
      "cb_w62" -> Map("pred_cb_main" -> 0.62, "pred_cb_alt" -> 0.38),
      "cb_te" -> Map(
        "pred_cb_main" -> 0.40, "pred_cb_alt" -> 0.24, "pred_main" -> 0.08,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 0.14,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 0.08,
        TargetEncode.colName("src_ratioq", bestM("src_ratioq")) -> 0.06
      ),
      "cb_max" -> Map(
        "pred_cb_main" -> 0.32, "pred_cb_alt" -> 0.20,
        "pred_main" -> 0.10, "pred_alt" -> 0.08, "pred_rf" -> 0.04,
        TargetEncode.colName("src_cq_dq", bestM("src_cq_dq")) -> 0.14,
        TargetEncode.colName("cq_dq", bestM("cq_dq")) -> 0.06,
        TargetEncode.colName("src_ratioq", bestM("src_ratioq")) -> 0.06
      )
    )

    var bestName = "classic6"
    var bestBlend = 0.0
    var bestWeights = recipes.find(_._1 == "classic6").get._2
    recipes.foreach { case (name, w) =>
      val a = aucBlend(ranked, w)
      println(f"[OOF] blend $name=$a%.5f")
      reportBuf.append(f"blend_$name=$a%.5f%n")
      if (a > bestBlend) {
        bestBlend = a
        bestName = name
        bestWeights = w
      }
    }
    println(f"[OOF] selected $bestName=$bestBlend%.5f")
    reportBuf.append(s"selected=$bestName\n")
    reportBuf.append(f"auc_blend=$bestBlend%.5f%n")

    println("[info] fitting full-data models")
    val fullScored = scoreSplit(
      trainRaw,
      testRaw.withColumn("label", lit(0.0)),
      nBags, gbtIter, rfTrees, teMs,
      withTrainMain = true
    )
    var testRanked = fullScored
    rankSources.distinct.filter(testRanked.columns.contains).foreach { c =>
      testRanked = withPercentRank(testRanked, c, "r__" + c)
    }
    testRanked = applyBlend(testRanked, bestWeights).withColumnRenamed("blend", "label")

    val predMap = testRanked.select("id", "label").collect().map { r =>
      r.getString(0) -> (if (r.isNullAt(1) || r.getDouble(1).isNaN) 0.5 else r.getDouble(1))
    }.toMap
    new File(outDir).mkdirs()
    val outPath = s"$outDir/submission.csv"
    val pw = new PrintWriter(new File(outPath), StandardCharsets.UTF_8.name())
    try {
      pw.println("id,label")
      testIds.foreach { id =>
        val v = predMap.getOrElse(id, 0.5)
        pw.println(f"$id,$v%.10f")
      }
    } finally pw.close()
    println(s"[info] wrote $outPath rows=${testIds.length} mapped=${predMap.size}")

    val header =
      s"""sparkml_oof
         |n_folds=$nFolds
         |n_bags=$nBags
         |gbt_iter=$gbtIter
         |rf_trees=$rfTrees
         |quick=$quick
         |auc_main=$aucMain
         |auc_alt=$aucAlt
         |auc_rf=$aucRf
         |auc_lr=$aucLr
         |auc_fm=$aucFm
         |auc_big=$aucBig
         |auc_te_pool=$aucPool
         |auc_cb_main=$aucCbMain
         |auc_cb_alt=$aucCbAlt
         |selected=$bestName
         |auc_blend=$bestBlend
         |""".stripMargin
    val report = header + reportBuf.toString
    java.nio.file.Files.write(java.nio.file.Paths.get(s"$outDir/oof_report.txt"), report.getBytes(StandardCharsets.UTF_8))
    println(report)
    spark.stop()
  }

  def scoreSplit(
      trainRaw: DataFrame,
      applyRaw: DataFrame,
      nBags: Int,
      gbtIter: Int,
      rfTrees: Int,
      teMs: Seq[Double],
      withTrainMain: Boolean
  ): DataFrame = {
    val stats = FeatureEngine.fitStats(trainRaw)
    val trF = FeatureEngine.transform(trainRaw, stats, isTrain = true).cache()
    trF.count()
    val vaF = FeatureEngine.transform(applyRaw, stats, isTrain = false).cache()
    vaF.count()
    val nq = vaF.groupBy("days_q").count().count()
    val cq = vaF.groupBy("cond_q").count().count()
    println(s"[debug] days_q bins=$nq cond_q bins=$cq src_cq_dq n=${vaF.select("src_cq_dq").distinct().count()}")

    val vaTe = TargetEncode.fitTransformMany(trF, vaF, FeatureEngine.TeKeys, teMs)
    val mainSpec = GbtTrainer.armMain(gbtIter)
    val altSpec = GbtTrainer.armAlt(gbtIter)
    val (trMain, pMain) = GbtTrainer.fitBaggedPair(trF, vaF, mainSpec, nBags)
    val pAlt = GbtTrainer.fitBagged(trF, vaF, altSpec, nBags)
    val pRf = GbtTrainer.fitRf(trF, vaF, mainSpec, rfTrees)
    val pLr = LinearArms.fitLr(trF, vaF, FeatureEngine.NumericMain, FeatureEngine.GlmLowCats, FeatureEngine.GlmCrossCats)
    val pFm = LinearArms.fitFm(trF, vaF, FeatureEngine.NumericAlt, FeatureEngine.GlmLowCats)
    val pBig =
      if (withTrainMain) GbtTrainer.fitBigCarResidual(trF, vaF, trMain, GbtTrainer.armBig(gbtIter))
      else vaF.select(col("id")).withColumn("pred_big", lit(0.0))

    val keepTe = FeatureEngine.TeKeys.flatMap(k => teMs.map(m => TargetEncode.colName(k, m))) ++ Seq("te_pool")
    val teKeep = (Seq("id") ++ keepTe.filter(vaTe.columns.contains)).distinct
    val hasLabel = vaTe.columns.contains("label")
    val baseCols = if (hasLabel) teKeep :+ "label" else teKeep
    var joined = vaTe.select(baseCols.map(col): _*)
      .join(pMain, Seq("id"))
      .join(pAlt, Seq("id"))
      .join(pRf, Seq("id"))
      .join(pLr, Seq("id"))
      .join(pFm, Seq("id"))
      .join(pBig, Seq("id"))

    val pCb =
      if (sys.env.getOrElse("CLAIM_CB_BACKEND", "auto") == "off") {
        vaF.select(col("id"))
          .withColumn("pred_cb_main", lit(0.1))
          .withColumn("pred_cb_alt", lit(0.1))
      } else {
        try {
          CatBoostArm.score(trF, vaTe)
        } catch {
          case e: OutOfMemoryError => throw e
          case e: Throwable =>
            println(s"[cb] skipped: ${e.getClass.getSimpleName}: ${e.getMessage}")
            vaF.select(col("id"))
              .withColumn("pred_cb_main", lit(0.1))
              .withColumn("pred_cb_alt", lit(0.1))
        }
      }
    joined = joined.join(pCb, Seq("id"))

    trF.unpersist()
    vaF.unpersist()
    joined
  }

  def assignStratifiedFolds(df: DataFrame, nFolds: Int, seed: Int): DataFrame = {
    val spark = df.sparkSession
    val rows = df.select(col("id").cast(StringType), col("label").cast(DoubleType)).collect()
    val byLabel = rows.groupBy(r => if (r.getDouble(1) > 0.5) 1 else 0)
    val rnd = new Random(seed)
    val idFold = scala.collection.mutable.ArrayBuffer[(String, Int)]()
    byLabel.values.foreach { arr =>
      val shuf = rnd.shuffle(arr.toSeq)
      shuf.zipWithIndex.foreach { case (r, i) =>
        idFold += ((r.getString(0), i % nFolds))
      }
    }
    val schema = StructType(Seq(StructField("id_f", StringType, nullable = false), StructField("fold", IntegerType, nullable = false)))
    val foldRows = idFold.map { case (id, f) => Row(id, f) }
    val foldDf = spark.createDataFrame(spark.sparkContext.parallelize(foldRows, 1), schema)
    df.join(foldDf, df("id") === foldDf("id_f"), "inner").drop("id_f")
  }

  /** Mann-Whitney AUC via Spark (collect scores; n=15k is fine). */
  def auc(df: DataFrame, label: String, pred: String): Double = {
    val pairs = df.select(col(label).cast(DoubleType), col(pred).cast(DoubleType)).collect().map { r =>
      val y = if (r.isNullAt(0)) 0.0 else r.getDouble(0)
      val p =
        if (r.isNullAt(1)) Double.NaN
        else r.getDouble(1)
      (y, p)
    }
    val pos = pairs.filter(_._1 > 0.5).map(_._2).filter(p => !p.isNaN && !p.isInfinity)
    val neg = pairs.filter(_._1 <= 0.5).map(_._2).filter(p => !p.isNaN && !p.isInfinity)
    if (pos.isEmpty || neg.isEmpty) return 0.5
    val all = (pos.map(_ -> 1) ++ neg.map(_ -> 0)).sortBy(_._1)
    var rank = 1.0
    var rankSum = 0.0
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

  /**
   * Average-rank percent in [0,1] computed on the driver for the given dataframe only.
   * Test ranks are computed on test scores alone (not mixed with OOF) — correct for rank fusion.
   */
  def withPercentRank(df: DataFrame, inCol: String, outCol: String): DataFrame = {
    val spark = df.sparkSession
    val recs = df.select(col("id").cast(StringType), col(inCol).cast(DoubleType)).collect()
    val n = recs.length
    val vals = recs.map { r =>
      if (r.isNullAt(1)) Double.NaN else r.getDouble(1)
    }
    val ranks = percentRanks(vals)
    val schema = StructType(Seq(StructField("id", StringType, nullable = false), StructField(outCol, DoubleType, nullable = false)))
    val rows = recs.indices.map { i => Row(recs(i).getString(0), ranks(i)) }
    val rankDf = spark.createDataFrame(spark.sparkContext.parallelize(rows, 1), schema)
    df.join(rankDf, Seq("id"))
  }

  def percentRanks(values: Array[Double]): Array[Double] = {
    val n = values.length
    if (n <= 1) return Array.fill(n)(0.5)
    val indexed = values.zipWithIndex.sortBy { case (v, _) =>
      if (v.isNaN || v.isInfinity) Double.NegativeInfinity else v
    }
    val out = Array.fill(n)(0.5)
    var i = 0
    while (i < n) {
      var j = i
      val vi = indexed(i)._1
      while (j + 1 < n && indexed(j + 1)._1 == vi) j += 1
      val avg = (i + j) / 2.0
      val pct = avg / (n - 1.0)
      var k = i
      while (k <= j) {
        out(indexed(k)._2) = pct
        k += 1
      }
      i = j + 1
    }
    out
  }

  def applyBlend(df: DataFrame, weights: Map[String, Double]): DataFrame = {
    val parts = weights.toSeq.flatMap { case (src, w) =>
      val rc = "r__" + src
      if (df.columns.contains(rc)) Some(lit(w) * col(rc)) else None
    }
    val denom = weights.toSeq.collect { case (src, w) if df.columns.contains("r__" + src) => w }.sum
    val expr =
      if (parts.isEmpty || denom == 0) lit(0.5)
      else parts.reduce(_ + _) / lit(denom)
    df.withColumn("blend", expr)
  }

  def aucBlend(df: DataFrame, weights: Map[String, Double]): Double = {
    auc(applyBlend(df, weights), "label", "blend")
  }

  def readIds(path: String): Array[String] = {
    val src = scala.io.Source.fromFile(path)
    try {
      val it = src.getLines()
      it.next()
      it.map { line =>
        val c = line.indexOf(',')
        if (c < 0) line else line.substring(0, c)
      }.toArray
    } finally src.close()
  }
}
