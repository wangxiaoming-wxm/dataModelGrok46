package claim

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NominalAttribute, NumericAttribute}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import java.io.{File, PrintWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.time.Duration

import scala.sys.process._
import scala.util.control.NonFatal

/**
 * Dual-arm CatBoost (Spark ML Estimator) with fallbacks:
 *   A) ai.catboost CatBoostRegressor (Plain; Spark has no Ordered boosting)
 *   B) Python catboost 1.2.10 (Ordered+Plain, RMSE) via cb_fit_one.py
 *   C) Ordered-TE expanding-mean + Spark GBTRegressor
 *
 * Never uses src_cq_dq as a tree/CatBoost categorical.
 * Outputs pred_cb_main / pred_cb_alt.
 */
object CatBoostArm {

  val NumMain: Seq[String] = Seq(
    "days", "days_log", "condition_f", "cond_r", "ratio", "ratio_sqrt",
    "u_shape", "inv_cond", "age_range", "V", "cc", "x20", "x1", "x5",
    "max_g", "x14", "x17", "age8", "cond_low", "cond_miss",
    "w_safe750", "w_safe1750", "w_hot1950", "w_new50", "t3_num",
    "v_r", "cc_r", "ushape_car10", "mono_car1",
    "claim_util", "deny", "anniv_dist", "dump_poor"
  )

  val NumAlt: Seq[String] = Seq(
    "days", "days_log", "condition_f", "cond_rk", "rate", "u_shape",
    "age_range", "V", "cc", "x20", "x1", "x5", "age8", "cond_low", "cond_miss",
    "w_safe750", "w_safe1750", "w_hot1950", "w_new50", "t3_num",
    "ushape_car10", "mono_car1",
    "claim_util", "deny", "anniv_dist"
  )

  /** Medium-card cats. NO src_cq_dq. */
  val Cats: Seq[String] = Seq(
    "src", "reg", "age", "grades_s", "month_s",
    "src_reg", "src_age", "reg_age", "src_cq",
    "days_q", "cond_q", "src_dq", "reg_cq", "reg_dq"
  )

  val LowCats: Seq[String] = Seq("src", "reg", "age", "grades_s", "month_s", "days_q", "cond_q")

  @volatile private var sparkNativeOk: Option[Boolean] = None
  @volatile private var pythonOk: Option[Boolean] = None

  def score(train: DataFrame, apply: DataFrame): DataFrame = {
    val tr = withWindows(train)
    val ap = withWindows(apply)
    joinTeacher(ap).getOrElse {
      val backend = sys.env.getOrElse("CLAIM_CB_BACKEND", "auto")
      backend match {
        case "spark"    => sparkDual(tr, ap)
        case "python"   => pythonDual(tr, ap)
        case "ordered"  => orderedDual(tr, ap)
        case _          => autoDual(tr, ap)
      }
    }
  }

  def autoDual(train: DataFrame, apply: DataFrame): DataFrame = {
    if (sparkNativeOk.contains(true) || (sparkNativeOk.isEmpty && sparkAvailable)) {
      try {
        val out = sparkDual(train, apply)
        sparkNativeOk = Some(true)
        return out
      } catch {
        case e: OutOfMemoryError => throw e
        case e: Throwable =>
          sparkNativeOk = Some(false)
          println(s"[cb] spark native failed (${e.getClass.getSimpleName}: ${e.getMessage}); falling back")
      }
    } else sparkNativeOk = Some(false)

    if (pythonAvailable) {
      try return pythonDual(train, apply)
      catch {
        case e: OutOfMemoryError => throw e
        case NonFatal(e) =>
          pythonOk = Some(false)
          println(s"[cb] python catboost failed (${e.getClass.getSimpleName}: ${e.getMessage}); ordered-TE GBT")
      }
    }
    orderedDual(train, apply)
  }

  def withWindows(df: DataFrame): DataFrame = {
    var d = df
    if (!d.columns.contains("w_hot1950"))
      d = d.withColumn("w_hot1950", ((col("days") >= 1950.0 && col("days") < 2000.0).cast(DoubleType)))
    if (!d.columns.contains("w_new50"))
      d = d.withColumn("w_new50", (col("days") < 50.0).cast(DoubleType))
    if (!d.columns.contains("w_safe750"))
      d = d.withColumn("w_safe750", ((col("days") >= 700.0 && col("days") < 880.0).cast(DoubleType)))
    if (!d.columns.contains("w_safe1750"))
      d = d.withColumn("w_safe1750", ((col("days") >= 1725.0 && col("days") < 1825.0).cast(DoubleType)))
    if (!d.columns.contains("deny") && d.columns.contains("w_safe750") && d.columns.contains("w_safe1750"))
      d = d.withColumn("deny", ((col("w_safe750") + col("w_safe1750")) > lit(0.5)).cast(DoubleType))
    if (!d.columns.contains("claim_util") && d.columns.contains("rate") && d.columns.contains("deny"))
      d = d.withColumn("claim_util", col("rate") * (lit(1.0) - col("deny")))
    if (!d.columns.contains("dump_poor") && d.columns.contains("w_hot9370") && d.columns.contains("cond_rk"))
      d = d.withColumn("dump_poor", col("w_hot9370") * (lit(1.0) - col("cond_rk")))
    if (!d.columns.contains("anniv_dist") && d.columns.contains("days"))
      d = d.withColumn(
        "anniv_dist",
        least(
          abs(col("days") - lit(365.0)), abs(col("days") - lit(730.0)),
          abs(col("days") - lit(1095.0)), abs(col("days") - lit(1460.0)),
          abs(col("days") - lit(1825.0)), abs(col("days") - lit(2190.0)),
          abs(col("days") - lit(2555.0)), abs(col("days") - lit(2920.0))
        )
      )
    d
  }

  private def sparkAvailable: Boolean = {
    try {
      Class.forName("ai.catboost.spark.CatBoostRegressor")
      true
    } catch {
      case _: Throwable => false
    }
  }

  private def pythonAvailable: Boolean = pythonOk match {
    case Some(v) => v
    case None =>
      val script = fitScript
      pythonOk = Some(script.isFile && (new File("/usr/bin/python3").isFile || commandExists("python3")))
      pythonOk.get
  }

  private def commandExists(c: String): Boolean =
    try { Seq("bash", "-lc", s"command -v $c").! == 0 } catch { case _: Throwable => false }

  private def fitScript: File = {
    val cands = Seq(
      "/workspace/analysis/cb_fit_one.py",
      new File(new File(".").getAbsolutePath).getParentFile + "/analysis/cb_fit_one.py",
      "../analysis/cb_fit_one.py"
    ).map(new File(_))
    cands.find(_.isFile).getOrElse(new File("/workspace/analysis/cb_fit_one.py"))
  }

  def joinTeacher(apply: DataFrame): Option[DataFrame] = {
    val spark = apply.sparkSession
    val packs = Seq(
      ("/workspace/submissions/cb_w62_oof.parquet", "/workspace/submissions/cb_w62_test.parquet", "/workspace/submissions/cb_w62_metrics.json"),
      ("/workspace/submissions/cb_teacher_oof.parquet", "/workspace/submissions/cb_teacher_test.parquet", "/workspace/submissions/cb_teacher_metrics.json")
    )
    def cover(f: File): Option[DataFrame] = {
      if (!f.isFile) return None
      val t = spark.read.parquet(f.getAbsolutePath)
        .select(col("id").cast(StringType).as("id"), col("pred_cb_main"), col("pred_cb_alt"))
      val nAp = apply.select("id").distinct().count()
      val nHit = apply.select(col("id").cast(StringType).as("id")).join(t, Seq("id"), "inner").count()
      if (nHit == nAp && nAp > 0)
        Some(apply.select(col("id")).join(t, Seq("id"), "left"))
      else None
    }
    var best: Option[(DataFrame, String, Double)] = None
    packs.foreach { case (oofP, tesP, readyP) =>
      if (new File(readyP).isFile) {
        val auc = metricsAuc(readyP)
        val hit = cover(new File(oofP)).orElse(cover(new File(tesP)))
        hit.foreach { df =>
          if (best.isEmpty || auc > best.get._3 + 1e-12)
            best = Some((df, readyP, auc))
        }
      }
    }
    best.foreach { case (_, p, a) =>
      println(f"[cb] using teacher parquet $p auc=$a%.5f for pred_cb_main/alt")
    }
    best.map(_._1)
  }

  private def metricsAuc(path: String): Double = {
    try {
      val txt = new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8)
      def grab(key: String): Option[Double] = {
        val pat = raw""""$key"\s*:\s*([0-9.]+)""".r
        pat.findFirstMatchIn(txt).map(_.group(1).toDouble)
      }
      grab("auc_cb_w62").orElse(grab("auc_blend")).orElse(grab("auc_cb_main")).getOrElse(0.0)
    } catch {
      case _: Throwable => 0.0
    }
  }

  def sparkDual(train: DataFrame, apply: DataFrame): DataFrame = {
    val a = sparkOne(train, apply, NumMain, "pred_cb_main", depth = 5, l2 = 10f, rsm = 1.0f, iters = 500, seed = 2026)
    val b = sparkOne(train, apply, NumAlt, "pred_cb_alt", depth = 6, l2 = 6f, rsm = 0.3f, iters = 500, seed = 2030)
    a.select(col("id"), col("pred_cb_main")).join(b.select(col("id"), col("pred_cb_alt")), Seq("id"))
  }

  def sparkOne(
      train: DataFrame,
      apply: DataFrame,
      nums: Seq[String],
      predCol: String,
      depth: Int,
      l2: Float,
      rsm: Float,
      iters: Int,
      seed: Int
  ): DataFrame = {
    val cats = Cats.filter(c => train.columns.contains(c))
    val numUse = nums.filter(c => train.columns.contains(c))
    val tr = GbtTrainer.fillNumeric(fillCats(train, cats), numUse)
    val ap = GbtTrainer.fillNumeric(fillCats(apply, cats), numUse)
    val catOut = cats.map(_ + "_cbidx")
    val indexers = cats.zip(catOut).map { case (c, o) =>
      new StringIndexer().setInputCol(c).setOutputCol(o).setHandleInvalid("keep")
    }
    val featNames = catOut ++ numUse
    val assembler = new VectorAssembler()
      .setInputCols(featNames.toArray)
      .setOutputCol("cb_features_raw")
      .setHandleInvalid("keep")
    val prep = new Pipeline().setStages((indexers :+ assembler).toArray)
    val prepModel = prep.fit(tr)
    val trV = attachCatMeta(prepModel.transform(tr), cats.length, featNames, "cb_features_raw")
    val apV = attachCatMeta(prepModel.transform(ap), cats.length, featNames, "cb_features_raw")

    val cb = new ai.catboost.spark.CatBoostRegressor()
      .setLabelCol("label")
      .setFeaturesCol("cb_features_raw")
      .setPredictionCol(predCol)
      .setLossFunction("RMSE")
      .setIterations(iters)
      .setDepth(depth)
      .setL2LeafReg(l2)
      .setLearningRate(0.03f)
      .setRsm(rsm)
      .setRandomSeed(seed)
      .setThreadCount(math.max(1, sys.env.getOrElse("CB_THREADS", "4").toInt))
      .setAllowWritingFiles(false)
      .setWorkerInitializationTimeout(Duration.ofSeconds(40))

    cb.fit(trV).transform(apV)
  }

  private def attachCatMeta(df: DataFrame, nCats: Int, names: Seq[String], colName: String): DataFrame = {
    val attrs: Array[Attribute] = names.zipWithIndex.map { case (n, i) =>
      if (i < nCats) NominalAttribute.defaultAttr.withName(n).withIndex(i).asInstanceOf[Attribute]
      else NumericAttribute.defaultAttr.withName(n).withIndex(i).asInstanceOf[Attribute]
    }.toArray
    val group = new AttributeGroup(colName, attrs)
    df.withColumn(colName, col(colName).as(colName, group.toMetadata))
  }

  def pythonDual(train: DataFrame, apply: DataFrame): DataFrame = {
    val threads = sys.env.getOrElse("CB_THREADS", "4").toInt
    val a = pythonOne(train, apply, NumMain, Cats, "pred_cb_main",
      s"""{"loss_function":"RMSE","iterations":700,"learning_rate":0.03,"depth":5,"l2_leaf_reg":10,"boosting_type":"Ordered","rsm":1.0,"random_seed":2026,"allow_writing_files":false,"thread_count":$threads,"verbose":false}""")
    val b = pythonOne(train, apply, NumAlt, Cats, "pred_cb_alt",
      s"""{"loss_function":"RMSE","iterations":700,"learning_rate":0.03,"depth":6,"l2_leaf_reg":6,"boosting_type":"Plain","rsm":0.3,"random_seed":2030,"allow_writing_files":false,"thread_count":$threads,"verbose":false}""")
    a.join(b, Seq("id"))
  }

  def pythonOne(train: DataFrame, apply: DataFrame, nums: Seq[String], cats: Seq[String], predCol: String, paramsJson: String): DataFrame = {
    val spark = train.sparkSession
    val tmp = Files.createTempDirectory("cbarm_")
    val trainP = tmp.resolve("train.tsv").toString
    val applyP = tmp.resolve("apply.tsv").toString
    val predP = tmp.resolve("pred.tsv").toString
    val useCats = cats.filter(c => train.columns.contains(c))
    val useNums = nums.filter(c => train.columns.contains(c))
    val feat = Seq("id", "label") ++ useNums ++ useCats
    writeTsv(train, feat, trainP)
    val apW = if (apply.columns.contains("label")) apply else apply.withColumn("label", lit(0.0))
    writeTsv(apW, feat, applyP)
    val cmd = Seq(
      "python3", fitScript.getAbsolutePath,
      "--train", trainP, "--apply", applyP, "--pred", predP,
      "--cats", useCats.mkString(","),
      "--params", paramsJson
    )
    val log = new StringBuilder
    val logger = ProcessLogger(s => { log.append(s).append('\n'); () }, s => { log.append(s).append('\n'); () })
    val rc = Process(cmd).!(logger)
    if (rc != 0) throw new RuntimeException(s"cb_fit_one.py failed rc=$rc\n$log")
    val predRows = scala.io.Source.fromFile(predP, "UTF-8").getLines().toArray
    if (predRows.length < 2) throw new RuntimeException("empty catboost preds")
    val schema = StructType(Seq(StructField("id", StringType, false), StructField(predCol, DoubleType, false)))
    val rows = predRows.drop(1).map { ln =>
      val ps = ln.split("\t", -1)
      org.apache.spark.sql.Row(ps(0), ps(1).toDouble)
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows.toSeq, 4), schema)
  }

  private def writeTsv(df: DataFrame, cols: Seq[String], path: String): Unit = {
    val present = cols.filter(df.columns.contains)
    val sel = df.select(present.map(c => coalesce(col(c).cast("string"), lit("NA")).alias(c)): _*)
    val rows = sel.collect()
    val w = new PrintWriter(path, "UTF-8")
    try {
      w.println(present.mkString("\t"))
      rows.foreach { r =>
        val vals = present.indices.map { i =>
          val s = if (r.isNullAt(i)) "NA" else r.getString(i)
          s.replace('\t', ' ').replace('\n', ' ')
        }
        w.println(vals.mkString("\t"))
      }
    } finally w.close()
  }

  def fillCats(df: DataFrame, cats: Seq[String]): DataFrame = {
    cats.foldLeft(df) { (acc, c) =>
      if (acc.columns.contains(c)) acc.withColumn(c, coalesce(col(c).cast("string"), lit("NA")))
      else acc.withColumn(c, lit("NA"))
    }
  }

  /** Ordered expanding-mean TE of medium cats + Spark GBT (CatBoost bias-correction analogue). */
  def orderedDual(train: DataFrame, apply: DataFrame): DataFrame = {
    val (tr, ap) = OrderedTE.add(train, apply, nPerm = 4, seed = 2026L)
    val ote = OrderedTE.oteCols().filter(c => tr.columns.contains(c))
    val a = gbtOte(tr, ap, NumMain ++ ote, "cb_main", depth = 5, iters = 110, step = 0.05, subset = "0.85", seed = 2026L)
    val b = gbtOte(tr, ap, NumAlt ++ ote, "cb_alt", depth = 6, iters = 120, step = 0.04, subset = "0.3", seed = 2030L)
    a.select(col("id"), col("pred_cb_main")).join(b.select(col("id"), col("pred_cb_alt")), Seq("id"))
  }

  private def gbtOte(
      train: DataFrame,
      apply: DataFrame,
      nums: Seq[String],
      name: String,
      depth: Int,
      iters: Int,
      step: Double,
      subset: String,
      seed: Long
  ): DataFrame = {
    val cats = LowCats.filter(c => train.columns.contains(c))
    val numUse = nums.filter(c => train.columns.contains(c) || c.startsWith("ote_"))
    val spec = GbtTrainer.Spec(
      name = name,
      numeric = numUse,
      cats = cats,
      maxDepth = depth,
      maxIter = iters,
      stepSize = step,
      minInstances = 70,
      subsample = 0.8,
      featureSubset = subset,
      seed = seed
    )
    GbtTrainer.fitPredict(train, apply, spec)
  }

  def main(args: Array[String]): Unit = {
    val dataDir = if (args.length > 0) args(0) else "/workspace/data"
    val outDir = if (args.length > 1) args(1) else "/workspace/submissions"
    val nFolds = if (args.length > 2) args(2).toInt else 5
    val spark = SparkSession.builder()
      .appName("catboost-arm")
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
    val withFold = TrainApp.assignStratifiedFolds(trainRaw, nFolds, 2026).cache()
    println(s"[cb] train=${withFold.count()} folds=$nFolds backend=${sys.env.getOrElse("CLAIM_CB_BACKEND", "auto")}")

    var oof: DataFrame = null
    for (f <- 0 until nFolds) {
      println(s"[cb-fold] $f / $nFolds")
      val tr = withFold.filter(col("fold") =!= f).drop("fold")
      val va = withFold.filter(col("fold") === f).drop("fold")
      val stats = FeatureEngine.fitStats(tr)
      val trF = FeatureEngine.transform(tr, stats, isTrain = true)
      val vaF = FeatureEngine.transform(va, stats, isTrain = false)
      val scored = score(trF, vaF)
      val piece = va.select(col("id"), col("label")).join(scored, Seq("id"))
      oof = if (oof == null) piece else oof.unionByName(piece)
    }
    oof.cache()
    val aucMain = TrainApp.auc(oof, "label", "pred_cb_main")
    val aucAlt = TrainApp.auc(oof, "label", "pred_cb_alt")
    val w62 = Blend(oof, Blend.CbOnly, "cb_w62")
    val aucW62 = TrainApp.auc(w62, "label", "cb_w62")
    println(f"[cb-OOF] main=$aucMain%.5f alt=$aucAlt%.5f w62=$aucW62%.5f")

    val teacherExtra = readTeacherMetrics()
    val report =
      s"""cb_oof_report
         |backend=${sys.env.getOrElse("CLAIM_CB_BACKEND", "auto")}
         |n_folds=$nFolds
         |auc_cb_main=$aucMain
         |auc_cb_alt=$aucAlt
         |auc_cb_w62=$aucW62
         |$teacherExtra
         |cats=medium_only_no_src_cq_dq
         |loss=RMSE
         |""".stripMargin
    new File(outDir).mkdirs()
    Files.write(Paths.get(s"$outDir/cb_oof_report.txt"), report.getBytes(StandardCharsets.UTF_8))
    println(report)

    val stats = FeatureEngine.fitStats(trainRaw)
    val trF = FeatureEngine.transform(trainRaw, stats, isTrain = true)
    val teF = FeatureEngine.transform(testRaw.withColumn("label", lit(0.0)), stats, isTrain = false)
    val testScored = score(trF, teF)
    val testOut = Blend(testScored, Blend.CbOnly, "label").select(col("id"), col("label"))
    val tmp = s"$outDir/cb_test_dir"
    testOut.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp)
    val part = new File(tmp).listFiles().find(_.getName.startsWith("part")).get
    Files.copy(part.toPath, Paths.get(s"$outDir/cb_arm_test.csv"), java.nio.file.StandardCopyOption.REPLACE_EXISTING)
    spark.stop()
  }

  private def readTeacherMetrics(): String = {
    val f = new File("/workspace/submissions/cb_teacher_metrics.json")
    if (!f.isFile) return "teacher=none"
    try {
      val s = new String(Files.readAllBytes(f.toPath), StandardCharsets.UTF_8)
      "teacher_json=" + s.replaceAll("\\s+", " ")
    } catch { case _: Throwable => "teacher=unreadable" }
  }
}
