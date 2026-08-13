package claim

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/** Fold-safe dual-world features. Quantiles / medians / percentiles always come from train stats. */
object FeatureEngine {

  val NumericMain: Seq[String] = Seq(
    "days", "days_log", "days_sqrt", "days2",
    "condition_f", "cond_log", "cond_r", "ratio", "ratio_sqrt", "log_ratio",
    "inv_cond", "days_x_inv", "u_shape",
    "age_range", "V", "cc", "x20", "x1", "x5",
    "age8", "cond_low", "cond_miss",
    "w_safe750", "w_safe1750", "w_hot1950", "w_new50",
    "t3_num", "max_g", "x14", "x17",
    "v_r", "cc_r", "maxg_r",
    "cond_z", "days_z",
    "ushape_car10", "mono_car1", "rev_car7"
  )

  val NumericAlt: Seq[String] = Seq(
    "days", "days_log", "days_sqrt",
    "condition_f", "cond_rk", "rate", "u_shape",
    "age_range", "V", "x20", "x1", "x5",
    "age8", "cond_low", "cond_miss",
    "w_safe750", "w_safe1750", "w_hot1950", "w_new50",
    "t3_num", "days_z", "cond_z",
    "ushape_car10", "mono_car1", "rev_car7"
  )

  val NumericBig: Seq[String] = Seq(
    "days", "days_log", "condition_f", "cond_r", "cond_rk",
    "ratio", "rate", "u_shape", "age_range", "inv_cond",
    "ushape_car10", "mono_car1", "days_z", "cond_z"
  )

  val AllNumeric: Seq[String] = (NumericMain ++ NumericAlt ++ NumericBig).distinct

  /** Low-card cats for trees. Do NOT include high-card crosses (those are TE scores only). */
  val LowCardCatsMain: Seq[String] = Seq(
    "src", "reg", "age", "days_q", "cond_q", "ratio_q", "grades_s", "month_s"
  )

  val LowCardCatsAlt: Seq[String] = Seq(
    "src", "reg", "age", "days_q", "cond_q", "rate_q", "grades_s", "month_s"
  )

  val LowCardCats: Seq[String] = (LowCardCatsMain ++ LowCardCatsAlt).distinct

  val GlmLowCats: Seq[String] = Seq(
    "src", "reg", "age", "days_q", "cond_q", "ratio_q", "rate_q", "grades_s"
  )

  val GlmCrossCats: Seq[String] = Seq(
    "src_cq", "src_dq", "cq_dq", "src_reg", "src_ratioq", "src_cq_dq"
  )

  val TeKeys: Seq[String] = Seq(
    "src_cq_dq", "cq_dq", "src_cq", "src_dq", "reg_dq", "src_ratioq",
    "src_rateq", "reg_age", "src_reg", "src_cq_age", "reg_cq_dq",
    "src_grades", "src_age", "src_reg_age"
  )

  val LargeCars: Seq[String] = Seq("CAR_0", "CAR_1", "CAR_2")

  case class Stats(
      srcMed: Map[String, Double],
      globMed: Double,
      srcMeanCond: Map[String, Double],
      srcStdCond: Map[String, Double],
      srcMeanDays: Map[String, Double],
      srcStdDays: Map[String, Double],
      daysEdges: Array[Double],
      condEdges: Array[Double],
      ratioEdges: Array[Double],
      rateEdges: Array[Double],
      srcCond: Map[String, Array[Double]]
  )

  def parseBase(df: DataFrame): DataFrame = {
    df.withColumn("t3_num", regexp_extract(col("t3"), "^([0-9.]+)", 1).cast(DoubleType))
      .withColumn("t3_letter", regexp_extract(col("t3"), "([A-Za-z])$", 1))
      .withColumn("car", split(col("source"), "\\|").getItem(0))
      .withColumn("cond_miss", col("condition").isNull.cast(DoubleType))
      .withColumn("age8", (col("age_range") >= 8).cast(DoubleType))
      .withColumn("src", col("source").cast(StringType))
      .withColumn("reg", col("region").cast(StringType))
      .withColumn("age", col("age_range").cast(IntegerType).cast(StringType))
      .withColumn("grades_s", col("grades").cast(StringType))
      .withColumn("code_s", col("code").cast(StringType))
      .withColumn("month_s", col("month").cast(StringType))
  }

  def fitStats(train: DataFrame): Stats = {
    val base = parseBase(train)
    val globMed = medianOf(base, "condition")
    val srcMed = aggMap(base.groupBy("source").agg(expr("percentile_approx(condition, 0.5)").as("v")), globMed)
    val srcMeanCond = aggMap(base.groupBy("source").agg(avg(col("condition")).as("v")), globMed)
    val srcStdCond = aggMap(base.groupBy("source").agg(stddev_pop(col("condition")).as("v")), 1.0)
    val srcMeanDays = aggMap(base.groupBy("source").agg(avg(col("days")).as("v")), 0.0)
    val srcStdDays = aggMap(base.groupBy("source").agg(stddev_pop(col("days")).as("v")), 1.0)

    val tmp = attachCondition(base, srcMed, globMed)
    val tmp2 = attachCondR(tmp, srcMed, globMed)
    val tmp3 = attachSelfPercentile(tmp2)
    val tmp4 = attachRatios(tmp3)
    val tmp5 = attachSourceZ(tmp4, srcMeanCond, srcStdCond, srcMeanDays, srcStdDays)

    Stats(
      srcMed = srcMed,
      globMed = globMed,
      srcMeanCond = srcMeanCond,
      srcStdCond = srcStdCond,
      srcMeanDays = srcMeanDays,
      srcStdDays = srcStdDays,
      daysEdges = quantileEdges(tmp5, "days"),
      condEdges = quantileEdges(tmp5, "condition_f"),
      ratioEdges = quantileEdges(tmp5, "ratio"),
      rateEdges = quantileEdges(tmp5, "rate"),
      srcCond = tmp5.select("source", "condition_f").collect()
        .groupBy(_.getAs[String]("source"))
        .map { case (s, rows) =>
          s -> rows.map { r =>
            val n = r.getAs[Number]("condition_f")
            if (n == null) globMed else n.doubleValue()
          }.sorted
        }
    )
  }

  def transform(df: DataFrame, stats: Stats, isTrain: Boolean): DataFrame = {
    val base = parseBase(df)
    val t1 = attachCondition(base, stats.srcMed, stats.globMed)
    val t2 = attachCondR(t1, stats.srcMed, stats.globMed)
    val t3 = if (isTrain) attachSelfPercentile(t2) else attachRefPercentile(t2, stats.srcCond)
    val t4 = attachRatios(t3)
    val t5 = attachSourceZ(t4, stats.srcMeanCond, stats.srcStdCond, stats.srcMeanDays, stats.srcStdDays)
    val t6 = t5
      .withColumn("days_q", bucketize(col("days"), stats.daysEdges))
      .withColumn("cond_q", bucketize(col("condition_f"), stats.condEdges))
      .withColumn("ratio_q", bucketize(col("ratio"), stats.ratioEdges))
      .withColumn("rate_q", bucketize(col("rate"), stats.rateEdges))
    val t7 = addCrosses(t6)
    sanitize(t7, AllNumeric)
  }

  def finiteFill(c: org.apache.spark.sql.Column, default: Double = 0.0): org.apache.spark.sql.Column = {
    when(
      c.isNull || isnan(c) || (c === lit(Double.PositiveInfinity)) || (c === lit(Double.NegativeInfinity)),
      lit(default)
    ).otherwise(c.cast(DoubleType))
  }

  def sanitize(df: DataFrame, cols: Seq[String]): DataFrame = {
    cols.foldLeft(df) { (acc, c) =>
      if (acc.columns.contains(c)) acc.withColumn(c, finiteFill(col(c), 0.0))
      else acc.withColumn(c, lit(0.0))
    }
  }

  private def medianOf(df: DataFrame, c: String): Double = {
    val v = df.stat.approxQuantile(c, Array(0.5), 0.0)
    if (v.isEmpty || v.head.isNaN) 1.0 else v.head
  }

  private def quantileEdges(df: DataFrame, c: String, q: Int = 10): Array[Double] = {
    val probs = (1 until q).map(i => i.toDouble / q).toArray
    df.stat.approxQuantile(c, probs, 0.0).filter(x => java.lang.Double.isFinite(x)).distinct.sorted
  }

  private def aggMap(g: DataFrame, fallback: Double): Map[String, Double] = {
    g.collect().flatMap { r =>
      val k = r.getAs[String](0)
      if (k == null) None
      else {
        val v =
          if (r.isNullAt(1)) fallback
          else {
            val n = r.get(1).asInstanceOf[Number].doubleValue()
            if (n.isNaN || n.isInfinity || math.abs(n) < 1e-12) fallback else n
          }
        Some(k -> v)
      }
    }.toMap
  }

  private def mapToDf(spark: SparkSession, m: Map[String, Double], k: String, v: String): DataFrame = {
    val schema = StructType(Seq(StructField(k, StringType, nullable = true), StructField(v, DoubleType, nullable = false)))
    val rows = m.toSeq.map { case (a, b) => org.apache.spark.sql.Row(a, b) }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 1), schema)
  }

  private def attachCondition(df: DataFrame, srcMed: Map[String, Double], globMed: Double): DataFrame = {
    val mdf = mapToDf(df.sparkSession, srcMed, "source_k", "src_med")
    df.join(broadcast(mdf), df("source") === mdf("source_k"), "left")
      .drop("source_k")
      .withColumn("condition_f", coalesce(col("condition"), col("src_med"), lit(globMed)))
      .drop("src_med")
  }

  private def attachCondR(df: DataFrame, srcMed: Map[String, Double], globMed: Double): DataFrame = {
    val mdf = mapToDf(df.sparkSession, srcMed, "source_k2", "src_med2")
    df.join(broadcast(mdf), df("source") === mdf("source_k2"), "left")
      .drop("source_k2")
      .withColumn("src_med2", coalesce(col("src_med2"), lit(globMed)))
      .withColumn(
        "cond_r",
        col("condition_f") / when(col("src_med2") <= 1e-12, lit(globMed)).otherwise(col("src_med2"))
      )
      .drop("src_med2")
  }

  private def attachSelfPercentile(df: DataFrame): DataFrame = {
    val w = Window.partitionBy("source").orderBy(col("condition_f"))
    df.withColumn("cond_rk", percent_rank().over(w))
  }

  private def attachRefPercentile(df: DataFrame, srcCond: Map[String, Array[Double]]): DataFrame = {
    val bc = df.sparkSession.sparkContext.broadcast(srcCond)
    val ud = udf { (source: String, cond: Double) =>
      val ref = bc.value.getOrElse(Option(source).getOrElse(""), Array.empty[Double])
      if (ref.isEmpty || cond.isNaN) 0.5
      else {
        val idx = java.util.Arrays.binarySearch(ref, cond)
        val k = if (idx >= 0) idx + 1 else math.min(ref.length, -idx - 1)
        k.toDouble / ref.length.toDouble
      }
    }
    df.withColumn("cond_rk", ud(col("source"), col("condition_f")))
  }

  private def attachSourceZ(
      df: DataFrame,
      meanC: Map[String, Double],
      stdC: Map[String, Double],
      meanD: Map[String, Double],
      stdD: Map[String, Double]
  ): DataFrame = {
    val spark = df.sparkSession
    val mc = mapToDf(spark, meanC, "sk_mc", "mu_c")
    val sc = mapToDf(spark, stdC, "sk_sc", "sd_c")
    val md = mapToDf(spark, meanD, "sk_md", "mu_d")
    val sd = mapToDf(spark, stdD, "sk_sd", "sd_d")
    df.join(broadcast(mc), df("source") === mc("sk_mc"), "left").drop("sk_mc")
      .join(broadcast(sc), col("source") === sc("sk_sc"), "left").drop("sk_sc")
      .join(broadcast(md), col("source") === md("sk_md"), "left").drop("sk_md")
      .join(broadcast(sd), col("source") === sd("sk_sd"), "left").drop("sk_sd")
      .withColumn("sd_c", when(col("sd_c").isNull || col("sd_c") < 1e-8, lit(1.0)).otherwise(col("sd_c")))
      .withColumn("sd_d", when(col("sd_d").isNull || col("sd_d") < 1e-8, lit(1.0)).otherwise(col("sd_d")))
      .withColumn("cond_z", (col("condition_f") - coalesce(col("mu_c"), col("condition_f"))) / col("sd_c"))
      .withColumn("days_z", (col("days") - coalesce(col("mu_d"), col("days"))) / col("sd_d"))
      .drop("mu_c", "sd_c", "mu_d", "sd_d")
  }

  private def attachRatios(df: DataFrame): DataFrame = {
    val crSafe = greatest(col("cond_r"), lit(1e-6))
    val condSafe = greatest(col("condition_f"), lit(1e-6))
    df.withColumn("ratio", col("days") / crSafe)
      .withColumn("rate", col("days") * (lit(1.0) - col("cond_rk")))
      .withColumn("ratio_sqrt", col("days") / sqrt(condSafe))
      .withColumn("log_ratio", log(greatest(col("days"), lit(1.0))) - lit(0.5) * log(condSafe))
      .withColumn("u_shape", pow(col("cond_rk") - lit(0.5), 2.0))
      .withColumn("inv_cond", lit(1.0) / greatest(col("condition_f"), lit(1e-4)))
      .withColumn("days_x_inv", col("days") / greatest(col("condition_f"), lit(1e-4)))
      .withColumn("days_log", log1p(greatest(col("days"), lit(0.0))))
      .withColumn("days_sqrt", sqrt(greatest(col("days"), lit(0.0))))
      .withColumn("days2", pow(col("days") / lit(5000.0), 2.0))
      .withColumn("cond_log", log(condSafe))
      .withColumn("cond_low", (col("condition_f") < 0.05).cast(DoubleType))
      .withColumn("w_safe750", ((col("days") >= 700.0 && col("days") < 880.0).cast(DoubleType)))
      .withColumn("w_safe1750", ((col("days") >= 1725.0 && col("days") < 1825.0).cast(DoubleType)))
      .withColumn("w_hot1950", ((col("days") >= 1950.0 && col("days") < 2000.0).cast(DoubleType)))
      .withColumn("w_new50", (col("days") < 50.0).cast(DoubleType))
      .withColumn("safe_750", col("w_safe750"))
      .withColumn("safe_1750", col("w_safe1750"))
      .withColumn("safe_1950", col("w_hot1950"))
      .withColumn("days_lt50", col("w_new50"))
      .withColumn("v_r", col("V") / crSafe)
      .withColumn("cc_r", col("cc") / crSafe)
      .withColumn("maxg_r", col("max_g") / crSafe)
      .withColumn("ushape_car10", col("u_shape") * (col("car") === lit("CAR_10")).cast(DoubleType))
      .withColumn("ushape_car0", col("u_shape") * (col("car") === lit("CAR_0")).cast(DoubleType))
      .withColumn("mono_car1", (lit(1.0) - col("cond_rk")) * (col("car") === lit("CAR_1")).cast(DoubleType))
      .withColumn("rev_car7", col("cond_rk") * (col("car") === lit("CAR_7")).cast(DoubleType))
  }

  def addCrosses(df: DataFrame): DataFrame = {
    df.withColumn("src_reg", concat_ws("|", col("src"), col("reg")))
      .withColumn("src_age", concat_ws("|", col("src"), col("age")))
      .withColumn("reg_age", concat_ws("|", col("reg"), col("age")))
      .withColumn("src_cq", concat_ws("|", col("src"), col("cond_q")))
      .withColumn("src_dq", concat_ws("|", col("src"), col("days_q")))
      .withColumn("reg_cq", concat_ws("|", col("reg"), col("cond_q")))
      .withColumn("reg_dq", concat_ws("|", col("reg"), col("days_q")))
      .withColumn("src_cq_dq", concat_ws("|", col("src"), col("cond_q"), col("days_q")))
      .withColumn("cq_dq", concat_ws("|", col("cond_q"), col("days_q")))
      .withColumn("src_ratioq", concat_ws("|", col("src"), col("ratio_q")))
      .withColumn("src_rateq", concat_ws("|", col("src"), col("rate_q")))
      .withColumn("src_cq_age", concat_ws("|", col("src"), col("cond_q"), col("age")))
      .withColumn("reg_cq_dq", concat_ws("|", col("reg"), col("cond_q"), col("days_q")))
      .withColumn("src_grades", concat_ws("|", col("src"), col("grades_s")))
      .withColumn("src_reg_age", concat_ws("|", col("src"), col("reg"), col("age")))
  }

  /**
   * Quantile bins from finite interior edges (10th..90th). No ±Infinity in the
   * comparison chain: `c >= -inf` is true for every finite value and used to
   * collapse every row into bucket 0 (TE OOF ~0.51).
   */
  def bucketize(c: org.apache.spark.sql.Column, edges: Array[Double]): org.apache.spark.sql.Column = {
    val cuts = edges.filter(x => java.lang.Double.isFinite(x)).sorted.distinct
    if (cuts.isEmpty) lit("0")
    else {
      var exprCol: org.apache.spark.sql.Column = lit(cuts.length)
      var i = cuts.length - 1
      while (i >= 0) {
        exprCol = when(c < lit(cuts(i)), lit(i)).otherwise(exprCol)
        i -= 1
      }
      exprCol.cast(StringType)
    }
  }
}
