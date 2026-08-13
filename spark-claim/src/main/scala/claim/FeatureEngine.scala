package claim

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/** Fold-safe dual-world features. Quantiles / medians / percentiles always come from train stats. */
object FeatureEngine {

  val NumericMain: Seq[String] = Seq(
    "days", "days_log", "condition_f", "cond_r", "ratio", "ratio_sqrt",
    "u_shape", "inv_cond", "age_range", "V", "cc", "x20", "x1", "x5",
    "age8", "cond_low", "safe_750", "safe_1750", "t3_num", "cond_miss",
    "max_g", "x14", "x17"
  )

  val NumericAlt: Seq[String] = Seq(
    "days", "days_log", "condition_f", "cond_rk", "rate", "u_shape",
    "age_range", "V", "x20", "x1", "age8", "cond_low", "safe_750",
    "safe_1750", "t3_num", "cond_miss", "x5"
  )

  val LowCardCats: Seq[String] = Seq(
    "src", "reg", "age", "days_q", "cond_q", "grades_s", "code_s", "month_s"
  )

  val TeKeys: Seq[String] = Seq(
    "src_cq_dq", "cq_dq", "src_cq", "src_dq", "reg_dq", "src_ratioq",
    "src_rateq", "reg_age", "src_reg", "src_cq_age", "reg_cq_dq"
  )

  case class Stats(
      srcMed: Map[String, Double],
      globMed: Double,
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
    val srcMed = base.groupBy("source")
      .agg(expr("percentile_approx(condition, 0.5)").as("m"))
      .collect()
      .map { r =>
        val k = r.getAs[String]("source")
        val m = if (r.isNullAt(1)) globMed else r.get(1).asInstanceOf[Number].doubleValue()
        k -> m
      }
      .toMap

    val tmp = attachCondition(base, srcMed, globMed)
    val tmp2 = attachCondR(tmp, srcMed, globMed)
    val tmp3 = attachSelfPercentile(tmp2)
    val tmp4 = attachRatios(tmp3)

    Stats(
      srcMed = srcMed,
      globMed = globMed,
      daysEdges = quantileEdges(tmp4, "days"),
      condEdges = quantileEdges(tmp4, "condition_f"),
      ratioEdges = quantileEdges(tmp4, "ratio"),
      rateEdges = quantileEdges(tmp4, "rate"),
      srcCond = tmp4.select("source", "condition_f").collect()
        .groupBy(_.getAs[String]("source"))
        .map { case (s, rows) =>
          s -> rows.map(_.getAs[Number]("condition_f").doubleValue()).sorted
        }
    )
  }

  def transform(df: DataFrame, stats: Stats, isTrain: Boolean): DataFrame = {
    val base = parseBase(df)
    val t1 = attachCondition(base, stats.srcMed, stats.globMed)
    val t2 = attachCondR(t1, stats.srcMed, stats.globMed)
    val t3 = if (isTrain) attachSelfPercentile(t2) else attachRefPercentile(t2, stats.srcCond)
    val t4 = attachRatios(t3)
    val t5 = t4
      .withColumn("days_q", bucketize(col("days"), stats.daysEdges))
      .withColumn("cond_q", bucketize(col("condition_f"), stats.condEdges))
      .withColumn("ratio_q", bucketize(col("ratio"), stats.ratioEdges))
      .withColumn("rate_q", bucketize(col("rate"), stats.rateEdges))
    addCrosses(t5)
  }

  private def medianOf(df: DataFrame, c: String): Double = {
    val v = df.stat.approxQuantile(c, Array(0.5), 0.0)
    if (v.isEmpty) 1.0 else v.head
  }

  private def quantileEdges(df: DataFrame, c: String, q: Int = 10): Array[Double] = {
    val probs = (1 until q).map(i => i.toDouble / q).toArray
    val e = df.stat.approxQuantile(c, probs, 0.0)
    (Array(Double.NegativeInfinity) ++ e ++ Array(Double.PositiveInfinity)).distinct.sorted
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
        col("condition_f") / when(col("src_med2") === 0.0, lit(globMed)).otherwise(col("src_med2"))
      )
      .drop("src_med2")
  }

  private def attachSelfPercentile(df: DataFrame): DataFrame = {
    val wOrd = Window.partitionBy("source").orderBy("condition_f")
    val wAll = Window.partitionBy("source")
    df.withColumn(
      "cond_rk",
      (rank().over(wOrd) - lit(1.0)) / greatest(count(lit(1)).over(wAll) - lit(1.0), lit(1.0))
    )
  }

  private def attachRefPercentile(df: DataFrame, srcCond: Map[String, Array[Double]]): DataFrame = {
    val bc = df.sparkSession.sparkContext.broadcast(srcCond)
    val ud = udf { (source: String, cond: Double) =>
      val ref = bc.value.getOrElse(Option(source).getOrElse(""), Array.empty[Double])
      if (ref.isEmpty) 0.5
      else {
        val idx = java.util.Arrays.binarySearch(ref, cond)
        val k = if (idx >= 0) idx + 1 else math.min(ref.length, -idx - 1)
        k.toDouble / ref.length.toDouble
      }
    }
    df.withColumn("cond_rk", ud(col("source"), col("condition_f")))
  }

  private def attachRatios(df: DataFrame): DataFrame = {
    df.withColumn("ratio", col("days") / col("cond_r"))
      .withColumn("rate", col("days") * (lit(1.0) - col("cond_rk")))
      .withColumn("ratio_sqrt", col("days") / sqrt(greatest(col("condition_f"), lit(1e-6))))
      .withColumn("u_shape", pow(col("cond_rk") - lit(0.5), 2.0))
      .withColumn("inv_cond", lit(1.0) / greatest(col("condition_f"), lit(1e-4)))
      .withColumn("days_log", log1p(col("days")))
      .withColumn("cond_low", (col("condition_f") < 0.05).cast(DoubleType))
      .withColumn("safe_750", ((col("days") >= 700.0 && col("days") < 880.0).cast(DoubleType)))
      .withColumn("safe_1750", ((col("days") >= 1725.0 && col("days") < 1825.0).cast(DoubleType)))
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
  }

  def bucketize(c: org.apache.spark.sql.Column, edges: Array[Double]): org.apache.spark.sql.Column = {
    var exprCol: org.apache.spark.sql.Column = lit(0)
    var i = edges.length - 2
    while (i >= 0) {
      exprCol = when(c >= lit(edges(i)), lit(i)).otherwise(exprCol)
      i -= 1
    }
    exprCol.cast(StringType)
  }
}
