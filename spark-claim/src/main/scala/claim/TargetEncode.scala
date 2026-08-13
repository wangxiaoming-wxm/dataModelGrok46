package claim

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

/**
 * Smoothed target encoding used as a STANDALONE score, not as a GBT input.
 * High-cardinality TE fed into Spark/sklearn GBT collapses (best_iter=1, AUC~0.50).
 *
 * Stats are fit on the outer-train fold only, then joined onto the apply frame.
 * Multiple smoothing m values are produced from one groupBy.
 */
object TargetEncode {

  val Ms: Seq[Double] = Seq(10.0, 15.0, 20.0, 30.0)

  def colName(key: String, m: Double): String = s"te_${key}_m${m.toInt}"

  def defaultCol(key: String): String = colName(key, 20.0)

  def fitTransformMany(
      train: DataFrame,
      applyDf: DataFrame,
      keys: Seq[String],
      ms: Seq[Double] = Ms,
      labelCol: String = "label"
  ): DataFrame = {
    var out = applyDf
    keys.foreach { k =>
      out = attach(train, out, k, ms, labelCol)
    }
    val m20 = keys.map(defaultCol)
    val present = m20.filter(c => out.columns.contains(c))
    if (present.nonEmpty) {
      val avg = present.map(c => col(c)).reduce(_ + _) / lit(present.size.toDouble)
      out.withColumn("te_pool", avg)
    } else out.withColumn("te_pool", lit(0.1))
  }

  def attach(
      train: DataFrame,
      applyDf: DataFrame,
      keyCol: String,
      ms: Seq[Double],
      labelCol: String
  ): DataFrame = {
    val prior = train.agg(avg(col(labelCol).cast(DoubleType))).first().getDouble(0)
    var stats = train.groupBy(keyCol).agg(
      sum(col(labelCol).cast(DoubleType)).as("_te_s"),
      count(lit(1)).as("_te_c")
    )
    val teCols = ms.map { m =>
      val name = colName(keyCol, m)
      stats = stats.withColumn(name, (col("_te_s") + lit(prior * m)) / (col("_te_c") + lit(m)))
      name
    }
    val keyAlias = keyCol + "_te_k"
    val slim = stats.select((col(keyCol).as(keyAlias) +: teCols.map(c => col(c))): _*)
    applyDf.join(broadcast(slim), applyDf(keyCol) === col(keyAlias), "left")
      .drop(keyAlias)
      .na.fill(prior, teCols)
  }
}
