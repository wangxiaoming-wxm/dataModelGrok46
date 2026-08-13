package claim

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

/**
 * Smoothed target encoding used as a STANDALONE score, not as a GBT input.
 * High-cardinality TE fed into Spark/sklearn GBT collapses (best_iter=1, AUC~0.50).
 */
object TargetEncode {

  case class Table(mean: Map[String, Double], prior: Double, m: Double) {
    def applyKey(k: String): Double = {
      mean.getOrElse(k, prior)
    }
  }

  def fit(train: DataFrame, keyCol: String, labelCol: String = "label", m: Double = 20.0): Table = {
    val prior = train.agg(avg(col(labelCol).cast(DoubleType))).first().getDouble(0)
    val rows = train.groupBy(keyCol).agg(sum(col(labelCol).cast(DoubleType)).as("s"), count(lit(1)).as("c")).collect()
    val mean = rows.map { r =>
      val k = Option(r.getString(0)).getOrElse("NA")
      val s = r.getDouble(1)
      val c = r.getLong(2).toDouble
      k -> ((s + prior * m) / (c + m))
    }.toMap
    Table(mean, prior, m)
  }

  def transform(df: DataFrame, keyCol: String, outCol: String, table: Table): DataFrame = {
    val bc = df.sparkSession.sparkContext.broadcast(table)
    val ud = udf { k: String => bc.value.applyKey(Option(k).getOrElse("NA")) }
    df.withColumn(outCol, ud(col(keyCol)))
  }

  def fitTransformMany(train: DataFrame, applyDf: DataFrame, keys: Seq[String], prefix: String = "te_"): DataFrame = {
    var out = applyDf
    keys.foreach { k =>
      val tab = fit(train, k)
      out = transform(out, k, prefix + k, tab)
    }
    out
  }
}
