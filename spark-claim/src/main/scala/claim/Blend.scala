package claim

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

/** Rank-weighted blend. 3-way TE is an independent score, never a tree feature. */
object Blend {

  case class Arm(col: String, w: Double)

  val Default: Seq[Arm] = Seq(
    Arm("pred_cb_main", 0.34),
    Arm("pred_cb_alt", 0.22),
    Arm("pred_main", 0.10),
    Arm("pred_alt", 0.06),
    Arm("pred_rf", 0.04),
    Arm("te_src_cq_dq", 0.14),
    Arm("te_cq_dq", 0.05),
    Arm("te_src_ratioq", 0.05)
  )

  val CbOnly: Seq[Arm] = Seq(
    Arm("pred_cb_main", 0.62),
    Arm("pred_cb_alt", 0.38)
  )

  def percentRank(df: DataFrame, c: String): org.apache.spark.sql.Column =
    percent_rank().over(Window.orderBy(col(c)))

  def apply(df: DataFrame, arms: Seq[Arm] = Default, out: String = "oof_blend"): DataFrame = {
    val present = arms.filter(a => df.columns.contains(a.col))
    if (present.isEmpty) return df.withColumn(out, lit(0.5))
    val wsum = present.map(_.w).sum
    val norm = present.map(a => a.copy(w = a.w / wsum))
    var acc = df
    norm.foreach { a =>
      acc = acc.withColumn("r_" + a.col, percentRank(acc, a.col))
    }
    val expr = norm.map(a => lit(a.w) * col("r_" + a.col)).reduce(_ + _)
    acc.withColumn(out, expr)
  }
}
