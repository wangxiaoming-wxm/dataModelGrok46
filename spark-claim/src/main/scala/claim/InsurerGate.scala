package claim

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * Frozen customer/insurer overlay on rank-space scores.
 *
 * Insurer will not pay in warranty pits (GENERATING_PROCESS confirm):
 *   [1725,1825) all-zero n=103; [700,880) rate 3.07%.
 * Customer dumps claims near cover end: [9370,9475) rate 18.9%.
 *
 * Apply AFTER rank fusion. Magnitudes are frozen — do not grid-search on full OOF.
 */
object InsurerGate {
  val Shift750: Double = 0.10
  val ShiftHot: Double = 0.05

  def onScore(df: DataFrame, scoreCol: String = "blend", daysCol: String = "days"): DataFrame = {
    if (!df.columns.contains(scoreCol) || !df.columns.contains(daysCol)) return df
    val s = col(scoreCol)
    val d = col(daysCol)
    df.withColumn(
      scoreCol,
      when(d >= 1725.0 && d < 1825.0, lit(-1.0))
        .when(d >= 700.0 && d < 880.0, s - lit(Shift750))
        .when(d >= 9370.0 && d < 9475.0, s + lit(ShiftHot))
        .otherwise(s)
    )
  }
}
