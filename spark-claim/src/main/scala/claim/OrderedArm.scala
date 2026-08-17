package claim

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * Spark-native CatBoost analogue: ordered expanding-mean TE on medium-card
 * keys, then GBTRegressor (squared error). Never feeds src_cq_dq into trees.
 */
object OrderedArm {

  def score(train: DataFrame, apply: DataFrame, gbtIter: Int): DataFrame = {
    val t0 = System.nanoTime()
    val nPerm = if (gbtIter <= 20) 2 else 4
    val (tr, ap) = OrderedTE.add(train, apply, nPerm = nPerm, seed = 2026L)
    val ote = OrderedTE.oteCols().filter(c => tr.columns.contains(c))
    val mainNums = (FeatureEngine.NumericMain ++ ote).distinct.filter(c => tr.columns.contains(c) || c.startsWith("ote_"))
    val altNums = (FeatureEngine.NumericAlt ++ ote).distinct.filter(c => tr.columns.contains(c) || c.startsWith("ote_"))
    val low = Seq("src", "reg", "age", "grades_s", "month_s", "days_q", "cond_q").filter(tr.columns.contains)
    val a = GbtTrainer.fitPredict(tr, ap, GbtTrainer.Spec(
      name = "cb_main",
      numeric = mainNums,
      cats = low,
      maxDepth = 5,
      maxIter = math.max(8, (gbtIter * 1.2).toInt),
      stepSize = 0.05,
      minInstances = 70,
      subsample = 0.8,
      featureSubset = "0.85",
      seed = 2026L
    ))
    val b = GbtTrainer.fitPredict(tr, ap, GbtTrainer.Spec(
      name = "cb_alt",
      numeric = altNums,
      cats = low,
      maxDepth = 6,
      maxIter = math.max(8, (gbtIter * 1.3).toInt),
      stepSize = 0.04,
      minInstances = 70,
      subsample = 0.75,
      featureSubset = "0.3",
      seed = 2030L
    ))
    println(f"[ordered-arm] ${(System.nanoTime()-t0)/1e9}%.1fs")
    a.select(col("id"), col("pred_cb_main")).join(b.select(col("id"), col("pred_cb_alt")), Seq("id"))
  }
}
