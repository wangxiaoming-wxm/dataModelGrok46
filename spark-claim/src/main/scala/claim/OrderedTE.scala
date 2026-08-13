package claim

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.util.Random

/**
 * CatBoost-style ordered target statistics: expanding mean over several
 * permutations. Medium-cardinality keys only — never feed src_cq_dq into trees.
 */
object OrderedTE {

  val MediumKeys: Seq[String] = Seq(
    "src", "reg", "age", "grades_s", "month_s",
    "src_reg", "src_age", "reg_age", "src_cq",
    "days_q", "cond_q", "src_dq", "reg_cq", "reg_dq"
  )

  def oteCols(keys: Seq[String] = MediumKeys): Seq[String] = keys.map("ote_" + _)

  def add(
      train: DataFrame,
      apply: DataFrame,
      keys: Seq[String] = MediumKeys,
      nPerm: Int = 4,
      seed: Long = 2026L,
      m: Double = 20.0
  ): (DataFrame, DataFrame) = {
    val spark = train.sparkSession
    val present = keys.filter(k => train.columns.contains(k) && apply.columns.contains(k))
    if (present.isEmpty) return (train, apply)

    val trRows = train.select((Seq("id", "label") ++ present).map(col): _*).collect()
    val apRows = apply.select((Seq("id") ++ present).map(col): _*).collect()
    val n = trRows.length
    val y = trRows.map(r => r.getAs[Number]("label").doubleValue())
    val prior = if (n == 0) 0.1 else y.sum / n.toDouble
    val idsTr = trRows.map(_.getAs[String]("id"))
    val idsAp = apRows.map(_.getAs[String]("id"))

    val trEnc = mutable.Map.empty[String, Array[Double]]
    val apEnc = mutable.Map.empty[String, Array[Double]]
    present.zipWithIndex.foreach { case (key, ki) =>
      val trK = trRows.map(r => Option(r.getAs[Any](key)).map(_.toString).getOrElse("NA"))
      val apK = apRows.map(r => Option(r.getAs[Any](key)).map(_.toString).getOrElse("NA"))
      val (te, ae) = encodeOne(trK, y, apK, prior, m, nPerm, seed + 17L * (ki + 1))
      trEnc(key) = te
      apEnc(key) = ae
    }

    def toDf(ids: Array[String], enc: mutable.Map[String, Array[Double]]): DataFrame = {
      val schema = StructType(
        StructField("id_ote", StringType, nullable = false) +:
          present.map(k => StructField("ote_" + k, DoubleType, nullable = false))
      )
      val rows = ids.indices.map { i =>
        Row.fromSeq(ids(i) +: present.map(k => enc(k)(i).asInstanceOf[Any]))
      }
      spark.createDataFrame(spark.sparkContext.parallelize(rows, 4), schema)
    }

    val trJoin = toDf(idsTr, trEnc)
    val apJoin = toDf(idsAp, apEnc)
    val trOut = train.join(trJoin, train("id") === trJoin("id_ote"), "left").drop("id_ote")
    val apOut = apply.join(apJoin, apply("id") === apJoin("id_ote"), "left").drop("id_ote")
    (fillOte(trOut, present), fillOte(apOut, present))
  }

  private def fillOte(df: DataFrame, keys: Seq[String]): DataFrame = {
    keys.foldLeft(df) { (acc, k) =>
      val c = "ote_" + k
      acc.withColumn(c, coalesce(col(c), lit(0.1)))
    }
  }

  def encodeOne(
      trK: Array[String],
      y: Array[Double],
      apK: Array[String],
      prior: Double,
      m: Double,
      nPerm: Int,
      seed: Long
  ): (Array[Double], Array[Double]) = {
    val n = trK.length
    val acc = Array.fill(n)(0.0)
    val rng = new Random(seed)
    var p = 0
    while (p < nPerm) {
      val order = rng.shuffle(Vector.range(0, n))
      val sum = mutable.HashMap.empty[String, Double]
      val cnt = mutable.HashMap.empty[String, Int]
      var t = 0
      while (t < order.length) {
        val i = order(t)
        val k = trK(i)
        val s = sum.getOrElse(k, 0.0)
        val c = cnt.getOrElse(k, 0)
        acc(i) += (s + prior * m) / (c + m)
        sum.update(k, s + y(i))
        cnt.update(k, c + 1)
        t += 1
      }
      p += 1
    }
    val trE = acc.map(_ / nPerm.toDouble)
    val sum = mutable.HashMap.empty[String, Double]
    val cnt = mutable.HashMap.empty[String, Int]
    var i = 0
    while (i < n) {
      val k = trK(i)
      sum.update(k, sum.getOrElse(k, 0.0) + y(i))
      cnt.update(k, cnt.getOrElse(k, 0) + 1)
      i += 1
    }
    val apE = apK.map { k =>
      val c = cnt.getOrElse(k, 0)
      if (c == 0) prior else (sum(k) + prior * m) / (c + m)
    }
    (trE, apE)
  }
}
