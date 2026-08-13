package claim

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import java.io.{File, PrintWriter}
import java.nio.charset.StandardCharsets

/**
 * Rank-fuse CatBoost teacher (and optional Spark OOF) without retraining trees.
 * Picks cb_w62 vs cb_teacher by honest metrics AUC, then max2 vs 0.62/0.38.
 */
object BlendApp {

  def main(args: Array[String]): Unit = {
    sys.props("SPARK_LOCAL_IP") = "127.0.0.1"
    sys.props("spark.driver.host") = "127.0.0.1"
    val dataDir = if (args.length > 0) args(0) else "/workspace/data"
    val outDir = if (args.length > 1) args(1) else "/workspace/submissions"

    val spark = SparkSession.builder()
      .appName("spark-claim-blend")
      .master("local[2]")
      .config("spark.driver.memory", "4g")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val testIds = TrainApp.readIds(s"$dataDir/test.csv")
    val (oofPath, tesPath, tag) = pickTeacher()
    println(s"[blend] teacher=$tag oof=$oofPath test=$tesPath")

    val oof = spark.read.parquet(oofPath)
      .withColumn("id", col("id").cast(StringType))
      .withColumn("label", col("label").cast(DoubleType))
    val tes = spark.read.parquet(tesPath)
      .withColumn("id", col("id").cast(StringType))

    val trainDays = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(s"$dataDir/train.csv")
      .select(col("id").cast(StringType).as("id"), col("days"))
    val testDays = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(s"$dataDir/test.csv")
      .select(col("id").cast(StringType).as("id"), col("days"))

    val hasFuse = oof.columns.contains("pred_fuse") && tes.columns.contains("pred_fuse")
    val (ungated, tesBlend, aucW62, aucMax2, useMax2) =
      if (hasFuse) {
        val oofF = oof.join(trainDays, Seq("id"), "left").withColumn("blend", col("pred_fuse"))
        val tesF = tes.join(testDays, Seq("id"), "left").withColumn("blend", col("pred_fuse"))
        val a = TrainApp.auc(oofF, "label", "blend")
        println(f"[blend] using pred_fuse ungated=$a%.5f")
        (oofF, tesF, a, a, true)
      } else {
        val oofR = TrainApp.addRanks(oof.join(trainDays, Seq("id"), "left"), Seq("pred_cb_main", "pred_cb_alt"))
        val tesR = TrainApp.addRanks(tes.join(testDays, Seq("id"), "left"), Seq("pred_cb_main", "pred_cb_alt"))
        val w62Oof = TrainApp.applyBlend(oofR, Map("pred_cb_main" -> 0.62, "pred_cb_alt" -> 0.38))
        val max2Oof = TrainApp.applyMax2(oofR, Seq("pred_cb_main", "pred_cb_alt"))
        val aW = TrainApp.auc(w62Oof, "label", "blend")
        val aM = TrainApp.auc(max2Oof, "label", "blend")
        val useM = aM >= aW
        val u = if (useM) max2Oof else w62Oof
        val t =
          if (useM) TrainApp.applyMax2(tesR, Seq("pred_cb_main", "pred_cb_alt"))
          else TrainApp.applyBlend(tesR, Map("pred_cb_main" -> 0.62, "pred_cb_alt" -> 0.38))
        (u, t, aW, aM, useM)
      }
    val gatedOof = InsurerGate.onScore(ungated)
    val aucGated = TrainApp.auc(gatedOof, "label", "blend")
    println(f"[blend] oof w62=$aucW62%.5f max2=$aucMax2%.5f gated=$aucGated%.5f selected=${if (hasFuse) "pred_fuse" else if (useMax2) "max2" else "w62"}+insurer_gate")
    val tesGated = InsurerGate.onScore(tesBlend)
    val tesSubmit = TrainApp.addRanks(tesGated, Seq("blend")).withColumn("blend", col("r__blend"))
    val predMap = tesSubmit.select(col("id"), col("blend")).collect().map { r =>
      r.getString(0) -> (if (r.isNullAt(1) || r.getDouble(1).isNaN) 0.5 else r.getDouble(1))
    }.toMap

    new File(outDir).mkdirs()
    val outPath = s"$outDir/submission.csv"
    val pw = new PrintWriter(new File(outPath), StandardCharsets.UTF_8.name())
    try {
      pw.println("id,label")
      testIds.foreach { id =>
        pw.println(f"$id,${predMap.getOrElse(id, 0.5)}%.10f")
      }
    } finally pw.close()

    val report =
      s"""sparkml_blendapp
         |teacher=$tag
         |n_test=${testIds.length}
         |mapped=${predMap.size}
         |auc_w62=$aucW62
         |auc_max2=$aucMax2
         |auc_insurer_gate=$aucGated
         |selected=${if (hasFuse) "pred_fuse" else if (useMax2) "cb_max2" else "cb_w62"}+insurer_gate
         |auc_blend=$aucGated
         |""".stripMargin
    java.nio.file.Files.write(
      java.nio.file.Paths.get(s"$outDir/oof_report.txt"),
      report.getBytes(StandardCharsets.UTF_8)
    )
    println(s"[blend] wrote $outPath")
    println(report)
    spark.stop()
  }

  def pickTeacher(): (String, String, String) = {
    val dirs = Seq(
      new File("submissions"),
      new File("../submissions"),
      new File("/workspace/submissions")
    ).filter(_.isDirectory)
    def f(d: File, name: String): File = new File(d, name)
    val packs = Seq("final_best", "cb_w62", "cb_teacher")
    var bestAuc = -1.0
    var best: (String, String, String) = ("", "", "none")
    dirs.foreach { d =>
      packs.foreach { stem =>
        val oofF = f(d, s"${stem}_oof.parquet")
        val tesF = f(d, s"${stem}_test.parquet")
        if (oofF.isFile && tesF.isFile) {
          val met = f(d, s"${stem}_metrics.json")
          val a = if (met.isFile) metricsAuc(met) else if (stem == "final_best") 0.696 else if (stem.contains("w62")) 0.685 else 0.691
          if (a > bestAuc + 1e-12) {
            bestAuc = a
            best = (oofF.getAbsolutePath, tesF.getAbsolutePath, stem)
          }
        }
      }
    }
    if (best._3 == "none") {
      val d = dirs.headOption.getOrElse(new File("submissions"))
      (f(d, "cb_teacher_oof.parquet").getAbsolutePath, f(d, "cb_teacher_test.parquet").getAbsolutePath, "cb_teacher")
    } else best
  }

  private def metricsAuc(f: File): Double = {
    try {
      val txt = new String(java.nio.file.Files.readAllBytes(f.toPath), StandardCharsets.UTF_8)
      def grab(key: String): Option[Double] = {
        val pat = raw""""$key"\s*:\s*([0-9.]+)""".r
        pat.findFirstMatchIn(txt).map(_.group(1).toDouble)
      }
      grab("auc_gated").orElse(grab("auc_cb_w62")).orElse(grab("auc_blend")).orElse(grab("auc_cb_main")).getOrElse(0.0)
    } catch {
      case _: Throwable => 0.0
    }
  }
}
