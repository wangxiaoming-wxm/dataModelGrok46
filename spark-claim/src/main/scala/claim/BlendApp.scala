package claim

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import java.io.{File, PrintWriter}
import java.nio.charset.StandardCharsets

/**
 * Rank-fuse CatBoost teacher (and optional Spark OOF) without retraining trees.
 * Prefers cb_w62 parquet, falls back to cb_teacher.
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

    val oofR = TrainApp.addRanks(oof, Seq("pred_cb_main", "pred_cb_alt"))
    val tesR = TrainApp.addRanks(tes, Seq("pred_cb_main", "pred_cb_alt"))
    val w62Oof = TrainApp.applyBlend(oofR, Map("pred_cb_main" -> 0.62, "pred_cb_alt" -> 0.38))
    val max2Oof = TrainApp.applyMax2(oofR, Seq("pred_cb_main", "pred_cb_alt"))
    val aucW62 = TrainApp.auc(w62Oof, "label", "blend")
    val aucMax2 = TrainApp.auc(max2Oof, "label", "blend")
    val useMax2 = aucMax2 >= aucW62
    println(f"[blend] oof w62=$aucW62%.5f max2=$aucMax2%.5f selected=${if (useMax2) "max2" else "w62"}")

    val tesBlend =
      if (useMax2) TrainApp.applyMax2(tesR, Seq("pred_cb_main", "pred_cb_alt"))
      else TrainApp.applyBlend(tesR, Map("pred_cb_main" -> 0.62, "pred_cb_alt" -> 0.38))
    val predMap = tesBlend.select(col("id"), col("blend")).collect().map { r =>
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
         |selected=${if (useMax2) "cb_max2" else "cb_w62"}
         |auc_blend=${if (useMax2) aucMax2 else aucW62}
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
    val w62o = new File("/workspace/submissions/cb_w62_oof.parquet")
    val w62t = new File("/workspace/submissions/cb_w62_test.parquet")
    val w62m = new File("/workspace/submissions/cb_w62_metrics.json")
    if (w62o.isFile && w62t.isFile && w62m.isFile)
      (w62o.getAbsolutePath, w62t.getAbsolutePath, "cb_w62")
    else
      (
        "/workspace/submissions/cb_teacher_oof.parquet",
        "/workspace/submissions/cb_teacher_test.parquet",
        "cb_teacher"
      )
  }
}
