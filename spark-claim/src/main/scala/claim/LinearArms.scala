package claim

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{FMRegressor, LinearRegression}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Low-correlation RMSE arms: Ridge GLM (explicit crosses) and FM (pairwise on low-card). */
object LinearArms {

  def fitLr(train: DataFrame, applyDf: DataFrame, numeric: Seq[String], lowCats: Seq[String], crossCats: Seq[String]): DataFrame = {
    try {
      val t0 = System.nanoTime()
      val trainF = FeatureEngine.sanitize(train, numeric)
      val applyF = FeatureEngine.sanitize(applyDf, numeric)
      val cats = (lowCats ++ crossCats).distinct.filter(c => trainF.columns.contains(c))
      val idxOut = cats.map(_ + "_lr_idx")
      val oheOut = cats.map(_ + "_lr_ohe")
      val indexers = cats.zip(idxOut).map { case (c, o) =>
        new StringIndexer().setInputCol(c).setOutputCol(o).setHandleInvalid("keep")
      }
      val ohe = new OneHotEncoder()
        .setInputCols(idxOut.toArray)
        .setOutputCols(oheOut.toArray)
        .setHandleInvalid("keep")
        .setDropLast(true)
      val assembler = new VectorAssembler()
        .setInputCols((numeric ++ oheOut).toArray)
        .setOutputCol("lr_features")
        .setHandleInvalid("keep")
      val lr = new LinearRegression()
        .setLabelCol("label")
        .setFeaturesCol("lr_features")
        .setPredictionCol("pred_lr")
        .setLoss("squaredError")
        .setSolver("auto")
        .setRegParam(2.0)
        .setElasticNetParam(0.0)
        .setStandardization(true)
        .setFitIntercept(true)
        .setMaxIter(200)
        .setTol(1e-5)
      val out = new Pipeline().setStages((indexers :+ ohe :+ assembler :+ lr).toArray)
        .fit(trainF).transform(applyF)
      println(f"[lr] ${(System.nanoTime()-t0)/1e9}%.1fs")
      out.withColumn("pred_lr", FeatureEngine.finiteFill(col("pred_lr"), 0.1))
        .select(col("id"), col("pred_lr"))
    } catch {
      case e: Exception =>
        println("[lr] failed: " + e.getMessage)
        applyDf.select(col("id")).withColumn("pred_lr", lit(0.1))
    }
  }

  def fitFm(train: DataFrame, applyDf: DataFrame, numeric: Seq[String], cats: Seq[String]): DataFrame = {
    try {
      val t0 = System.nanoTime()
      val trainF = FeatureEngine.sanitize(train, numeric)
      val applyF = FeatureEngine.sanitize(applyDf, numeric)
      val useCats = cats.filter(c => trainF.columns.contains(c))
      val idxOut = useCats.map(_ + "_fm_idx")
      val oheOut = useCats.map(_ + "_fm_ohe")
      val indexers = useCats.zip(idxOut).map { case (c, o) =>
        new StringIndexer().setInputCol(c).setOutputCol(o).setHandleInvalid("keep")
      }
      val ohe = new OneHotEncoder()
        .setInputCols(idxOut.toArray)
        .setOutputCols(oheOut.toArray)
        .setHandleInvalid("keep")
        .setDropLast(true)
      val assembler = new VectorAssembler()
        .setInputCols((numeric ++ oheOut).toArray)
        .setOutputCol("fm_raw")
        .setHandleInvalid("keep")
      val scaler = new StandardScaler()
        .setInputCol("fm_raw")
        .setOutputCol("fm_features")
        .setWithMean(false)
        .setWithStd(true)
      val fm = new FMRegressor()
        .setLabelCol("label")
        .setFeaturesCol("fm_features")
        .setPredictionCol("pred_fm")
        .setFactorSize(8)
        .setFitIntercept(true)
        .setFitLinear(true)
        .setRegParam(0.2)
        .setMaxIter(60)
        .setStepSize(0.03)
        .setMiniBatchFraction(1.0)
        .setSolver("adamW")
        .setSeed(2026L)
      val out = new Pipeline().setStages((indexers :+ ohe :+ assembler :+ scaler :+ fm).toArray)
        .fit(trainF).transform(applyF)
      println(f"[fm] ${(System.nanoTime()-t0)/1e9}%.1fs")
      out.withColumn("pred_fm", FeatureEngine.finiteFill(col("pred_fm"), 0.1))
        .select(col("id"), col("pred_fm"))
    } catch {
      case e: Exception =>
        println("[fm] failed: " + e.getMessage)
        applyDf.select(col("id")).withColumn("pred_fm", lit(0.1))
    }
  }
}
