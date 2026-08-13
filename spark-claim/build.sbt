name := "spark-claim"
version := "1.0.0"
scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.5",
  "org.apache.spark" %% "spark-sql" % "3.5.5",
  "org.apache.spark" %% "spark-mllib" % "3.5.5",
  ("ai.catboost" % "catboost-spark_3.5_2.12" % "1.2.10")
    .exclude("org.apache.spark", "spark-core_2.12")
    .exclude("org.apache.spark", "spark-sql_2.12")
    .exclude("org.apache.spark", "spark-mllib_2.12")
    .exclude("org.apache.spark", "spark-catalyst_2.12")
    .exclude("org.apache.spark", "spark-tags_2.12")
)

run / fork := true
Compile / packageBin / exportJars := true

val jdkOpens = Seq(
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
  "--add-opens=java.base/java.io=ALL-UNNAMED",
  "--add-opens=java.base/java.net=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
  "--add-opens=java.base/sun.security.action=ALL-UNNAMED"
)

run / javaOptions ++= Seq(
  "-Xmx12g",
  "-Dspark.master=local[4]",
  "-Dspark.driver.host=127.0.0.1",
  "-DSPARK_LOCAL_IP=127.0.0.1"
) ++ jdkOpens

javaOptions ++= jdkOpens
