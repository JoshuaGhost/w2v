HADOOP_HOME=/opt/cloudera/parcels/CDH/
HADOOP_STREAM_JAR=${HADOOP_HOME}/jars/hadoop-streaming-2.6.0-mr1-cdh5.12.0.jar
HADOOP_BIN=${HADOOP_HOME}/bin/hadoop
virtualenv env
source env/bin/activate
pip install gensim
cd env
zip -r ../env.zip *
cd ..
hdfs dfs -rm -r output;
$HADOOP_BIN jar $HADOOP_STREAM_JAR\
           -archives env.zip#env\
           -file mapper.py\
           -mapper mapper.py\
           -file reducer.py\
           -reducer reducer.py\
           -file upload/*\
           -file ./filename.txt\
           -input filename.txt\
           -output output;
rm -r output;
hdfs dfs -get output
