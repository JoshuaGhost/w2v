HADOOP_HOME=/opt/cloudera/parcels/CDH/
HADOOP_STREAM_JAR=${HADOOP_HOME}/jars/hadoop-streaming-2.6.0-mr1-cdh5.13.0.jar
HADOOP_BIN=${HADOOP_HOME}/bin/hadoop

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r output;
fi

hdfs dfs -put test_mapper.py
hdfs dfs -put test_reducer.py

$HADOOP_BIN jar $HADOOP_STREAM_JAR\
            -archives env.zip#env\
            -file filenames.txt\
            -input filenames.txt\
            -file test_mapper.py\
            -file test_reducer.py\
            -mapper test_mapper.py\
            -reducer test_reducer.py\
            -output output;

if [ -d output ]; then
    rm -r output;
fi

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    hdfs dfs -get output
fi

