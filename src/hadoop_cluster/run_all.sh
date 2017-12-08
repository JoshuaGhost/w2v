HADOOP_HOME=/opt/cloudera/parcels/CDH/
HADOOP_STREAM_JAR=${HADOOP_HOME}/jars/hadoop-streaming-2.6.0-mr1-cdh5.13.0.jar
HADOOP_BIN=${HADOOP_HOME}/bin/hadoop
if [ ! -f env.zip ]; then
    virtualenv env
    source env/bin/activate
    pip install gensim
    cd env
    zip -r ../env.zip *
    cd ..
fi

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r output;
fi

hdfs dfs -rm -r {filenames.txt,text*,mapper.py,reducer.py}

hdfs dfs -put filenames.txt
hdfs dfs -put upload/*
hdfs dfs -put mapper.py
hdfs dfs -put reducer.py

$HADOOP_BIN jar $HADOOP_STREAM_JAR\
           -archives env.zip#env\
           -file mapper.py\
           -mapper mapper.py\
           -file reducer.py\
           -reducer reducer.py\
           -file upload/*\
           -file ./filenames.txt\
           -input filenames.txt\
           -output output;

if [ -d output ]; then
    rm -r output;
fi

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    hdfs dfs -get output
fi

