HOSTAME=$(uname -n)
tempPath=$PATH
if [ $HOSTNAME == "Watchdog" ]; then
    HADOOP_HOME=/usr/local/hadoop/
    HADOOP_STREAM_JAR=${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar
elif [ $HOSTNAME == "watchdog" ]; then
    HADOOP_STREAM_JAR=${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar
    PATH=${PATH}:$HADOOP_HOME/bin
else
    HADOOP_HOME=/opt/cloudera/parcels/CDH/
    HADOOP_STREAM_JAR=${HADOOP_HOME}/jars/*hadoop-streaming*.jar
fi
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

#hdfs dfs -rm -r {filenames.txt,text*,mapper.py,reducer.py}

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
PATH=$tempPath
