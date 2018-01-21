HOSTAME=$(uname -n)
tempPath=$PATH
LOCAL_OUTPUT='output/sampling'

if [ $HOSTNAME == "Watchdog" ]; then
    HADOOP_HOME=/usr/local/hadoop/
    HADOOP_STREAM_JAR=${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar
elif [ $HOSTNAME == "watchdog" ]; then
    HADOOP_STREAM_JAR=${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar
    PATH=${PATH}:$HADOOP_HOME/bin
else
    HADOOP_HOME=/opt/cloudera/parcels/CDH/
    HADOOP_STREAM_JAR=${HADOOP_HOME}/jars/hadoop-streaming*mr*.jar
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

for file in {filenames.txt,sampling.article.*.txt,mapper.py,reducer.py}; do
    hdfs dfs -test -e $file
    if [ $? -ne 0 ]; then
        hdfs dfs -put $file;
    fi;
done

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r output;
fi

echo $HADOOP_HOME
echo $HADOOP_STREAM_JAR

hadoop jar $HADOOP_STREAM_JAR\
    -archives ./env.zip#env\
    -file filenames.txt\
    -file mapper.py\
    -file reducer.py\
    -file sampling.article.*.txt\
    -input filenames.txt\
    -mapper mapper.py\
    -reducer reducer.py\
    -output output; 1>>mapred_std_out.txt 2>>mapred_std_err.txt

if [ -d output ]; then
    rm -r output;
fi

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    mkdir -pv $LOCAL_OUTPUT
    hdfs dfs -get output $LOCAL_OUTPUT
fi
PATH=$tempPath
